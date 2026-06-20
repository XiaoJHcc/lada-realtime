# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
Clock-driven realtime AppSrc.

Unlike the buffer-first FrameRestorerAppSrc (lada/gui/watch/gstreamer_pipeline_appsrc.py)
which blocks on the FrameRestorer's output queue and lets the pipeline pause+buffer when
the GPU can't keep up, this AppSrc never blocks waiting for AI frames:

- A PassthroughFrameRestorer decodes the original video sequentially. This is the master
  beat: pushing it is throttled by GStreamer downstream queue backpressure + the sink's
  clock (buffer PTS vs pipeline clock), so playback advances at wall-clock speed and the
  sink never starves -> no stutter.
- The AI FrameRestorer is drained NON-blocking each beat. Restored frames are matched to
  the current passthrough PTS. If a restored frame for the current PTS is ready we push it,
  otherwise we immediately push the original (passthrough) frame and the late AI frame is
  discarded once playback has moved past it.

Milestone 1 goal: prove "playback never stalls + passthrough fallback when AI isn't ready".
It does not yet add the low-latency knobs (small clip window, downscaling, frame skipping)
needed to make the AI actually keep up in realtime.
"""

import logging
import queue
import threading
import time

import torch
from gi.repository import Gst, GstApp, GObject

from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerProvider, PassthroughFrameRestorer
from lada.gui.watch.gstreamer_pipeline_appsrc import GstPaddingHelpers
from lada.utils import video_utils, VideoMetadata, threading_utils
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, StopMarker, EofMarker, ErrorMarker

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


class RealtimeFrameRestorerAppSrc(GstApp.AppSrc):
    GST_PLUGIN_NAME = 'realtimeframerestorerappsrc'

    __gstmetadata__ = ('RealtimeFrameRestorerAppSrc', 'Src', 'Clock-driven realtime FrameRestorer AppSrc element', 'Lada Authors')

    __gsttemplates__ = (
        Gst.PadTemplate.new("src",
                            Gst.PadDirection.SRC,
                            Gst.PadPresence.ALWAYS,
                            Gst.Caps.new_any()),
    )

    __gproperties__ = {
        "frame-restorer-provider": (GObject.TYPE_PYOBJECT,
                          "FrameRestorerProvider",
                          "Frame restorer provider object to get a FrameRestorer instance",
                          GObject.ParamFlags.READWRITE
                          ),
        "video-metadata": (GObject.TYPE_PYOBJECT,
                          "VideoMetadata",
                          "Metadata of the video file that should be restored by FrameRestorer",
                          GObject.ParamFlags.READWRITE
                          )
    }

    def __init__(self):
        super().__init__()

        self.video_metadata: VideoMetadata | None = None
        self.cpu_frame: torch.Tensor | None = None

        self.frame_restorer: FrameRestorer | None = None
        self.passthrough_restorer: PassthroughFrameRestorer | None = None
        self.frame_restorer_provider: FrameRestorerProvider | None = None
        self.frame_restorer_lock: threading.Lock = threading.Lock()

        # restored AI frames that arrived but whose PTS we haven't reached yet, keyed by PTS
        self.ai_ready_frames: dict[int, torch.Tensor] = {}
        self.ai_eof: bool = False

        self.appsource_thread: threading.Thread | None = None
        self.appsource_thread_should_be_running: bool = False
        self.appsource_thread_stop_requested = False
        self.appsource_thread_shutdown_requested = False
        self.appsource_thread_eof = False

        self.appsrc_lock: threading.Lock = threading.Lock()

        self.frame_duration_ns: float = 0
        self.current_timestamp_ns = 0

        self.set_property('is-live', False)
        self.set_property('emit-signals', True)
        self.set_property('stream-type', GstApp.AppStreamType.SEEKABLE)
        self.set_property('format', Gst.Format.TIME)
        self.set_property('max-buffers', 5)
        self.set_property('max-bytes', 0)
        self.set_property('block', False)

        self.connect('need-data', self._on_need_data)
        self.connect('enough-data', self._on_enough_data)
        self.connect('seek-data', self._on_seek_data)

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'video-metadata':
            return self.video_metadata
        elif prop.name == 'frame-restorer-provider':
            return self.frame_restorer_provider
        else:
            return super().do_get_property(prop)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'video-metadata':
            self.appsource_thread_eof = False
            if self.video_metadata is None:
                self._set_video_metadata(value)
            else:
                with self.appsrc_lock:
                    should_start = self.appsource_thread is not None and not self.appsource_thread_stop_requested
                    self._stop_appsource_worker()
                    self.current_timestamp_ns = 0
                    self._set_video_metadata(value)
                    if should_start:
                        self._start_appsource_worker()
        elif prop.name == 'frame-restorer-provider':
            self.frame_restorer_provider = value
        else:
            super().do_set_property(prop, value)

    def do_state_changed(self, oldstate: Gst.State, newstate: Gst.State, pending: Gst.State) -> None:
        logger.debug(f"realtime appsource state change: {oldstate.name} -> {newstate.name} (pending: {pending.name})")
        if oldstate == Gst.State.READY and newstate == Gst.State.NULL:
            self._stop_appsource_worker(shutdown=True)
        elif oldstate == Gst.State.NULL and newstate == Gst.State.READY:
            self.appsource_thread_shutdown_requested = False

    def _set_video_metadata(self, video_metadata: VideoMetadata):
        self.video_metadata = video_metadata
        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={GstPaddingHelpers.get_padded_width(self.video_metadata.video_width)},height={self.video_metadata.video_height},framerate={self.video_metadata.video_fps_exact.numerator}/{self.video_metadata.video_fps_exact.denominator}")
        self.set_property('caps', caps)
        self.set_property('duration', int((self.video_metadata.frames_count * self.frame_duration_ns)))
        logger.debug(f"realtime appsource set video metadata: {video_metadata.video_file}")

    def _on_need_data(self, src, length):
        logger.debug("realtime appsource need-data")
        with self.appsrc_lock:
            self._start_appsource_worker()
        return True

    def _on_enough_data(self, src):
        logger.debug("realtime appsource enough-data")
        with self.appsrc_lock:
            self._request_stop_appsource_worker()
        return True

    def _on_seek_data(self, appsrc, offset_ns):
        logger.debug(f"realtime appsource seek: offset (sec): {offset_ns / Gst.SECOND}, current position (sec): {self.current_timestamp_ns / Gst.SECOND}")
        with self.appsrc_lock:
            if offset_ns == self.current_timestamp_ns:
                logger.debug("realtime appsource seek: skipped seek as we're already at the seek position")
                return True
            if self.appsource_thread_shutdown_requested:
                logger.debug("realtime appsource seek: skipped seek as shutdown was requested.")
                return True
            self.appsource_thread_eof = False
            self._stop_appsource_worker()
            self._start_appsource_worker(seek_position=offset_ns)
        return True

    def _start_appsource_worker(self, seek_position=None):
        with self.frame_restorer_lock:
            if self.appsource_thread_shutdown_requested:
                logger.debug("realtime appsource worker: requested to start but shutdown was requested. Will not start")
                return
            if self.appsource_thread_eof:
                logger.debug("realtime appsource worker: requested to start but EOF. Will not start")
                return
            self.appsource_thread_stop_requested = False
            self.appsource_thread_should_be_running = True

            if self.appsource_thread and self.appsource_thread.is_alive():
                logger.debug("realtime appsource worker: requested to start but already started")
                return

            if seek_position:
                assert self.appsource_thread is None, "starting realtime appsource worker with pending timestamp but worker is still running"
                assert self.frame_restorer is None, "starting realtime appsource worker with pending timestamp but frame restorer is still running"
                assert self.passthrough_restorer is None, "starting realtime appsource worker with pending timestamp but passthrough restorer is still running"

            if not self.frame_restorer:
                logger.debug("realtime appsource worker: setting up frame restorer + passthrough source")
                start_ns = int(seek_position) if seek_position is not None else int(self.current_timestamp_ns)
                self.ai_ready_frames = {}
                self.ai_eof = False
                self.frame_restorer = self.frame_restorer_provider.get()
                self.passthrough_restorer = PassthroughFrameRestorer(self.video_metadata.video_file)
                self.frame_restorer.start(start_ns=start_ns)
                self.passthrough_restorer.start(start_ns=start_ns)
                self.current_timestamp_ns = start_ns

            self.appsource_thread = threading.Thread(target=self._appsource_worker)
            self.appsource_thread.start()

    def _request_stop_appsource_worker(self):
        with self.frame_restorer_lock:
            self.appsource_thread_stop_requested = True
            self.appsource_thread_should_be_running = False

    def _stop_appsource_worker(self, shutdown=False):
        with self.frame_restorer_lock:
            start = time.time()
            if shutdown:
                logger.debug("realtime appsource worker: shutdown requested")
                self.appsource_thread_shutdown_requested = True
            self.appsource_thread_stop_requested = True
            self.appsource_thread_should_be_running = False

            ai_queue = None
            if self.frame_restorer:
                logger.debug("realtime appsource worker: stopping frame restorer")
                self.frame_restorer.stop()
                ai_queue = self.frame_restorer.get_frame_restoration_queue()
                # unblock consumer (worker only drains non-blocking, but be consistent with stop handshake)
                threading_utils.put_queue_stop_marker(ai_queue)

            if self.appsource_thread:
                self.appsource_thread.join()
                logger.debug("realtime appsource worker: joined appsource_thread")
                self.appsource_thread = None

            if self.frame_restorer:
                threading_utils.empty_out_queue(ai_queue)
                self.frame_restorer = None

            if self.passthrough_restorer:
                self.passthrough_restorer.stop()
                self.passthrough_restorer = None

            self.ai_ready_frames = {}

            logger.debug(f"realtime appsource worker: stopped, took {time.time() - start}")

    def _appsource_worker(self):
        logger.debug("realtime appsource worker: started")
        marker = None
        while self.appsource_thread_should_be_running:
            marker = self._get_next_frame_and_push_buffer()
            if marker is EOF_MARKER:
                self.appsource_thread_should_be_running = False
                self.appsource_thread_eof = True
                self.emit("end-of-stream")
            elif marker is STOP_MARKER:
                self.appsource_thread_should_be_running = False
                if not self.appsource_thread_stop_requested:
                    logger.warning("realtime appsource worker: Invalid state. Received stop marker but not requested to shutdown")
        if marker is EOF_MARKER:
            logger.debug("realtime appsource worker: stopped itself, EOF")
        elif marker is STOP_MARKER:
            logger.debug("realtime appsource worker: stopped by request")

    def _drain_ai_queue(self):
        """Non-blocking: move all currently available restored frames into ai_ready_frames."""
        if self.ai_eof:
            return
        ai_queue = self.frame_restorer.get_frame_restoration_queue()
        while True:
            try:
                elem = ai_queue.get(block=False)
            except queue.Empty:
                return
            if elem is EOF_MARKER:
                self.ai_eof = True
                return
            if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                # AI side stopped/crashed; keep playing passthrough, stop draining
                self.ai_eof = True
                if isinstance(elem, ErrorMarker):
                    logger.error(f"realtime appsource worker: AI frame restorer crashed, continuing with passthrough only: {elem}")
                return
            ai_frame, ai_pts = elem
            self.ai_ready_frames[int(ai_pts)] = ai_frame

    def _pick_frame(self, passthrough_frame: torch.Tensor, pts: int) -> torch.Tensor:
        """Use the restored AI frame for this PTS if ready, else fall back to original frame.
        Discards any restored frames whose PTS playback has already passed."""
        self._drain_ai_queue()
        ai_frame = self.ai_ready_frames.pop(pts, None)
        # prune stale restored frames the clock has already moved past (would never be shown)
        if self.ai_ready_frames:
            stale = [k for k in self.ai_ready_frames if k < pts]
            for k in stale:
                del self.ai_ready_frames[k]
        return ai_frame if ai_frame is not None else passthrough_frame

    def _get_next_frame_and_push_buffer(self) -> StopMarker | EofMarker | None:
        # master beat: read the next original frame sequentially (throttled by downstream
        # queue backpressure + sink clock). Never blocks on the AI source.
        result = self.passthrough_restorer.get_frame_restoration_queue().get()
        if self.appsource_thread_stop_requested:
            logger.debug("realtime appsource worker: passthrough consumer unblocked by stop")
            return STOP_MARKER
        if result is None:
            return EOF_MARKER

        passthrough_frame, frame_pts = result
        pts = int(frame_pts)
        frame = self._pick_frame(passthrough_frame, pts)

        frame_timestamp_ns = int((frame_pts * self.video_metadata.time_base) * Gst.SECOND)
        frame = GstPaddingHelpers.pad_frame(frame)
        device_type = frame.device.type
        if device_type in ('cuda', 'xpu', 'mps'):
            use_async_copy = device_type in ('cuda', 'xpu')
            if self.cpu_frame is None or frame.shape != self.cpu_frame.shape:
                self.cpu_frame = torch.empty((frame.shape[0], frame.shape[1], frame.shape[2]), dtype=frame.dtype, device='cpu', pin_memory=use_async_copy)
            self.cpu_frame.copy_(frame, non_blocking=use_async_copy)
            if use_async_copy:
                if device_type == 'cuda':
                    torch.cuda.synchronize()
                else:
                    torch.xpu.synchronize()
            data = self.cpu_frame.numpy().tobytes()
        else:
            data = frame.numpy().tobytes()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = round(self.frame_duration_ns)
        buf.pts = frame_timestamp_ns
        buf.offset = video_utils.offset_ns_to_frame_num(frame_timestamp_ns, self.video_metadata.video_fps_exact)
        self.emit('push-buffer', buf)
        self.current_timestamp_ns = frame_timestamp_ns

        return None


GObject.type_register(RealtimeFrameRestorerAppSrc)
__gstelementfactory__ = (RealtimeFrameRestorerAppSrc.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, RealtimeFrameRestorerAppSrc)
