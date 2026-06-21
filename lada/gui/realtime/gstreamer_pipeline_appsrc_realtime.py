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

Clip-based scheduling: the AI restorer is started at playhead + clip (one realtime clip
ahead), so the clip the playhead currently sits in is abandoned (too late to serve) and the
NEXT clip is ready by the time the clock reaches it. When the AI output head falls behind the
playhead (GPU couldn't keep up, or the user seeked), the in-flight AI work is discarded and
the restorer is repositioned to playhead + 2*clip (one extra clip of headroom to absorb the
stop/start/first-frame restart cost). Seek and "GPU fell behind" thus share the same re-aim
rule. The reposition runs on a background thread and never blocks the play loop: while the AI
restorer is down/swapping, the loop falls back to passthrough.
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

# Realtime lookahead tuning. The AI is allowed to process up to playhead + lookahead window,
# building a lead of restored frames during easy/empty stretches to spend on hard stretches.
# The lookahead window is user-configurable (realtime_lookahead_frames). The output
# queue must be large enough to actually hold that lead, else it backpressures and caps the
# lead below the window. Frames are CPU tensors, so this is host RAM, not VRAM; capped at
# REALTIME_FRAME_BUFFER_MAX_BYTES so high-res videos don't exhaust memory.
REALTIME_FRAME_BUFFER_MAX_BYTES = 3 * 1024 * 1024 * 1024  # 3 GiB host RAM ceiling


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

        # Short-held lock guarding the (frame_restorer, ai_ready_frames, ai_eof) triple so the
        # play loop and a background reposition can swap the AI restorer without tearing each
        # other's view. NEVER held across .stop()/.start()/provider.get() (those are slow) ->
        # the play loop only ever blocks on it for the microseconds of a pointer swap.
        self._ai_lock: threading.Lock = threading.Lock()

        # restored AI frames that arrived but whose PTS we haven't reached yet, keyed by PTS
        self.ai_ready_frames: dict[int, torch.Tensor] = {}
        self.ai_eof: bool = False

        # Reposition (clip-based re-aim) state. When the AI output head falls behind the
        # playhead, a background thread stops the old restorer and starts a fresh one ahead of
        # the playhead. _reposition_lock guards this lifecycle; the generation counter lets a
        # stop/seek cancel an in-flight reposition. Never held across a join/stop.
        self._reposition_lock: threading.Lock = threading.Lock()
        self._reposition_thread: threading.Thread | None = None
        self._reposition_gen: int = 0
        self._reposition_in_progress: bool = False

        # Lightweight realtime diagnostics. Updated in the hot push loop without extra
        # threads or GPU calls. Read via get_stats() from a GLib timeout in the view.
        self.stats_lock: threading.Lock = threading.Lock()
        self._reset_stats_locked()

        self.appsource_thread: threading.Thread | None = None
        self.appsource_thread_should_be_running: bool = False
        self.appsource_thread_stop_requested = False
        self.appsource_thread_shutdown_requested = False
        self.appsource_thread_eof = False

        self.appsrc_lock: threading.Lock = threading.Lock()

        self.frame_duration_ns: float = 0
        self.current_timestamp_ns = 0

        # Realtime clip length (frames). The AI restorer is started one clip ahead of the
        # playhead (and re-aimed two clips ahead when it falls behind). This is the realtime-only
        # clip length (config realtime_clip_length), separate from max_clip_duration which drives
        # watch preview + CLI export. User tunable via config; set by the view before (re)start.
        self.clip_frames: int = 30

        # Realtime lookahead/buffer window (frames): how far ahead of the playhead the frontier
        # gate lets the AI work, i.e. how big a lead of restored frames it may build during easy
        # stretches. Frames (not seconds) because the lead is bounded by clip length and memory,
        # both frame-based, and fps varies. User tunable via config (UI label "Buffer window");
        # set by the view. Effective value is clamped to >= 2*clip and capped by the output
        # buffer (see _update_processing_frontier).
        self.lookahead_frames: int = 300

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
                self.reset_stats()
                self.frame_restorer = self.frame_restorer_provider.get(
                    frame_restoration_queue_max_bytes=self._compute_frame_buffer_max_bytes())
                self.passthrough_restorer = PassthroughFrameRestorer(self.video_metadata.video_file)
                # Clip-based start: the AI restorer starts one clip AHEAD of the playhead so its
                # frames are ready by the time the clock arrives. The passthrough (master beat /
                # play position) starts exactly at start_ns -> the 0..clip region shows the
                # original, then playback switches to AI output seamlessly. The clip the playhead
                # currently sits in is abandoned (too late to serve). Clamped so we never start
                # the AI past EOF.
                clip = max(1, int(self.clip_frames))
                clip_ns = int(clip * self.frame_duration_ns)
                duration_ns = int(self.video_metadata.frames_count * self.frame_duration_ns)
                ai_start_ns = min(start_ns + clip_ns, max(start_ns, duration_ns - 1))
                self.frame_restorer.start(start_ns=ai_start_ns)
                self.passthrough_restorer.start(start_ns=start_ns)
                self.current_timestamp_ns = start_ns
                # Clamp the detector immediately, before the first buffer is pushed, so the fast
                # YOLO detector can't race the whole lookahead window ahead during startup. The
                # FrameRestorer further bounds this to the AI output position internally.
                self._update_processing_frontier(video_utils.offset_ns_to_frame_num(start_ns, self.video_metadata.video_fps_exact))
                logger.debug(f"realtime appsource worker: playhead start {start_ns/Gst.SECOND:.2f}s, AI clip start {ai_start_ns/Gst.SECOND:.2f}s (+{clip} frames)")

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

            # Cancel + join any in-flight reposition before tearing down the restorer. Safe to
            # join here (we hold frame_restorer_lock): the reposition thread NEVER takes
            # frame_restorer_lock, so there's no lock cycle. After this returns frame_restorer is
            # either None (reposition detached/discarded) or a fully-started new restorer, which
            # the normal stop block below then cleans up.
            self._cancel_and_join_reposition()

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

    def _reset_stats_locked(self):
        """Reset counters. Caller must hold stats_lock (or be in __init__)."""
        self.stats_hit = 0
        self.stats_fallback = 0
        self.stats_discarded_total = 0
        self.stats_max_ai_pts = None  # highest restored-frame PTS seen (raw pts units)
        self.stats_ai_frames_drained = 0
        self.stats_ready_map_size = 0

    def reset_stats(self):
        with self.stats_lock:
            self._reset_stats_locked()

    def get_stats(self) -> dict:
        """Snapshot of realtime diagnostics counters. Cheap; safe to call from GLib timeout."""
        with self.stats_lock:
            max_ai_pts_ns = None
            if self.stats_max_ai_pts is not None and self.video_metadata is not None:
                max_ai_pts_ns = int((self.stats_max_ai_pts * self.video_metadata.time_base) * Gst.SECOND)
            total = self.stats_hit + self.stats_fallback
            stats = {
                "hit": self.stats_hit,
                "fallback": self.stats_fallback,
                "hit_rate": (self.stats_hit / total) if total > 0 else 0.0,
                "discarded_total": self.stats_discarded_total,
                "max_ai_pts_ns": max_ai_pts_ns,
                "ai_frames_drained": self.stats_ai_frames_drained,
                "ready_map_size": self.stats_ready_map_size,
                "playhead_ns": int(self.current_timestamp_ns),
                "frame_duration_ns": self.frame_duration_ns,
            }
        # Production-side throughput, read straight from the AI restorer's worker threads so
        # it reflects real GPU output even while playback is paused or falling back (the drain
        # counter above only moves when the play loop consumes a frame). frame_restorer can be
        # None between stop/seek; report None so the manager holds the last sample.
        #
        # fps is measured per batch/clip at production time (get_*_fps), NOT by differentiating
        # the monotonic *_frames_done counters here: a realtime reposition builds a fresh
        # FrameRestorer whose counters restart at 0, which a cross-sample delta reads as 0 fps
        # exactly when the GPU is busiest. output_frame_pos is the restorer's live consume
        # position, used for ahead/behind so it stays correct while playback is paused.
        with self._ai_lock:
            fr = self.frame_restorer
        if fr is not None and not isinstance(fr, PassthroughFrameRestorer):
            stats["detector_fps_live"] = fr.get_detector_fps()
            stats["restorer_fps_live"] = fr.get_restorer_fps()
            stats["output_frame_pos"] = fr.get_output_frame_pos()
            stats["ready_start_frame"] = fr.get_start_frame()
        else:
            stats["detector_fps_live"] = None
            stats["restorer_fps_live"] = None
            stats["output_frame_pos"] = None
            stats["ready_start_frame"] = None
        # Buffer-bar window length (frames): how far ahead of the playhead the AI may work.
        stats["window_frames"] = max(0, int(self.lookahead_frames))
        return stats

    def _drain_ai_queue(self):
        """Non-blocking: move all currently available restored frames into ai_ready_frames.
        Caller holds _ai_lock so a concurrent reposition can't swap frame_restorer mid-drain."""
        fr = self.frame_restorer
        if fr is None or self.ai_eof:
            return
        ai_queue = fr.get_frame_restoration_queue()
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
            with self.stats_lock:
                self.stats_ai_frames_drained += 1
                ai_pts_int = int(ai_pts)
                if self.stats_max_ai_pts is None or ai_pts_int > self.stats_max_ai_pts:
                    self.stats_max_ai_pts = ai_pts_int

    def _pick_frame(self, passthrough_frame: torch.Tensor, pts: int) -> torch.Tensor:
        """Use the restored AI frame for this PTS if ready, else fall back to original frame.
        Discards any restored frames whose PTS playback has already passed. Held under _ai_lock
        so a concurrent reposition (which resets ai_ready_frames / nulls frame_restorer) can't
        tear the drain+pop+prune view; falling back to passthrough is always safe."""
        with self._ai_lock:
            self._drain_ai_queue()
            ai_frame = self.ai_ready_frames.pop(pts, None)
            # prune stale restored frames the clock has already moved past (would never be shown)
            stale_count = 0
            if self.ai_ready_frames:
                stale = [k for k in self.ai_ready_frames if k < pts]
                for k in stale:
                    del self.ai_ready_frames[k]
                stale_count = len(stale)
            ready_map_size = len(self.ai_ready_frames)
        with self.stats_lock:
            if ai_frame is not None:
                self.stats_hit += 1
            else:
                self.stats_fallback += 1
            self.stats_discarded_total += stale_count
            self.stats_ready_map_size = ready_map_size
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

        # Drive the processing-frontier gate: let the AI pipeline work on frames up to
        # playhead + window, but no further. This keeps the GPU focused on frames near the
        # playhead (so the current clip's restoration can keep up) instead of racing
        # thousands of frames ahead on future content that gets discarded.
        self._update_processing_frontier(buf.offset)

        # If the AI output head has fallen behind the playhead, abandon the in-flight (now
        # stale) work and re-aim the restorer ahead of the playhead. No-op while a reposition
        # is already running or the AI is still ahead.
        self._maybe_reposition(buf.offset)

        return None

    def _frame_nbytes(self) -> int:
        """Approx bytes of one decoded BGR frame (host RAM)."""
        return max(1, self.video_metadata.video_width * self.video_metadata.video_height * 3)

    def _compute_frame_buffer_max_bytes(self) -> int:
        """Size the output queue to hold the full lookahead window (+ a clip margin),
        capped at the host-RAM ceiling. Returns bytes for FrameRestorer's output queue.
        Uses self.clip_frames (known before the restorer exists) so it's correct on cold
        start and when a reposition rebuilds the queue."""
        lookahead_frames = max(0, int(self.lookahead_frames))
        clip = max(1, int(self.clip_frames))
        want_frames = max(1, lookahead_frames + clip)
        want_bytes = want_frames * self._frame_nbytes()
        if want_bytes > REALTIME_FRAME_BUFFER_MAX_BYTES:
            capped_frames = REALTIME_FRAME_BUFFER_MAX_BYTES // self._frame_nbytes()
            logger.info(f"realtime appsource: frame buffer capped at {REALTIME_FRAME_BUFFER_MAX_BYTES // (1024*1024)}MB "
                        f"(~{capped_frames} frames) instead of {want_frames} for this resolution")
            return REALTIME_FRAME_BUFFER_MAX_BYTES
        return want_bytes

    def _update_processing_frontier(self, playhead_frame: int):
        with self._ai_lock:
            fr = self.frame_restorer
        if fr is None:
            return
        # Let the AI work up to playhead + window, but no further. Window (user-configured
        # "Buffer window" in frames) is clamped to:
        #   >= 2*clip   so a clip can always fill ahead of the playhead (one clip to fill +
        #               one clip of slack), matching FrameRestorer's 2*clip detector lead
        # and capped by what the output buffer can actually hold (else the feeder runs ahead
        # only to stall on a full queue).
        # NOTE: this is only the playhead-based CAP. FrameRestorer.set_processing_frontier
        # further clamps the detector to (AI output position + a clip lead), so the fast
        # detector can't race this whole window ahead and starve the slow restorer of GPU.
        # A large buffer window therefore grows the restored-frame lead (output queue)
        # without making the detector burn GPU on far-future frames.
        clip = max(1, int(self.clip_frames))
        lookahead_frames = max(0, int(self.lookahead_frames))
        window = max(lookahead_frames, clip * 2)
        buffer_frames = REALTIME_FRAME_BUFFER_MAX_BYTES // self._frame_nbytes()
        if window + clip > buffer_frames:
            window = max(clip * 2, buffer_frames - clip)
        fr.set_processing_frontier(playhead_frame + window)

    def _maybe_reposition(self, playhead_frame: int):
        """If the AI output head has fallen to/behind the playhead, abandon the in-flight work
        and re-aim the restorer at playhead + 2*clip. No-op while a reposition is already
        running (frame_restorer is None / flag set) or while the AI is still ahead.

        Anti-thrash relies on two things, not a timer: (a) a reposition in progress nulls
        frame_restorer so this returns immediately, and (b) the 2*clip headroom lands the new
        output head well ahead of the playhead, so it won't instantly re-trigger."""
        if self._reposition_in_progress:
            return
        fr = self.frame_restorer
        if fr is None or self.ai_eof:
            return
        if fr.get_output_frame_pos() > playhead_frame:
            return
        clip = max(1, int(self.clip_frames))
        self._spawn_reposition(playhead_frame + 2 * clip)

    def _spawn_reposition(self, ai_start_frame: int):
        with self._reposition_lock:
            if self._reposition_in_progress:
                return
            if self.appsource_thread_stop_requested or self.appsource_thread_shutdown_requested:
                return
            self._reposition_in_progress = True
            self._reposition_gen += 1
            gen = self._reposition_gen
            t = threading.Thread(target=self._reposition_worker, args=(ai_start_frame, gen), daemon=True)
            self._reposition_thread = t
            t.start()

    def _reposition_cancelled(self, gen: int) -> bool:
        if self.appsource_thread_stop_requested or self.appsource_thread_shutdown_requested:
            return True
        with self._reposition_lock:
            return self._reposition_gen != gen

    def _clamp_frame_to_start_ns(self, frame_num: int) -> int:
        duration_ns = int(self.video_metadata.frames_count * self.frame_duration_ns)
        start_ns = int(max(0, frame_num) * self.frame_duration_ns)
        return min(start_ns, max(0, duration_ns - 1))

    def _reposition_worker(self, ai_start_frame: int, gen: int):
        """Background re-aim. NEVER joins the play loop. Detaches + stops the old restorer
        (slow, outside _ai_lock so the play loop only ever sees the cheap pointer swap), then
        starts a fresh one ahead of the playhead and swaps it in iff still the current gen."""
        logger.debug(f"realtime reposition: re-aiming AI to frame {ai_start_frame} (gen {gen})")
        try:
            # 1) detach old + reset ready map under _ai_lock (cheap). Play loop now sees
            #    frame_restorer=None and falls back to passthrough until the new one is in.
            with self._ai_lock:
                old = self.frame_restorer
                self.frame_restorer = None
                self.ai_ready_frames = {}
                self.ai_eof = False

            # 2) stop old OUTSIDE _ai_lock (joins worker threads). Preserve the stop handshake.
            if old is not None:
                old_q = old.get_frame_restoration_queue()
                old.stop()
                threading_utils.put_queue_stop_marker(old_q)
                threading_utils.empty_out_queue(old_q)

            if self._reposition_cancelled(gen):
                logger.debug(f"realtime reposition: cancelled before start (gen {gen})")
                return

            # 3) build + start the new restorer (models are cached -> no reload).
            start_ns = self._clamp_frame_to_start_ns(ai_start_frame)
            new = self.frame_restorer_provider.get(
                frame_restoration_queue_max_bytes=self._compute_frame_buffer_max_bytes())
            new.start(start_ns=start_ns)

            # 4) swap in iff still current gen, else discard the freshly-built one.
            doomed = None
            with self._ai_lock:
                if self._reposition_cancelled(gen):
                    doomed = new
                else:
                    self.frame_restorer = new
                    self.ai_ready_frames = {}
                    self.ai_eof = False
            if doomed is not None:
                logger.debug(f"realtime reposition: cancelled after start, discarding (gen {gen})")
                doomed_q = doomed.get_frame_restoration_queue()
                doomed.stop()
                threading_utils.put_queue_stop_marker(doomed_q)
                threading_utils.empty_out_queue(doomed_q)
                return

            self._update_processing_frontier(
                video_utils.offset_ns_to_frame_num(self.current_timestamp_ns, self.video_metadata.video_fps_exact))
            logger.debug(f"realtime reposition: AI now at {start_ns/Gst.SECOND:.2f}s (gen {gen})")
        finally:
            with self._reposition_lock:
                if self._reposition_gen == gen:
                    self._reposition_in_progress = False
                    self._reposition_thread = None

    def _cancel_and_join_reposition(self):
        """Cancel an in-flight reposition and wait for its thread to finish. Called from
        _stop_appsource_worker while holding frame_restorer_lock. Safe: the reposition thread
        never takes frame_restorer_lock, so there's no lock cycle."""
        with self._reposition_lock:
            self._reposition_gen += 1  # invalidate any running gen
            t = self._reposition_thread
        if t is not None and t is not threading.current_thread():
            t.join()
        with self._reposition_lock:
            self._reposition_in_progress = False
            self._reposition_thread = None


GObject.type_register(RealtimeFrameRestorerAppSrc)
__gstelementfactory__ = (RealtimeFrameRestorerAppSrc.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, RealtimeFrameRestorerAppSrc)
