# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
Clock-driven realtime pipeline manager.

Subclasses the buffer-first PipelineManager (lada/gui/watch/gstreamer_pipeline_manager.py)
and only changes the video branch:
- uses RealtimeFrameRestorerAppSrc instead of FrameRestorerAppSrc
- the downstream queue has NO min-threshold-time (no pre-buffering before playback starts)
  and only a small max-size-time to absorb jitter
- does NOT wire underrun -> "waiting-for-data" (which the watch path uses to pause+rebuffer).
  Clock-driven playback must never pause on the AI source falling behind.

Audio / subtitles / seek behaviour is inherited unchanged.
"""

import logging
import sys
import time

from gi.repository import GLib, Gst, Gdk

from lada import LOG_LEVEL
from lada.gui.watch.gstreamer_pipeline_manager import PipelineManager
from lada.gui.realtime.gstreamer_pipeline_appsrc_realtime import RealtimeFrameRestorerAppSrc

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


class RealtimePipelineManager(PipelineManager):
    def pipeline_add_video(self):
        appsrc = RealtimeFrameRestorerAppSrc()
        appsrc.set_property('video-metadata', self.video_metadata)
        appsrc.set_property('frame-restorer-provider', self.frame_restorer_provider)

        def on_appsrc_end_of_stream(src):
            logger.debug("realtime appsource end-of-stream")
            return False
        appsrc.connect("end-of-stream", on_appsrc_end_of_stream)
        self.pipeline.add(appsrc)

        # Small jitter buffer only. No min-threshold-time: playback starts immediately and
        # is paced by the sink clock, never waiting for the AI source to fill a buffer.
        buffer_queue = Gst.ElementFactory.make('queue', None)
        buffer_queue.set_property('max-size-bytes', 0)
        buffer_queue.set_property('max-size-buffers', 0)
        buffer_queue.set_property('max-size-time', self.buffer_queue_max_thresh_time * Gst.SECOND)
        # deliberately no 'min-threshold-time' and no underrun/overrun -> waiting-for-data wiring
        self.pipeline.add(buffer_queue)

        gtksink = Gst.ElementFactory.make('gtk4paintablesink', None)
        paintable: Gdk.Paintable = gtksink.get_property('paintable')
        # TODO: workaround for #62 (same as watch path): on Windows + Nvidia, OpenGL paintable
        #  causes messed up colors, so don't use glsinkbin there.
        if paintable.props.gl_context and sys.platform != 'win32':
            video_sink = Gst.ElementFactory.make('glsinkbin', None)
            video_sink.set_property('sink', gtksink)
        else:
            video_sink = Gst.Bin.new()
            convert = Gst.ElementFactory.make('videoconvert', None)
            video_sink.add(convert)
            video_sink.add(gtksink)
            convert.link(gtksink)
            video_sink.add_pad(Gst.GhostPad.new('sink', convert.get_static_pad('sink')))
        self.pipeline.add(video_sink)

        appsrc.link(buffer_queue)
        buffer_queue.link(video_sink)

        self.video_sink = video_sink
        self.video_buffer_queue = buffer_queue
        self.frame_restorer_app_src = appsrc
        self.paintable = paintable
        self.paintable.connect("invalidate-size", lambda obj: GLib.idle_add(lambda: self.emit("paintable-size-changed")))

    def update_gst_buffers(self, buffer_queue_min_thresh_time, buffer_queue_max_thresh_time):
        # Realtime path ignores min-threshold-time entirely; only keep a jitter cap.
        self.video_buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)
        if self.has_audio:
            self.audio_buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)

    def set_preheat_duration(self, seconds: float):
        """How far ahead of the playhead the AI restorer starts after a (re)start/seek."""
        appsrc = getattr(self, "frame_restorer_app_src", None)
        if appsrc is not None:
            appsrc.preheat_duration_sec = max(0.0, float(seconds))

    def set_lookahead_frames(self, frames: int):
        """How far ahead of the playhead the frontier gate lets the AI work (lead it may build)."""
        appsrc = getattr(self, "frame_restorer_app_src", None)
        if appsrc is not None:
            appsrc.lookahead_frames = max(0, int(frames))

    def get_realtime_stats(self) -> dict | None:
        """Snapshot of realtime AI diagnostics + derived ahead/behind frame counts.
        Returns None if the appsrc isn't set up yet."""
        appsrc = getattr(self, "frame_restorer_app_src", None)
        if appsrc is None or not hasattr(appsrc, "get_stats"):
            return None
        stats = appsrc.get_stats()

        # Prefer the pipeline clock position over the appsrc's last-pushed PTS.
        playhead_ns = self.get_position_ns()
        if playhead_ns is None or playhead_ns < 0:
            playhead_ns = stats.get("playhead_ns", 0)

        frame_dur = stats.get("frame_duration_ns") or 0
        max_ai_pts_ns = stats.get("max_ai_pts_ns")

        ahead_frames = 0
        behind_frames = 0
        if max_ai_pts_ns is not None and frame_dur > 0:
            # AI processes sequentially, so everything from the playhead up to the most
            # advanced processed PTS is effectively a contiguous processed segment.
            delta_frames = (max_ai_pts_ns - playhead_ns) / frame_dur
            if delta_frames >= 0:
                ahead_frames = int(round(delta_frames))
            else:
                behind_frames = int(round(-delta_frames))

        stats["playhead_ns"] = playhead_ns
        stats["ahead_frames"] = ahead_frames
        stats["behind_frames"] = behind_frames

        # AI throughput (frames/sec): rate of drained restored frames between samples.
        now = time.monotonic()
        drained = stats.get("ai_frames_drained", 0)
        prev_t = getattr(self, "_fps_sample_time", None)
        prev_drained = getattr(self, "_fps_sample_drained", None)
        ai_fps = 0.0
        if prev_t is not None and now > prev_t:
            dt = now - prev_t
            d_frames = drained - prev_drained
            if dt > 0 and d_frames >= 0:
                ai_fps = d_frames / dt
        self._fps_sample_time = now
        self._fps_sample_drained = drained
        stats["ai_fps"] = ai_fps
        return stats

