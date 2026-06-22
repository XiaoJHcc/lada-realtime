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

    def set_realtime_clip_frames(self, frames: int):
        """Realtime-only clip length (frames). The AI restorer starts one clip ahead of the
        playhead and re-aims two clips ahead when it falls behind. Separate from the shared
        max_clip_duration used by watch/export."""
        appsrc = getattr(self, "frame_restorer_app_src", None)
        if appsrc is not None:
            appsrc.clip_frames = max(1, int(frames))

    def set_lookahead_frames(self, frames: int):
        """How far ahead of the playhead the frontier gate lets the AI work (lead it may build)."""
        appsrc = getattr(self, "frame_restorer_app_src", None)
        if appsrc is not None:
            appsrc.lookahead_frames = max(0, int(frames))

    def set_cold_start_clips(self, clips: int):
        """How many clips ahead of the playhead the AI restorer starts on cold start / seek.
        The lead region plays the original until the AI catches up; larger values hide a bigger
        cold-start cost at the price of more original-frame lead-in after each seek."""
        appsrc = getattr(self, "frame_restorer_app_src", None)
        if appsrc is not None:
            appsrc.cold_start_clips = max(1, int(clips))

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

        # Ahead/behind from the restorer's LIVE consume position (output_frame_pos), so it keeps
        # updating while playback is paused. The AI processes sequentially, so output_frame_pos
        # is the head of the contiguous processed segment. Fall back to max_ai_pts (last drained
        # PTS) only when the restorer isn't up. playhead in frames for an apples-to-apples diff.
        ahead_frames = 0
        behind_frames = 0
        output_frame_pos = stats.get("output_frame_pos")
        max_ai_pts_ns = stats.get("max_ai_pts_ns")
        if frame_dur > 0:
            playhead_frame = playhead_ns / frame_dur
            ai_head_frame = None
            if output_frame_pos is not None:
                ai_head_frame = output_frame_pos
            elif max_ai_pts_ns is not None:
                ai_head_frame = max_ai_pts_ns / frame_dur
            if ai_head_frame is not None:
                delta_frames = ai_head_frame - playhead_frame
                if delta_frames >= 0:
                    ahead_frames = int(round(delta_frames))
                else:
                    behind_frames = int(round(-delta_frames))

        stats["playhead_ns"] = playhead_ns
        stats["ahead_frames"] = ahead_frames
        stats["behind_frames"] = behind_frames

        # Buffer-bar fields (frame numbers). The bar draws the window [playhead, playhead+window)
        # and fills it where the ready interval [ready_start_frame, output_frame_pos) overlaps.
        # playhead_frame derived from the clock so it tracks playback; ready bounds come live
        # from the restorer so the bar keeps filling while paused.
        if frame_dur > 0:
            stats["playhead_frame"] = int(round(playhead_ns / frame_dur))
        else:
            stats["playhead_frame"] = 0
        # ready_start_frame / output_frame_pos / window_frames are already in stats (appsrc).

        # Production-side GPU throughput, measured per batch/clip in the worker threads and
        # forwarded as detector_fps_live / restorer_fps_live. Stays correct across realtime
        # repositions (a fresh restorer reports a valid rate after its first batch) where the
        # old differentiate-a-counter approach read 0. None means idle -> hold the last sample.
        def _hold(live_value, hold_attr):
            if live_value is not None:
                setattr(self, hold_attr, live_value)
                return live_value
            return getattr(self, hold_attr, 0.0)

        stats["detector_fps"] = _hold(stats.get("detector_fps_live"), "_fps_hold_detector")
        stats["restorer_fps"] = _hold(stats.get("restorer_fps_live"), "_fps_hold_restorer")

        # Consume-side rate: how fast the play loop actually drains restored frames. Kept for
        # comparison with restorer_fps (production high + consume low => building a lead;
        # both low while behind => genuinely out of compute). Goes to 0 when paused.
        now = time.monotonic()
        prev_t = getattr(self, "_fps_sample_time", None)
        dt = (now - prev_t) if (prev_t is not None and now > prev_t) else 0.0
        cur_drained = stats.get("ai_frames_drained", 0)
        prev_drained = getattr(self, "_fps_sample_drained", None)
        self._fps_sample_drained = cur_drained
        if cur_drained is None or prev_drained is None or dt <= 0:
            stats["ai_fps"] = 0.0
        else:
            d = cur_drained - prev_drained
            stats["ai_fps"] = d / dt if d >= 0 else 0.0  # d<0 => counter reset on (re)start

        self._fps_sample_time = now
        return stats

