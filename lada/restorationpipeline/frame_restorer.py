# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import textwrap
import threading
import time

import cv2
import torch
import numpy as np

from lada import LOG_LEVEL
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, StopMarker, EofMarker, PipelineQueue, PipelineThread, \
    ErrorMarker
from lada.utils import image_utils, video_utils, threading_utils, mask_utils, ImageTensor, Image
from lada.utils import visualization_utils
from lada.restorationpipeline.mosaic_detector import MosaicDetector
from lada.restorationpipeline.mosaic_detector import Clip
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class FrameRestorer:
    def __init__(self, device, video_file, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model: Yolo11SegmentationModel, mosaic_restoration_model, preferred_pad_mode,
                 mosaic_detection=False, frame_restoration_queue_max_bytes=512 * 1024 * 1024):
        self.device = torch.device(device)
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False

        # Output queue size. Default ~512MB = upstream behaviour (CLI export). The realtime
        # path passes a larger value so the AI can build up a lead of restored frames during
        # easy/empty stretches to spend on harder stretches (frames are CPU tensors).
        max_frames_in_frame_restoration_queue = max(1, frame_restoration_queue_max_bytes // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3))
        self.frame_restoration_queue = PipelineQueue(name="frame_restoration_queue", maxsize=max_frames_in_frame_restoration_queue)

        # limit queue size to approx 512MB
        max_clips_in_mosaic_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)) # 4 = 3 color channels + mask
        self.mosaic_clip_queue = PipelineQueue(name="mosaic_clip_queue", maxsize=max_clips_in_mosaic_clips_queue)

        # limit queue size to approx 512MB
        max_clips_in_restored_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)) # 4 = 3 color channels + mask
        self.restored_clip_queue = PipelineQueue(name="restored_clip_queue", maxsize=max_clips_in_restored_clips_queue)

        # no queue size limit needed, elements are tiny
        self.frame_detection_queue = PipelineQueue(name="frame_detection_queue")

        self.mosaic_detector = MosaicDetector(self.mosaic_detection_model, self.video_meta_data,
                                              frame_detection_queue=self.frame_detection_queue,
                                              mosaic_clip_queue=self.mosaic_clip_queue,
                                              device=self.device,
                                              max_clip_length=self.max_clip_length,
                                              pad_mode=self.preferred_pad_mode,
                                              error_handler=self._on_worker_thread_error)

        self.clip_restoration_thread: PipelineThread | None = None
        self.frame_restoration_thread: PipelineThread | None = None
        self.start_stop_lock: threading.Lock = threading.Lock()
        self.stop_requested = False

        # Production-side throughput counter: total frames BasicVSR++/DeepMosaics has actually
        # restored. Monotonic, never reset on pause -> sampling it twice gives the true
        # restoration rate even while playback is paused or falling back to passthrough.
        self._restorer_frames_done = 0
        self._restorer_frames_lock = threading.Lock()

        # Live restoration throughput: rolling window of (mono_time, frames, proc_seconds) per
        # restored clip. fps = sum(frames)/sum(proc_seconds), the model's real processing rate
        # while busy. Measured per clip at production time so it's valid immediately on a fresh
        # restorer (a realtime reposition builds a new FrameRestorer; a cross-sample delta on a
        # monotonic counter would read 0 right after each reposition). None when idle.
        self._restorer_fps_window: list[tuple[float, int, float]] = []
        self._restorer_fps_lock = threading.Lock()
        self._fps_window_sec = 2.0

        # Just-in-time detector gate (realtime only). The realtime appsrc sets a playhead-based
        # cap via set_processing_frontier (playhead + lookahead window). On its own that lets the
        # fast YOLO detector race the WHOLE window ahead at startup, hogging the GPU from the much
        # slower restoration model and producing clips the restorer won't consume for a long time.
        # We additionally clamp the detector to the AI OUTPUT position (the frame restoration
        # worker, the real clip consumer) plus a small lead, so the detector only runs just ahead
        # of what the restorer is about to need. Lead must be >= max_clip_length so the clip
        # covering the current output frame can always complete.
        #
        # Lead multiplier (units of max_clip_length), profiled on 1080p30 + 4080, clip_T=30,
        # n=5 seek points (scripts/realtime_coldstart_profile.py):
        #   2.0x: cold-start first frame ~2227ms, steady delivered ~40.7fps (steady optimum)
        #   1.2x: cold-start first frame ~2131ms, steady delivered ~39.0fps
        #   1.0x: cold-start first frame ~2220ms, steady delivered ~37.2fps (restorer stalls
        #         at the clip boundary waiting on the detector — net loss)
        # 1.2x is the cold-start sweet spot (~96ms faster first frame, significant across all
        # seeks); the steady-state delta vs 2.0x is within run-to-run noise (mixed signs, n=5).
        # We default to 1.2x to win the first-frame latency at no measurable steady cost.
        self._playhead_frontier: int | None = None
        self._output_frame_pos: int = 0
        self._detector_lead = max(self.max_clip_length, round(1.2 * self.max_clip_length))
        self._frontier_lock = threading.Lock()

    def start(self, start_ns=0):
        with self.start_stop_lock:
            assert self.frame_restoration_thread is None and self.clip_restoration_thread is None, "Illegal State: Tried to start FrameRestorer when it's already running. You need to stop it first"
            assert self.mosaic_clip_queue.empty()
            assert self.restored_clip_queue.empty()
            assert self.frame_detection_queue.empty()
            assert self.frame_restoration_queue.empty()

            self.start_ns = start_ns
            # Anchor start_frame to the REAL frame the decoder lands on after seek. PyAV seeks
            # BACKWARD to the nearest keyframe, so the first decoded frame is usually earlier
            # than start_ns; both this worker and the detector label that first frame as
            # start_frame and count +1 from there. Deriving it from the nominal offset (the old
            # behaviour) leaves a per-seek GOP offset between this frame number and the real
            # pts, which the realtime frontier gate then mis-compares against the (real)
            # playhead -> AI production cut short. Computing it from the first decoded pts puts
            # the frame numbering back on the real-pts coordinate system. CLI/watch never read
            # these frame numbers, so their behaviour is unchanged.
            if start_ns > 0:
                self.start_frame = video_utils.first_decoded_frame_num_after_seek(
                    self.video_meta_data.video_file, start_ns,
                    self.video_meta_data.time_base, self.video_meta_data.video_fps_exact)
            else:
                self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
            self.stop_requested = False
            self._output_frame_pos = self.start_frame

            self.frame_restoration_thread = PipelineThread(name="frame restoration worker", target=self._frame_restoration_worker, error_handler=self._on_worker_thread_error)
            self.clip_restoration_thread = PipelineThread(name="clip restoration worker", target=self._clip_restoration_worker, error_handler=self._on_worker_thread_error)

            self.mosaic_detector.start(start_ns=start_ns, start_frame=self.start_frame)
            self.clip_restoration_thread.start()
            self.frame_restoration_thread.start()

    def stop(self):
        logger.debug("FrameRestorer: stopping...")
        start = time.time()
        with self.start_stop_lock:
            self.stop_requested = True

            self.mosaic_detector.stop()

            # unblock consumer
            threading_utils.put_queue_stop_marker(self.mosaic_clip_queue)
            # unblock producer
            threading_utils.empty_out_queue(self.restored_clip_queue)
            # wait until thread stopped
            if self.clip_restoration_thread:
                self.clip_restoration_thread.join()
                logger.debug("FrameRestorer: joined clip_restoration_thread")
            self.clip_restoration_thread = None

            # unblock consumer
            threading_utils.put_queue_stop_marker(self.frame_detection_queue)
            threading_utils.put_queue_stop_marker(self.restored_clip_queue)
            # unblock producer
            threading_utils.empty_out_queue(self.frame_restoration_queue)
            # wait until thread stopped
            if self.frame_restoration_thread:
                self.frame_restoration_thread.join()
                logger.debug("FrameRestorer: joined frame_restoration_thread")
            self.frame_restoration_thread = None

            # garbage collection
            threading_utils.empty_out_queue(self.mosaic_clip_queue)
            threading_utils.empty_out_queue(self.restored_clip_queue)
            threading_utils.empty_out_queue(self.frame_detection_queue)
            threading_utils.empty_out_queue(self.frame_restoration_queue)

            assert self.mosaic_clip_queue.empty()
            assert self.restored_clip_queue.empty()
            assert self.frame_detection_queue.empty()
            assert self.frame_restoration_queue.empty()

            logger.debug(f"FrameRestorer: stopped, took {time.time() - start}")
            self._dump_queue_stats()

    def _on_worker_thread_error(self, error: ErrorMarker):
        def stop_and_notify():
            self.stop()
            # unblock CLI/GUI consumer
            self.frame_restoration_queue.put(error)
        thread = threading.Thread(target=stop_and_notify, daemon=True)
        thread.start()

    def _dump_queue_stats(self):
        logger.debug(textwrap.dedent(f"""\
            FrameRestorer: Queue stats:
                frame_restoration_queue/wait-time-get: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_get"]:.0f}
                frame_restoration_queue/wait-time-put: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_put"]:.0f}
                frame_restoration_queue/max-qsize: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_max_size"]}/{self.frame_restoration_queue.maxsize}
                ---
                mosaic_clip_queue/wait-time-get: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_get"]:.0f}
                mosaic_clip_queue/wait-time-put: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_put"]:.0f}
                mosaic_clip_queue/max-qsize: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_max_size"]}/{self.mosaic_clip_queue.maxsize}
                ---
                frame_detection_queue/wait-time-get: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_get"]:.0f}
                frame_detection_queue/wait-time-put: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_put"]:.0f}
                frame_detection_queue/max-qsize: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_max_size"]}/{self.frame_detection_queue.maxsize}
                ---
                restored_clip_queue/wait-time-get: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_get"]:.0f}
                restored_clip_queue/wait-time-put: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_put"]:.0f}
                restored_clip_queue/max-qsize: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_max_size"]}/{self.restored_clip_queue.maxsize}
                ---
                frame_feeder_queue/wait-time-get: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_get"]:.0f}
                frame_feeder_queue/wait-time-put: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_put"]:.0f}
                frame_feeder_queue/max-qsize: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_max_size"]}/{self.mosaic_detector.frame_feeder_queue.maxsize}"""))

    def _restore_clip_frames(self, images: list[ImageTensor]):
        if self.mosaic_restoration_model_name.startswith("deepmosaics"):
            from lada.restorationpipeline.deepmosaics_mosaic_restorer import DeepmosaicsMosaicRestorer
            assert isinstance(self.mosaic_restoration_model, DeepmosaicsMosaicRestorer)
            restored_clip_images = self.mosaic_restoration_model.restore(images)
        elif self.mosaic_restoration_model_name.startswith("basicvsrpp"):
            from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
            assert isinstance(self.mosaic_restoration_model, BasicvsrppMosaicRestorer)
            restored_clip_images = self.mosaic_restoration_model.restore(images)
        else:
            raise NotImplementedError()
        return restored_clip_images

    def _restore_frame(self, frame: ImageTensor, frame_num: int, restored_clips: list[Clip]):
        """
        Takes mosaic frame and restored clips and replaces mosaic regions in frame with restored content from the clips starting at the same frame number as mosaic frame.
        Pops starting frame from each restored clip in the process if they actually start at the same frame number as frame.
        """
        is_cpu_input = frame.device.type == 'cpu'
        target_dtype = torch.float32 if is_cpu_input else self.mosaic_restoration_model.dtype
        def _blend_gpu(blend_mask: torch.Tensor, clip_img: torch.Tensor, orig_clip_box: tuple[int, int, int, int]):
            t, l, b, r = orig_clip_box
            frame_roi = frame[t:b + 1, l:r + 1, :]
            roi_f = frame_roi.to(dtype=self.mosaic_restoration_model.dtype)
            temp = clip_img.to(dtype=self.mosaic_restoration_model.dtype, device=frame_roi.device)
            temp.sub_(roi_f)
            temp.mul_(blend_mask.unsqueeze(-1))
            temp.add_(roi_f)
            temp.round_().clamp_(0, 255)
            frame_roi[:] = temp

        def _blend_cpu(blend_mask: torch.Tensor, clip_img: torch.Tensor, orig_clip_box: tuple[int, int, int, int]):
            blend_mask = blend_mask.cpu().numpy()
            clip_img = clip_img.cpu().numpy()
            t, l, b, r = orig_clip_box
            frame_roi = frame[t:b + 1, l:r + 1, :].numpy()
            temp_buffer = np.empty_like(frame_roi, dtype=np.float32)
            np.subtract(clip_img, frame_roi, out=temp_buffer, dtype=np.float32)
            np.multiply(temp_buffer, blend_mask[..., None], out=temp_buffer)
            np.add(temp_buffer, frame_roi, out=temp_buffer)
            frame_roi[:] = temp_buffer.astype(np.uint8)
            
        blend = _blend_cpu if is_cpu_input else _blend_gpu

        for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = buffered_clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(clip_mask, orig_crop_shape[:2],interpolation=cv2.INTER_NEAREST)
            blend_mask = mask_utils.create_blend_mask(clip_mask.to(device=self.device).float()).to(device=clip_img.device, dtype=target_dtype)

            blend(blend_mask, clip_img, orig_clip_box)

    def _restore_clip(self, clip: Clip):
        """
        Restores each contained from of the mosaic clip. If self.mosaic_detection is True will instead draw mosaic detection
        boundaries on each frame.
        """
        if self.mosaic_detection:
            restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
        else:
            restored_clip_images = self._restore_clip_frames(clip.frames)
        assert len(restored_clip_images) == len(clip.frames)

        for i in range(len(restored_clip_images)):
            assert clip.frames[i].shape == restored_clip_images[i].shape
            clip.frames[i] = restored_clip_images[i]

    def _collect_garbage(self, clip_buffer):
        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
        has_processed_clips = len(processed_clips) > 0
        for processed_clip in processed_clips:
            clip_buffer.remove(processed_clip)

        if has_processed_clips:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()

    def _clip_buffer_contains_all_cips_needed_for_current_restoration(self, current_frame_num, num_mosaic_detections, clip_buffer):
        num_clips_starting_at_frame = len([clip for clip in clip_buffer if clip.frame_start == current_frame_num])
        assert num_clips_starting_at_frame <= num_mosaic_detections
        return num_clips_starting_at_frame == num_mosaic_detections

    def _clip_restoration_worker(self):
        logger.debug("clip restoration worker: started")
        eof = False
        while not (eof or self.stop_requested):
            clip = self.mosaic_clip_queue.get()
            if self.stop_requested or clip is STOP_MARKER:
                logger.debug("clip restoration worker: mosaic_clip_queue consumer unblocked")
                break
            if clip is EOF_MARKER:
                eof = True
                self.restored_clip_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                    break
            else:
                _t0 = time.monotonic()
                self._restore_clip(clip)
                _proc = time.monotonic() - _t0
                with self._restorer_frames_lock:
                    self._restorer_frames_done += len(clip)
                self._record_restorer_fps(len(clip), _proc)
                # Release MPS driver cached memory to prevent unbounded growth
                if self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                self.restored_clip_queue.put(clip)
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                    break
        if eof:
            logger.debug("clip restoration worker: stopped itself, EOF")
        else:
            logger.debug("clip restoration worker: stopped by request")

    def _read_next_frame(self, video_frames_generator, expected_frame_num) -> tuple[int, np.ndarray, int] | StopMarker | EofMarker:
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            elem = self.frame_detection_queue.get()
            if self.stop_requested or elem is STOP_MARKER:
                logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
                return STOP_MARKER
            assert elem is EOF_MARKER, f"Illegal state: Expected to read EOF_MARKER from detection queue but received f{elem}"
            return EOF_MARKER
        elem = self.frame_detection_queue.get()
        if self.stop_requested or elem is STOP_MARKER:
            logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
            return STOP_MARKER
        assert elem is not EOF_MARKER and elem is not STOP_MARKER, f"Illegal state: Expected to read detection result from detection queue but received {elem}"
        detection_frame_num, num_mosaics_detected = elem
        assert detection_frame_num == expected_frame_num, f"frame detection queue out of sync: received {detection_frame_num} expected {expected_frame_num}"
        return num_mosaics_detected, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer) -> StopMarker | EofMarker | None:
        clip = self.restored_clip_queue.get()
        if self.stop_requested or clip is STOP_MARKER:
            logger.debug("frame restoration worker: restored_clip_queue consumer unblocked")
            return STOP_MARKER
        if clip is EOF_MARKER:
            return EOF_MARKER
        assert clip.frame_start >= current_frame_num, "clip queue out of sync!"
        clip_buffer.append(clip)
        return None

    def _frame_restoration_worker(self):
        logger.debug("frame restoration worker: started")
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()

            frame_num = self.start_frame
            queue_marker = None
            clip_buffer = []

            while not (self.eof or self.stop_requested):
                # Publish the consumer position and re-apply the just-in-time detector gate so
                # the detector is allowed to run only a small lead ahead of what the restorer is
                # about to consume (no-op when the realtime playhead gate is disabled).
                self._set_output_frame_pos(frame_num)
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if self.stop_requested or _frame_result is STOP_MARKER:
                    break
                if _frame_result is EOF_MARKER:
                    self.eof = True
                    self.frame_restoration_queue.put(EOF_MARKER)
                    break
                num_mosaics_detected, frame, frame_pts = _frame_result
                if num_mosaics_detected > 0:
                    while queue_marker is None and not self._clip_buffer_contains_all_cips_needed_for_current_restoration(frame_num, num_mosaics_detected, clip_buffer):
                        queue_marker = self._read_next_clip(frame_num, clip_buffer)
                    if queue_marker is STOP_MARKER:
                        break

                    self._restore_frame(frame, frame_num, clip_buffer)
                    self.frame_restoration_queue.put((frame, frame_pts))
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                        break
                    self._collect_garbage(clip_buffer)
                else:
                    self.frame_restoration_queue.put((frame, frame_pts))
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                        break
                frame_num += 1
        if self.eof:
            logger.debug("frame restoration worker: stopped itself, EOF")
        else:
            logger.debug("frame restoration worker: stopped by request")

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, int] | ErrorMarker | StopMarker:
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration
        else:
            while True:
                elem = self.frame_restoration_queue.get()
                if self.stop_requested or elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                    logger.debug("frame_restoration_queue consumer unblocked")
                    return elem
                if elem is EOF_MARKER:
                    raise StopIteration
                return elem

    def get_frame_restoration_queue(self) -> PipelineQueue:
        return self.frame_restoration_queue

    def get_restorer_frames_done(self) -> int:
        """Total frames the restoration model (BasicVSR++/DeepMosaics) has restored so far
        (monotonic). Sample twice and divide by elapsed wall time to get the restoration fps."""
        with self._restorer_frames_lock:
            return self._restorer_frames_done

    def _record_restorer_fps(self, frames: int, proc_seconds: float):
        """Append one restored clip's (frames, processing seconds) to the rolling window."""
        now = time.monotonic()
        with self._restorer_fps_lock:
            self._restorer_fps_window.append((now, frames, proc_seconds))
            cutoff = now - self._fps_window_sec
            while self._restorer_fps_window and self._restorer_fps_window[0][0] < cutoff:
                self._restorer_fps_window.pop(0)

    def get_restorer_fps(self) -> float | None:
        """Live restoration rate: frames per second of model time over the last ~window
        seconds. None when idle, so the consumer holds the previous value instead of 0."""
        now = time.monotonic()
        with self._restorer_fps_lock:
            cutoff = now - self._fps_window_sec
            recent = [s for s in self._restorer_fps_window if s[0] >= cutoff]
        total_frames = sum(s[1] for s in recent)
        total_proc = sum(s[2] for s in recent)
        if total_frames == 0 or total_proc <= 0:
            return None
        return total_frames / total_proc

    def get_detector_fps(self) -> float | None:
        """Live detection rate (frames/sec of model time), forwarded from the detector."""
        return self.mosaic_detector.get_detector_fps()

    def get_detector_frames_done(self) -> int:
        """Total frames the YOLO detector has run inference on so far (monotonic)."""
        return self.mosaic_detector.get_detector_frames_done()

    def get_output_frame_pos(self) -> int:
        """The frame number the frame-restoration worker is currently consuming (advances
        sequentially from start_frame). The realtime appsrc uses this to detect the AI output
        head falling behind the playhead and trigger a reposition."""
        with self._frontier_lock:
            return self._output_frame_pos

    def get_start_frame(self) -> int:
        """The frame number this restorer started producing from. With output_frame_pos it
        bounds the contiguous ready interval [start_frame, output_frame_pos) the realtime
        buffer bar visualizes."""
        return self.start_frame

    def set_processing_frontier(self, frame_num: int | None):
        """Limit processing to frames up to frame_num (None disables; default upstream
        behaviour). frame_num is the playhead-based cap from the realtime appsrc (playhead +
        lookahead window). We forward the MIN of that cap and a just-in-time bound anchored to
        the AI output position, so the fast detector can't race the whole window ahead of the
        slow restorer and hog the GPU. Forwards to the detector's feeder gate, which
        backpressures the whole chain. Only the realtime path calls this; CLI/watch never do."""
        with self._frontier_lock:
            self._playhead_frontier = frame_num
            self._apply_frontier_locked()

    def _set_output_frame_pos(self, frame_num: int):
        """Called by the frame restoration worker as it consumes frames. Advances the
        just-in-time detector bound so the detector keeps a small lead over the restorer."""
        with self._frontier_lock:
            self._output_frame_pos = frame_num
            self._apply_frontier_locked()

    def _apply_frontier_locked(self):
        """Combine the playhead cap with the output-anchored just-in-time bound and push the
        effective frontier to the detector. Caller holds _frontier_lock."""
        if self._playhead_frontier is None:
            # Gate disabled (CLI/watch): forward None so the detector runs unrestricted.
            self.mosaic_detector.set_processing_frontier(None)
            return
        jit_bound = self._output_frame_pos + self._detector_lead
        effective = min(self._playhead_frontier, jit_bound)
        self.mosaic_detector.set_processing_frontier(effective)