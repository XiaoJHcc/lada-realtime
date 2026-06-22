# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import threading
import time
from typing import List, Tuple, Callable

import cv2
import torch

from lada import LOG_LEVEL
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel
from lada.utils import Box
from lada.utils import VideoMetadata, threading_utils, ImageTensor, MaskTensor, Pad
from lada.utils import image_utils
from lada.utils import video_utils
from lada.utils.box_utils import box_overlap
from lada.utils.scene_utils import crop_to_box_v3
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, PipelineQueue, StopMarker, PipelineThread, ErrorMarker
from lada.utils.ultralytics_utils import convert_yolo_box, convert_yolo_mask_tensor, UltralyticsResults

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class Scene:
    def __init__(self, file_path: str, video_meta_data: VideoMetadata):
        self.file_path = file_path
        self.video_meta_data = video_meta_data
        self.frames: list[ImageTensor] = []
        self.masks: list[MaskTensor] = []
        self.boxes: list[Box] = []
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self._index: int = 0

    def __len__(self):
        return len(self.frames)

    def add_frame(self, frame_num: int, img: ImageTensor, mask: MaskTensor, box: Box):
        if self.frame_start is None:
            self.frame_start = frame_num
            self.frame_end = frame_num
        else:
            assert frame_num == self.frame_end + 1
            self.frame_end = frame_num

        self.frames.append(img)
        self.masks.append(mask)
        self.boxes.append(box)

    def merge_mask_box(self, mask: MaskTensor, box: Box):
        assert self.belongs(box)
        current_box = self.boxes[-1]
        t = min(current_box[0], box[0])
        l = min(current_box[1], box[1])
        b = max(current_box[2], box[2])
        r = max(current_box[3], box[3])
        new_box = (t, l, b, r)
        self.boxes[-1] = new_box
        self.masks[-1] = torch.maximum(self.masks[-1], mask)

    def belongs(self, box: Box):
        if len(self.boxes) == 0:
            return False
        last_scene_box = self.boxes[-1]
        return box_overlap(last_scene_box, box)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self):
            item = self.frames[self._index], self.masks[self._index], self.boxes[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration


class Clip:
    def __init__(self, scene: Scene, size, pad_mode, id):
        self.id = id
        self.file_path = scene.file_path
        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end
        assert self.frame_start <= self.frame_end
        self.size = size
        self.pad_mode = pad_mode
        self.frames: list[ImageTensor] = []
        self.masks: list[MaskTensor] = []
        self.boxes: list[Box] = []
        self.crop_shapes: List[Tuple[int, int]] = []
        self.pad_after_resizes: List[Pad] = []
        self._index: int = 0

        # crop scene
        for i in range(len(scene)):
            img, mask, box = scene.frames[i], scene.masks[i], scene.boxes[i]
            cropped_img, cropped_mask, cropped_box, _ = crop_to_box_v3(box, img, mask, (size, size), max_box_expansion_factor=1., border_size=0.06)
            self.frames.append(cropped_img)
            self.masks.append(cropped_mask)
            self.boxes.append(cropped_box)
            self.crop_shapes.append(cropped_img.shape)

        # resize crops to out_size
        max_width, max_height = self.get_max_width_height()
        scale_width, scale_height = size/max_width, size/max_height

        for i, (cropped_img, cropped_mask, cropped_box) in enumerate(zip(self.frames, self.masks, self.boxes)):
            crop_shape = cropped_img.shape

            resize_shape = (int(crop_shape[0] * scale_height), int(crop_shape[1] * scale_width))
            cropped_img = image_utils.resize(cropped_img, resize_shape, interpolation=cv2.INTER_LINEAR)
            cropped_mask = image_utils.resize(cropped_mask, resize_shape, interpolation=cv2.INTER_NEAREST)
            assert cropped_mask.shape[:2] == cropped_img.shape[:2], f"{cropped_mask.shape[:2]}, {cropped_img.shape[:2]}"
            assert cropped_img.shape[0] <= size or cropped_img.shape[1] <= size

            cropped_img, pad_after_resize = image_utils.pad_image(cropped_img, size, size, mode=self.pad_mode)
            cropped_mask, _ = image_utils.pad_image(cropped_mask, size, size, mode='zero')

            self.frames[i] = cropped_img
            self.masks[i] = cropped_mask
            self.boxes[i] = cropped_box
            self.crop_shapes[i] = crop_shape
            self.pad_after_resizes.append(pad_after_resize)

    def get_max_width_height(self):
        max_width = 0
        max_height = 0
        for box in self.boxes:
            t, l, b, r = box
            width, height = r - l + 1, b - t + 1
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
        return max_width, max_height

    def pop(self):
        self.frame_start += 1
        if self.frame_start > self.frame_end:
            self.frame_start = None
            self.frame_end = None

        return self.frames.pop(0), self.masks.pop(0), self.boxes.pop(0), self.crop_shapes.pop(0), self.pad_after_resizes.pop(0)

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self):
            item = self.frames[self._index], self.masks[self._index], self.boxes[self._index], self.crop_shapes[self._index], self.pad_after_resizes[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.frames[item], self.masks[item], self.boxes[item]

class MosaicDetector:
    def __init__(self, model: Yolo11SegmentationModel, video_metadata: VideoMetadata, frame_detection_queue: PipelineQueue, mosaic_clip_queue: PipelineQueue, error_handler: Callable[[ErrorMarker], None], max_clip_length=30, clip_size=256, device: torch.device | None = None, pad_mode='reflect', batch_size=4):
        self.model = model
        self.video_meta_data = video_metadata
        self.device = torch.device(device) if device is not None else device
        self.max_clip_length = max_clip_length
        assert max_clip_length > 0
        self.clip_size = clip_size
        self.pad_mode = pad_mode
        self.clip_counter = 0
        self.start_ns = 0
        self.start_frame = 0
        self.frame_detection_queue = frame_detection_queue
        self.mosaic_clip_queue = mosaic_clip_queue
        self.frame_feeder_queue = PipelineQueue(name="frame_feeder_queue", maxsize=8)
        self.inference_queue = PipelineQueue(name="frame_feeder_queue", maxsize=8)
        self.error_handler = error_handler
        self.frame_detector_thread: PipelineThread | None = None
        self.frame_feeder_thread: PipelineThread | None = None
        self.inference_thread: PipelineThread | None = None
        self.stop_requested = False
        self.batch_size = batch_size

        # Production-side throughput counter: total frames the YOLO detector has actually
        # run inference on. Monotonic, never reset on pause -> a consumer sampling it twice
        # gets the true detection rate even while playback is paused/falling back. Plain int
        # guarded by a lock; the GIL makes the += atomic but the lock keeps reads coherent.
        self._detector_frames_done = 0
        self._detector_frames_lock = threading.Lock()

        # Live detection throughput: a short rolling window of (mono_time, frames, proc_seconds)
        # for each inference batch. fps = sum(frames)/sum(proc_seconds) over the window, i.e.
        # the model's actual processing rate while busy. Measured per batch at production time
        # so it's valid immediately on a fresh detector (no cross-sample delta that a realtime
        # reposition would reset to 0). Returns None when idle so the consumer can hold the last
        # value instead of flickering to 0.
        self._detector_fps_window: list[tuple[float, int, float]] = []
        self._detector_fps_lock = threading.Lock()
        self._fps_window_sec = 2.0

        # Optional processing-frontier gate (default OFF -> identical to upstream behaviour).
        # When set, the feeder blocks before decoding a batch whose frame_num has reached the
        # frontier, until the frontier advances (realtime playhead) or stop is requested.
        # This backpressures the whole detect->restore chain to follow the playhead.
        self._frontier_frame: int | None = None
        self._frontier_cond = threading.Condition()

    def start(self, start_ns, start_frame=None):
        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        self.start_ns = start_ns
        # start_frame is the REAL frame number the decoder lands on after a BACKWARD seek
        # (the keyframe at-or-before start_ns), supplied by FrameRestorer so the detector and
        # the restoration worker count from an identical base. Falling back to the nominal
        # offset only happens when no override is given (no realtime path does that).
        if start_frame is not None:
            self.start_frame = start_frame
        else:
            self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
        self.stop_requested = False

        self.frame_detector_thread = PipelineThread(name="frame detector worker", target=self._frame_detector_worker, error_handler=self.error_handler)
        self.frame_detector_thread.start()

        self.inference_thread = PipelineThread(name="frame inference worker", target=self._frame_inference_worker, error_handler=self.error_handler)
        self.inference_thread.start()

        self.frame_feeder_thread = PipelineThread(name="frame feeder worker", target=self._frame_feeder_worker, error_handler=self.error_handler)
        self.frame_feeder_thread.start()

    def get_detector_frames_done(self) -> int:
        """Total frames the YOLO detector has run inference on so far (monotonic).
        Sample twice and divide by elapsed wall time to get the detection fps."""
        with self._detector_frames_lock:
            return self._detector_frames_done

    def _record_detector_fps(self, frames: int, proc_seconds: float):
        """Append one inference batch's (frames, processing seconds) to the rolling window."""
        now = time.monotonic()
        with self._detector_fps_lock:
            self._detector_fps_window.append((now, frames, proc_seconds))
            cutoff = now - self._fps_window_sec
            while self._detector_fps_window and self._detector_fps_window[0][0] < cutoff:
                self._detector_fps_window.pop(0)

    def get_detector_fps(self) -> float | None:
        """Live detection rate: frames processed per second of model time over the last
        ~window seconds. None when idle (no recent batch / zero processing time), so the
        consumer holds the previous value instead of dropping to 0."""
        now = time.monotonic()
        with self._detector_fps_lock:
            cutoff = now - self._fps_window_sec
            recent = [s for s in self._detector_fps_window if s[0] >= cutoff]
        total_frames = sum(s[1] for s in recent)
        total_proc = sum(s[2] for s in recent)
        if total_frames == 0 or total_proc <= 0:
            return None
        return total_frames / total_proc

    def set_processing_frontier(self, frame_num: int | None):
        """Allow the feeder to decode up to (but not past) frame_num. None disables the
        gate entirely (upstream behaviour). Safe to call from any thread."""
        with self._frontier_cond:
            self._frontier_frame = frame_num
            self._frontier_cond.notify_all()

    def _wait_for_frontier(self, frame_num: int):
        """Block while the gate is enabled and frame_num has reached the frontier.
        Wakes on frontier advance or stop. No-op when the gate is disabled (frontier None)."""
        with self._frontier_cond:
            while (self._frontier_frame is not None
                   and frame_num >= self._frontier_frame
                   and not self.stop_requested):
                self._frontier_cond.wait(timeout=0.1)

    def stop(self):
        logger.debug("MosaicDetector: stopping...")
        start = time.time()
        self.stop_requested = True

        # wake the feeder if it's parked on the processing-frontier gate
        with self._frontier_cond:
            self._frontier_cond.notify_all()

        # unblock producer
        threading_utils.empty_out_queue(self.frame_feeder_queue)
        if self.frame_feeder_thread:
            self.frame_feeder_thread.join()
            logger.debug("MosaicDetector: joined frame_feeder_thread")
        self.frame_feeder_thread = None
        
        # unblock consumer
        threading_utils.put_queue_stop_marker(self.frame_feeder_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.inference_queue)
        if self.inference_thread:
            self.inference_thread.join()
            logger.debug("MosaicDetector: joined inference_thread")
        self.inference_thread = None

        # unblock consumer
        threading_utils.put_queue_stop_marker(self.inference_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.mosaic_clip_queue)
        if self.frame_detector_thread:
            self.frame_detector_thread.join()
            logger.debug("MosaicDetector: joined frame_detector_thread")
        self.frame_detector_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.frame_feeder_queue)
        threading_utils.empty_out_queue(self.inference_queue)

        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        logger.debug(f"MosaicDetector: stopped, took: {time.time() - start}")

    def _create_clips_for_completed_scenes(self, scenes, frame_num, eof) -> StopMarker | None:
        completed_scenes = []
        for current_scene in scenes:
            if (current_scene.frame_end < frame_num or len(current_scene) >= self.max_clip_length or eof) and current_scene not in completed_scenes:
                completed_scenes.append(current_scene)
                other_scenes = [other for other in scenes if other != current_scene]
                for other_scene in other_scenes:
                    if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                        completed_scenes.append(other_scene)

        for completed_scene in sorted(completed_scenes, key=lambda s: s.frame_start):
            clip = Clip(completed_scene, self.clip_size, self.pad_mode, self.clip_counter)
            self.mosaic_clip_queue.put(clip)
            if self.stop_requested:
                logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
                return STOP_MARKER
            #print(f"frame {frame_num}, yielding clip starting {clip.frame_start}, ending {clip.frame_end}, all scene starts: {[s.frame_start for s in scenes]}, completed scenes: {[s.frame_start for s in completed_scenes]}")
            scenes.remove(completed_scene)
            self.clip_counter += 1
        return None

    def _create_or_append_scenes_based_on_prediction_result(self, results: UltralyticsResults, scenes: list[Scene], frame_num):
        for i in range(len(results.boxes)):
            mask = convert_yolo_mask_tensor(results.masks[i], results.orig_shape).to(device=results.orig_img.device)
            box = convert_yolo_box(results.boxes[i], results.orig_shape)

            current_scene = None
            for scene in scenes:
                if scene.belongs(box):
                    if scene.frame_end == frame_num:
                        current_scene = scene
                        current_scene.merge_mask_box(mask, box)
                    else:
                        current_scene = scene
                        current_scene.add_frame(frame_num, results.orig_img, mask, box)
                    break
            if current_scene is None:
                current_scene = Scene(self.video_meta_data.video_file, self.video_meta_data)
                scenes.append(current_scene)
                current_scene.add_frame(frame_num, results.orig_img, mask, box)

    def _frame_feeder_worker(self):
        logger.debug("frame feeder: started")
        eof = False
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)
            video_frames_generator = video_reader.frames()
            frame_num = self.start_frame
            while not (eof or self.stop_requested):
                # Processing-frontier gate: park here until the playhead lets this batch
                # through (no-op when the gate is disabled). Keeps the GPU on frames near
                # the playhead instead of racing thousands of frames ahead.
                self._wait_for_frontier(frame_num)
                if self.stop_requested:
                    break
                try:
                    frames = []
                    for i in range(self.batch_size):
                        frame, _ = next(video_frames_generator)
                        frames.append(frame)
                except StopIteration:
                    eof = True
                if len(frames) > 0:
                    frames_batch = self.model.preprocess(frames)
                    data = (frames_batch, frames, frame_num)
                    self.frame_feeder_queue.put(data)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
                frame_num += len(frames)
                if eof:
                    self.frame_feeder_queue.put(EOF_MARKER)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
        if eof:
            logger.debug("frame feeder worker: stopped itself, EOF")
        else:
            logger.debug("frame feeder worker: stopped by request")

    def _frame_inference_worker(self):
        logger.debug("frame inference worker: started")
        eof = False
        while not (eof or self.stop_requested):
            frames_data = self.frame_feeder_queue.get()
            if self.stop_requested or frames_data is STOP_MARKER:
                logger.debug("inference worker: frame_feeder_queue consumer unblocked")
                break
            if frames_data is EOF_MARKER:
                eof = True
                self.inference_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("inference worker: inference_queue producer unblocked")
                    break
                break
            frames_batch, frames, frame_num = frames_data

            _t0 = time.monotonic()
            batch_prediction_results = self.model.inference_and_postprocess(frames_batch, frames)
            _proc = time.monotonic() - _t0

            with self._detector_frames_lock:
                self._detector_frames_done += len(frames)
            self._record_detector_fps(len(frames), _proc)

            self.inference_queue.put((batch_prediction_results, frames_batch, frame_num))
            if self.stop_requested:
                logger.debug("inference worker: inference_queue producer unblocked")
                break
        if eof:
            logger.debug("inference worker: stopped itself, EOF")
        else:
            logger.debug("inference worker: stopped by request")

    def _frame_detector_worker(self):
        logger.debug("frame detector worker: started")
        scenes: list[Scene] = []
        frame_num = self.start_frame
        eof = False
        while not (eof or self.stop_requested):
            inference_data = self.inference_queue.get()
            if self.stop_requested or inference_data is STOP_MARKER:
                logger.debug("frame detector worker: inference_queue consumer unblocked")
                break
            eof = inference_data is EOF_MARKER
            if eof:
                self._create_clips_for_completed_scenes(scenes, frame_num, eof=True)
                self.frame_detection_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("frame detector worker: frame_detection_queue producer unblocked")
                    break
                self.mosaic_clip_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
                    break
            else:
                batch_prediction_results, preprocessed_frames, _frame_num = inference_data
                assert frame_num == _frame_num, "frame detector worker out of sync with frame reader"
                assert len(preprocessed_frames) == len(batch_prediction_results)
                for i, results in enumerate(batch_prediction_results):
                    self._create_or_append_scenes_based_on_prediction_result(results, scenes, frame_num)
                    num_scenes_containing_frame = len([scene for scene in scenes if scene.frame_start <= frame_num <= scene.frame_end])
                    self.frame_detection_queue.put((frame_num, num_scenes_containing_frame))
                    if self.stop_requested:
                        logger.debug("frame detector worker: frame_detection_queue producer unblocked")
                        break
                    queue_marker = self._create_clips_for_completed_scenes(scenes, frame_num, eof=False)
                    if queue_marker is STOP_MARKER:
                        break
                    frame_num += 1
                # Release MPS driver cached memory to prevent unbounded growth
                if self.device is not None and self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        if eof:
            logger.debug("frame detector worker: stopped itself, EOF")
        else:
            logger.debug("frame detector worker: stopped by request")
