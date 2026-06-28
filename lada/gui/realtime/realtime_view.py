# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
Clock-driven realtime preview view.

A trimmed-down sibling of WatchView (lada/gui/watch/watch_view.py). It reuses the same
widgets (Timeline, ConfigSidebar, the video Picture) and the FrameRestorerProvider, but
drives playback through RealtimePipelineManager so the player never pauses to buffer:
when the AI restorer can't keep up the original (passthrough) frame is shown instead.

Milestone 1 intentionally omits subtitles, mute, fullscreen, seek-preview thumbnails and
the no-GPU banner that WatchView has, to keep the surface small.
"""

import logging
import pathlib
import threading
import time

from gi.repository import Gtk, GObject, GLib, Gio, Gst, Adw, Gdk, Graphene

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config
from lada.gui.config.config_sidebar import ConfigSidebar
from lada.gui.frame_restorer_provider import FrameRestorerProvider, FrameRestorerOptions, FRAME_RESTORER_PROVIDER, FrameRestorerOptionsBuilder
from lada.gui.realtime.gstreamer_pipeline_manager_realtime import RealtimePipelineManager
from lada.gui.realtime import realtime_trace
from lada.gui.watch.gstreamer_pipeline_manager import PipelineState
from lada.gui.watch.headerbar_files_drop_down import HeaderbarFilesDropDown
from lada.restorationpipeline.progress import set_load_progress_callback, clear_load_progress_callback
from lada.gui.watch.overlay_elements_controller import OverlayElementsController
from lada.gui.watch.seek_preview_popover import SeekPreviewPopover
from lada.gui.watch.timeline import Timeline
from lada.gui.shortcuts import ShortcutsManager
from lada.utils import video_utils

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


@Gtk.Template(string=utils.translate_ui_xml(here / 'realtime_view.ui'))
class RealtimeView(Gtk.Widget):
    __gtype_name__ = 'RealtimeView'

    button_play_pause = Gtk.Template.Child()
    picture_video_player: Gtk.Picture = Gtk.Template.Child()
    widget_timeline: Timeline = Gtk.Template.Child()
    button_image_play_pause = Gtk.Template.Child()
    label_current_time = Gtk.Template.Child()
    label_cursor_time = Gtk.Template.Child()
    box_playback_controls: Gtk.Box = Gtk.Template.Child()
    box_video_player = Gtk.Template.Child()
    box_header_bar_banner = Gtk.Template.Child()
    drop_down_files: HeaderbarFilesDropDown = Gtk.Template.Child()
    spinner_overlay = Gtk.Template.Child()
    label_loading_status: Gtk.Label = Gtk.Template.Child()
    config_sidebar: ConfigSidebar = Gtk.Template.Child()
    stack_video_player: Gtk.Stack = Gtk.Template.Child()
    view_switcher: Adw.ViewSwitcher = Gtk.Template.Child()
    button_open_files: Gtk.Button = Gtk.Template.Child()
    toggle_button_pane: Gtk.ToggleButton = Gtk.Template.Child()
    header_bar: Adw.HeaderBar = Gtk.Template.Child()
    button_toggle_fullscreen: Gtk.Button = Gtk.Template.Child()
    button_toggle_fullscreen_overlay: Gtk.Button = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None
        self._video_preview_init_done = False
        self._shortcuts_manager: ShortcutsManager | None = None

        self.seek_preview_popover = SeekPreviewPopover()
        self.seek_preview_popover.set_parent(self.box_playback_controls)
        self._last_seek_preview_timestamp_ns = 0
        self._last_seek_preview_mouse_x = 0.0
        self._video_thumbnailer: video_utils.VideoThumbnailer | None = None
        self._thumbnailer_lock = threading.Lock()
        self._thread_counter = 0
        self._thread_counter_lock = threading.Lock()
        self._thumbnail_size = (220, 124)

        self.pipeline_connection_handler_ids = []
        self.eos = False

        self.frame_restorer_provider: FrameRestorerProvider = FRAME_RESTORER_PROVIDER
        self.file_duration_ns = 0
        self.frame_duration_ns = None
        self.files: list[Gio.File] = []
        self.video_metadata: video_utils.VideoMetadata | None = None
        self.should_be_paused = False
        self.seek_in_progress = False
        self.appsource_worker_reset_requested = False

        self._config: Config | None = None
        self._view_stack: Adw.ViewStack | None = None

        self.widget_timeline.connect('seek_requested', lambda widget, seek_position: self.seek_video(seek_position))
        self.widget_timeline.connect('cursor_position_changed', lambda widget, cursor_position, x: self.show_cursor_position(cursor_position if cursor_position >= 0 else None, x if x >= 0 else None))

        self.overlay_elements_controller: OverlayElementsController = OverlayElementsController(self, [self.box_playback_controls, self.box_header_bar_banner, self.button_toggle_fullscreen_overlay])

        self.pipeline_manager: RealtimePipelineManager | None = None
        self.stack_video_player.set_visible_child_name("spinner")

        self.drop_down_selected_handler_id = self.drop_down_files.connect("notify::selected", lambda obj, spec: self.play_file(obj.get_property(spec.name)))

        self.setup_double_click_fullscreen()

        drop_target = utils.create_files_drop_target(lambda files: self.emit("files-opened", files), lambda files: None)
        self.add_controller(drop_target)

        def on_files_opened(obj, files):
            self.button_open_files.set_sensitive(True)
            self.add_files(files)
            if self._video_preview_init_done:
                last_file_idx = len(self.files) - 1
                if self.drop_down_files.get_selected() != last_file_idx:
                    self.drop_down_files.handler_block(self.drop_down_selected_handler_id)
                    self.drop_down_files.set_selected(last_file_idx)
                    self.drop_down_files.handler_unblock(self.drop_down_selected_handler_id)
                    self.play_file(last_file_idx)
            else:
                self.drop_down_files.set_sensitive(False)
        self.connect("files-opened", on_files_opened)

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        self.setup_config_signal_handlers()

    @GObject.Property(type=ShortcutsManager)
    def shortcuts_manager(self):
        return self._shortcuts_manager

    @shortcuts_manager.setter
    def shortcuts_manager(self, value):
        self._shortcuts_manager = value
    @GObject.Property(type=Adw.ViewStack)
    def view_stack(self):
        return self._view_stack

    @view_stack.setter
    def view_stack(self, value: Adw.ViewStack):
        self._view_stack = value
        def on_visible_child_name_changed(object, spec):
            visible_child_name = object.get_property(spec.name)
            if visible_child_name != "realtime":
                self.should_be_paused = True
                self.pause_if_currently_playing()
            else:
                if not self._video_preview_init_done:
                    self.play_file(0)
                elif self.appsource_worker_reset_requested:
                    self.reset_appsource_worker()
                self.config_sidebar.init_sidebar_from_config(self._config)
        self._view_stack.connect("notify::visible-child-name", on_visible_child_name_changed)

    @GObject.Signal(name="files-opened", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_opened_signal(self, files: list[Gio.File]):
        pass

    @GObject.Signal(name="toggle-fullscreen-requested")
    def toggle_fullscreen_requested(self):
        pass

    @GObject.Signal(name="window-resize-requested", arg_types=(Gdk.Paintable,))
    def video_size_changed(self, paintable: Gdk.Paintable):
        pass

    @Gtk.Template.Callback()
    def button_toggle_fullscreen_callback(self, button_clicked):
        self.emit("toggle-fullscreen-requested")

    @Gtk.Template.Callback()
    def button_play_pause_callback(self, button_clicked):
        if not self._video_preview_init_done or self.seek_in_progress:
            return
        if self.pipeline_manager.state == PipelineState.PLAYING:
            self.should_be_paused = True
            self.pipeline_manager.pause()
        elif self.pipeline_manager.state == PipelineState.PAUSED:
            self.should_be_paused = False
            if self.eos:
                self.seek_video(0)
            self.pipeline_manager.play()
        else:
            logger.warning(f"unhandled pipeline state in button_play_pause_callback: {self.pipeline_manager.state}")

    @Gtk.Template.Callback()
    def button_open_files_callback(self, button_clicked):
        self.button_open_files.set_sensitive(False)
        callback = lambda files: self.emit("files-opened", files)
        dismissed_callback = lambda *args: self.button_open_files.set_sensitive(True)
        utils.show_open_files_dialog(callback, dismissed_callback)

    @Gtk.Template.Callback()
    def toggle_button_pane_clicked_callback(self, button_clicked: Gtk.ToggleButton):
        is_sidebar_open = button_clicked.get_active()
        self.overlay_elements_controller.on_sidebar_opened(is_sidebar_open)

    @property
    def frame_restorer_options(self):
        return self._frame_restorer_options

    @frame_restorer_options.setter
    def frame_restorer_options(self, value: FrameRestorerOptions):
        if self._frame_restorer_options == value:
            return
        self._frame_restorer_options = value
        if self._video_preview_init_done:
            if self._view_stack.props.visible_child_name == "realtime":
                self.reset_appsource_worker()
            else:
                self.appsource_worker_reset_requested = True

    def setup_config_signal_handlers(self):
        def on_show_mosaic_detections(*args):
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).mosaic_detection(self._config.show_mosaic_detections).build()
        self._config.connect("notify::show-mosaic-detections", on_show_mosaic_detections)

        def on_device(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).device(self._config.device).build()
        self._config.connect("notify::device", on_device)

        def on_mosaic_restoration_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).mosaic_restoration_model_name(self._config.mosaic_restoration_model).build()
        self._config.connect("notify::mosaic-restoration-model", on_mosaic_restoration_model)

        def on_mosaic_detection_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).mosaic_detection_model_name(self._config.mosaic_detection_model).build()
        self._config.connect("notify::mosaic-detection-model", on_mosaic_detection_model)

        def on_realtime_clip_length(object, spec):
            # Realtime uses its own clip length, not the shared max_clip_duration. Clip length
            # changes the detector's clip grouping + queue sizing, so the pipeline must restart.
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).max_clip_length(self._config.realtime_clip_length).build()
            if self.pipeline_manager is not None:
                self.pipeline_manager.set_realtime_clip_frames(self._config.realtime_clip_length)
        self._config.connect("notify::realtime-clip-length", on_realtime_clip_length)

        def on_realtime_max_regions(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).realtime_max_regions(self._config.realtime_max_regions).build()
        self._config.connect("notify::realtime-max-regions", on_realtime_max_regions)

        def on_fp16_enabled(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).fp16_enabled(self._config.fp16_enabled).build()
        self._config.connect("notify::fp16-enabled", on_fp16_enabled)

        def on_detect_face_mosaics(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = FrameRestorerOptionsBuilder(self.frame_restorer_options).detect_face_mosaics(self._config.detect_face_mosaics).build()
        self._config.connect("notify::detect-face-mosaics", on_detect_face_mosaics)

        def on_realtime_lookahead_frames(object, spec):
            if self.pipeline_manager is not None:
                self.pipeline_manager.set_lookahead_frames(self._config.realtime_lookahead_frames)
        self._config.connect("notify::realtime-lookahead-frames", on_realtime_lookahead_frames)

        def on_realtime_cold_start_clips(object, spec):
            # Cold-start lead only affects the next cold start / seek start point, not the
            # running pipeline, so no restart needed.
            if self.pipeline_manager is not None:
                self.pipeline_manager.set_cold_start_clips(self._config.realtime_cold_start_clips)
        self._config.connect("notify::realtime-cold-start-clips", on_realtime_cold_start_clips)

    def seek_video(self, seek_position_ns):
        if self.seek_in_progress:
            return
        self.eos = False
        self.seek_in_progress = True
        self.label_current_time.set_text(self.get_time_label_text(seek_position_ns))
        self.widget_timeline.set_property("playhead-position", seek_position_ns)
        self.pipeline_manager.seek_async(seek_position_ns)
        self.seek_in_progress = False

    def setup_double_click_fullscreen(self):
        click_gesture = Gtk.GestureClick()
        def on_click(click_obj, n_press, x, y):
            if n_press == 2:
                self.emit("toggle-fullscreen-requested")
        click_gesture.connect("pressed", on_click)
        self.box_video_player.add_controller(click_gesture)

    def on_fullscreened(self, fullscreened: bool):
        if fullscreened:
            self.header_bar.set_visible(False)
            self.button_toggle_fullscreen_overlay.set_visible(True)
            self.button_toggle_fullscreen.set_property("icon-name", "view-restore-symbolic")
            self.button_toggle_fullscreen_overlay.set_property("icon-name", "view-restore-symbolic")
            self.button_play_pause.grab_focus()
            self.box_video_player.set_css_classes(["fullscreen-video-player"])
            self.toggle_button_pane.set_active(False)
        else:
            self.header_bar.set_visible(True)
            self.button_toggle_fullscreen_overlay.set_visible(False)
            self.button_toggle_fullscreen.set_property("icon-name", "view-fullscreen-symbolic")
            self.button_toggle_fullscreen_overlay.set_property("icon-name", "view-fullscreen-symbolic")
            self.button_play_pause.grab_focus()
            self.box_video_player.remove_css_class("fullscreen-video-player")

    def on_window_focused(self, focused: bool):
        if focused:
            self.button_play_pause.grab_focus()
        self.overlay_elements_controller.on_window_focused(focused)

    def show_cursor_position(self, cursor_position_ns: int | None, x: float | None):
        if x is not None and cursor_position_ns is not None:
            if self._config.seek_preview_enabled:
                self.label_cursor_time.set_visible(False)
                if self._should_update_seek_preview(cursor_position_ns, x):
                    self.update_seek_preview(cursor_position_ns, x)
            else:
                self.label_cursor_time.set_visible(True)
                label_text = self.get_time_label_text(cursor_position_ns)
                self.label_cursor_time.set_text(label_text)
                self.seek_preview_popover.popdown()
        else:
            self.label_cursor_time.set_visible(False)
            self.seek_preview_popover.popdown()

    def _get_seek_preview_popover_pointing_rect(self, mouse_x_in_timeline: float) -> Gdk.Rectangle | None:
        success, transformed_point = self.widget_timeline.compute_point(self.box_playback_controls, Graphene.Point().init(mouse_x_in_timeline, 0))
        if success:
            mouse_x_in_controls = transformed_point.x
        else:
            logger.error(f"Couldn't convert cursor coordinates from timeline to controls box: x: {mouse_x_in_timeline}")
            return None

        controls_width = self.box_playback_controls.get_allocated_width()
        popover_width = self._thumbnail_size[0] + 18

        spacing = 8

        pointing_rect = Gdk.Rectangle()
        pointing_rect.x = int(mouse_x_in_controls - popover_width // 2)
        pointing_rect.x = max(spacing, min(pointing_rect.x, controls_width - popover_width - spacing))

        timeline_allocation = self.widget_timeline.get_allocation()
        y_offset = 5
        pointing_rect.y = timeline_allocation.y - y_offset

        pointing_rect.width = popover_width
        pointing_rect.height = 1

        return pointing_rect

    def _should_update_seek_preview(self, timestamp_ns: int, mouse_x: float):
        time_delta_ns = abs(timestamp_ns - self._last_seek_preview_timestamp_ns)
        position_delta = abs(mouse_x - self._last_seek_preview_mouse_x)

        time_threshold_ns = 2 * Gst.SECOND
        position_threshold = 10

        return time_delta_ns > time_threshold_ns or position_delta > position_threshold

    def update_seek_preview(self, timestamp_ns: int, mouse_x: float):
        self._last_seek_preview_timestamp_ns = timestamp_ns
        self._last_seek_preview_mouse_x = mouse_x

        time_text = self.get_time_label_text(timestamp_ns)
        self.seek_preview_popover.set_text(time_text)
        self.seek_preview_popover.show_spinner()
        pointing_rect = self._get_seek_preview_popover_pointing_rect(mouse_x)
        if pointing_rect is None:
            return
        self.seek_preview_popover.set_pointing_to(pointing_rect)
        self.seek_preview_popover.popup()

        def generate_thumbnail(current_thread_id):
            with self._thumbnailer_lock:
                with self._thread_counter_lock:
                    if current_thread_id < self._thread_counter:
                        return

                if self._video_thumbnailer is None:
                    self._video_thumbnailer = video_utils.VideoThumbnailer(self.video_metadata.video_file, thumb_width=self._thumbnail_size[0], thumb_height=self._thumbnail_size[1])
                    self._video_thumbnailer.open()

                thumbnail = self._video_thumbnailer.get_thumbnail(timestamp_ns)
                self.seek_preview_popover.set_thumbnail(thumbnail)

        with self._thread_counter_lock:
            self._thread_counter += 1
            threading.Thread(target=generate_thumbnail, args=(self._thread_counter,), daemon=True).start()

    def close_thumbnailer(self):
        with self._thumbnailer_lock:
            self._thread_counter += 1
            if self._video_thumbnailer:
                self._video_thumbnailer.close()
                self._video_thumbnailer = None

    def play_file(self, idx):
        self._show_spinner()
        self._reinit_open_file_async(self.files[idx])

    def add_files(self, files: list[Gio.File]):
        unique_files_to_add = []
        for file_to_add in files:
            if any(file_to_add.get_path() == file_already_added.get_path() for file_already_added in self.files):
                continue
            self.files.append(file_to_add)
            unique_files_to_add.append(file_to_add)

        if len(unique_files_to_add) > 0:
            self.drop_down_files.handler_block(self.drop_down_selected_handler_id)
            self.drop_down_files.add_files(files)
            self.drop_down_files.handler_unblock(self.drop_down_selected_handler_id)

    def _reinit_open_file_async(self, file: Gio.File):
        def run():
            if self._video_preview_init_done:
                for id in self.pipeline_connection_handler_ids: self.pipeline_manager.handler_block(id)
                self._video_preview_init_done = False
                self.pipeline_manager.close_video_file()
                self.close_thumbnailer()
                self.seek_preview_popover.clear_thumbnail()
                for id in self.pipeline_connection_handler_ids: self.pipeline_manager.handler_unblock(id)
            video_metadata = video_utils.get_video_meta_data(file.get_path())
            GLib.idle_add(lambda: self._open_file(video_metadata))
        threading.Thread(target=run, daemon=True).start()

    def _open_file(self, video_metadata: video_utils.VideoMetadata):
        assert not self._video_preview_init_done
        self.video_metadata = video_metadata
        self.frame_restorer_options = FrameRestorerOptions(self.config.mosaic_restoration_model,
                                                           self.config.mosaic_detection_model, self.video_metadata,
                                                           self.config.device,
                                                           self.config.realtime_clip_length,
                                                           self.config.show_mosaic_detections,
                                                           False,
                                                           self.config.fp16_enabled,
                                                           self.config.detect_face_mosaics,
                                                           self.config.realtime_max_regions)

        self.should_be_paused = False
        self.seek_in_progress = False

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        self.widget_timeline.set_property("duration", self.file_duration_ns)

        self.frame_restorer_provider.init(self._frame_restorer_options)

        if self.pipeline_manager:
            self.pipeline_manager.init_pipeline(self.video_metadata, None)
            self.pipeline_manager.set_realtime_clip_frames(self.config.realtime_clip_length)
            self.pipeline_manager.set_lookahead_frames(self.config.realtime_lookahead_frames)
            self.pipeline_manager.set_cold_start_clips(self.config.realtime_cold_start_clips)
        else:
            # min_thresh=0: no pre-buffering on either the video or audio queue (clock-driven)
            self.pipeline_manager = RealtimePipelineManager(self.frame_restorer_provider, 0, 1.0, self.config.mute_audio, self.config.subtitles_font_size)
            self.pipeline_manager.init_pipeline(self.video_metadata, None)
            self.pipeline_manager.set_realtime_clip_frames(self.config.realtime_clip_length)
            self.pipeline_manager.set_lookahead_frames(self.config.realtime_lookahead_frames)
            self.pipeline_manager.set_cold_start_clips(self.config.realtime_cold_start_clips)
            self.picture_video_player.set_paintable(self.pipeline_manager.paintable)
            self._install_frame_clock_keepalive()
            self._install_presentation_trace()
            self.pipeline_connection_handler_ids = [
                self.pipeline_manager.connect("paintable-size-changed", lambda obj: GLib.idle_add(lambda: self.emit("window-resize-requested", self.pipeline_manager.paintable))),
                self.pipeline_manager.connect("eos", lambda obj: GLib.idle_add(lambda: self.on_eos())),
                self.pipeline_manager.connect("notify::state", lambda obj, spec: GLib.idle_add(lambda: self.on_pipeline_state(obj.get_property(spec.name)))),
            ]
            GLib.timeout_add(100, self.update_current_position)
            GLib.timeout_add(500, self.update_diagnostics)

        def play():
            logger.debug("Finished opening file, play realtime pipeline...")
            self.pipeline_manager.play()
        threading.Thread(target=play).start()

    def _install_frame_clock_keepalive(self):
        """Keep the video widget's GdkFrameClock running so gtk4paintablesink paces frame
        presentation by the display refresh instead of a coarse timer.

        ROOT-CAUSE FIX for the realtime "judder / uneven frame rate". gtk4paintablesink hands each
        new frame to the GdkPaintable and relies on the widget's GdkFrameClock to schedule the
        on-screen update. With nothing else animating the widget, GTK lets the frame clock go idle
        between frames; the sink then falls back to a coarse timed wait whose granularity on
        Windows is the 15.625ms system timer. A 33.37ms (29.97fps) frame can't be paced on a
        15.625ms grid, so it alternates 2x (31.25ms) and 3x (46.875ms) quanta -- a measured hard
        70%/16% bimodal at exactly those values -> persistent stutter. Registering a no-op tick
        callback keeps the frame clock ticking at the monitor refresh, so presentation is paced by
        the refresh clock and the bimodal collapses (measured: frames held >43ms drop from ~16% to
        <1%). Cross-platform and independent of CPU load (capping CPU threads changed nothing);
        costs only a no-op main-thread wake per refresh while the realtime view is shown."""
        if getattr(self, "_frame_clock_keepalive_id", None) is not None:
            return
        try:
            self._frame_clock_keepalive_id = self.picture_video_player.add_tick_callback(
                lambda _widget, _clock: GLib.SOURCE_CONTINUE)
        except Exception as e:
            logger.warning(f"realtime frame-clock keepalive install failed: {e}")

    def _install_presentation_trace(self):
        """Diagnostics (LADA_REALTIME_TRACE only): measure the TRUE on-screen cadence, not just
        the GStreamer sink. Two GTK-main-thread probes:
          - paintable 'invalidate-contents' -> one event per frame actually handed to the display
          - the widget's GdkFrameClock tick -> one event per screen-refresh opportunity
        Together they show how many refreshes each frame is held for (even == smooth)."""
        if not realtime_trace.TRACE_ENABLED:
            return
        tracer = realtime_trace.get_tracer()
        if tracer is None:
            return
        try:
            paintable = self.pipeline_manager.paintable
            if paintable is not None:
                paintable.connect("invalidate-contents",
                                  lambda *_a: tracer.record_newframe(time.perf_counter_ns()))
            # The tick callback keeps a handle on the frame clock; to rule out that it perturbs
            # the clock rate, LADA_TRACE_NO_TICKS=1 skips it (present/invalidate-contents jitter
            # is measured independently and is the real on-screen-cadence signal).
            if not realtime_trace._env_truthy(__import__("os").environ.get("LADA_TRACE_NO_TICKS")):
                def _on_tick(widget, frame_clock):
                    tracer.record_tick(frame_clock.get_frame_time())
                    return GLib.SOURCE_CONTINUE
                self.picture_video_player.add_tick_callback(_on_tick)
        except Exception as e:
            logger.warning(f"realtime presentation trace install failed: {e}")

    def on_eos(self):
        self.eos = True
        self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
        tracer = realtime_trace.get_tracer()
        if tracer is not None:
            tracer.dump()

    def on_pipeline_state(self, state: PipelineState):
        if state == PipelineState.PLAYING:
            self.button_image_play_pause.set_property("icon-name", "media-playback-pause-symbolic")
        elif state == PipelineState.PAUSED:
            self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
        if not self._video_preview_init_done and state == PipelineState.PLAYING:
            self._video_preview_init_done = True
            self._show_video_preview()

    def pause_if_currently_playing(self):
        if not self._video_preview_init_done:
            return
        if self.pipeline_manager.state == PipelineState.PLAYING:
            self.should_be_paused = True
            self.pipeline_manager.pause()

    def grab_focus(self):
        self.button_play_pause.grab_focus()

    def reset_appsource_worker(self):
        self._show_spinner()
        self.appsource_worker_reset_requested = False
        self._video_preview_init_done = False
        self.frame_restorer_provider.init(self._frame_restorer_options)

        def reinit_pipeline():
            self.pipeline_manager.pause()
            self.pipeline_manager.reinit_appsrc()
            self.pipeline_manager.play()
        threading.Thread(target=reinit_pipeline).start()

    def update_current_position(self):
        position = self.pipeline_manager.get_position_ns()
        if position is not None:
            label_text = self.get_time_label_text(position)
            self.label_current_time.set_text(label_text)
            self.widget_timeline.set_property("playhead-position", position)
        return True

    def update_diagnostics(self):
        if not self._video_preview_init_done or self.pipeline_manager is None:
            return True
        stats = self.pipeline_manager.get_realtime_stats()
        if stats is not None:
            self.config_sidebar.update_diagnostics(stats)
        return True

    def get_time_label_text(self, time_ns):
        if not time_ns or time_ns == -1:
            return '00:00:00'
        seconds = int(time_ns / Gst.SECOND)
        minutes = int(seconds / 60)
        hours = int(minutes / 60)
        seconds = seconds % 60
        minutes = minutes % 60
        hours, minutes, seconds = int(hours), int(minutes), int(seconds)
        return f"{minutes}:{seconds:02d}" if hours == 0 else f"{hours}:{minutes:02d}:{seconds:02d}"

    def _show_spinner(self, *args):
        self.config_sidebar.set_property("disabled", True)
        self.drop_down_files.set_sensitive(False)
        self.view_switcher.set_sensitive(False)
        self.button_open_files.set_sensitive(False)
        self.stack_video_player.set_visible_child_name("spinner")
        self.overlay_elements_controller.on_spinner_visible(True)
        # Surface model-load / TRT-compile progress on the spinner label. The
        # callback fires on the background loading thread, so marshal to the
        # main loop. Reset to the default text first (covers the cached-engine
        # case where no compile progress is reported).
        self.label_loading_status.set_text(_("Loading models…"))
        set_load_progress_callback(
            lambda msg: GLib.idle_add(lambda: self.label_loading_status.set_text(msg))
        )

    def _show_video_preview(self, *args):
        clear_load_progress_callback()
        self.config_sidebar.set_property("disabled", False)
        self.drop_down_files.set_sensitive(True)
        self.view_switcher.set_sensitive(True)
        self.button_open_files.set_sensitive(True)
        self.stack_video_player.set_visible_child_name("video-player")
        self.overlay_elements_controller.on_spinner_visible(False)
        self.grab_focus()

    def close(self, block=False):
        if not self.pipeline_manager:
            return
        tracer = realtime_trace.get_tracer()
        if tracer is not None:
            tracer.dump()
        self._video_preview_init_done = False
        self.close_thumbnailer()
        if block:
            self.pipeline_manager.close_video_file()
        else:
            threading.Thread(target=self.pipeline_manager.close_video_file).start()
