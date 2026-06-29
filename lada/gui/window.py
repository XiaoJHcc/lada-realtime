# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
import threading
from math import sqrt

from gi.repository import Adw, Gtk, Gio, GLib, GObject, Gdk

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config
from lada.gui.export.export_view import ExportView
from lada.gui.fileselection.file_selection_view import FileSelectionView
from lada.gui.watch.watch_view import WatchView
from lada.gui.realtime.realtime_view import RealtimeView
from lada.gui.shortcuts import ShortcutsManager

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'window.ui'))
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    file_selection_view: FileSelectionView = Gtk.Template.Child()
    export_view: ExportView = Gtk.Template.Child()
    watch_view: WatchView = Gtk.Template.Child()
    realtime_view: RealtimeView = Gtk.Template.Child()
    view_stack: Adw.ViewStack = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @GObject.Property(type=ShortcutsManager)
    def shortcuts_manager(self):
        return self._shortcuts_manager

    @shortcuts_manager.setter
    def shortcuts_manager(self, value):
        self._shortcuts_manager = value
        self._setup_shortcuts()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._config: Config | None
        self._shortcuts_manager: ShortcutsManager | None = None

        self.set_title("Lada Realtime")

        self.connect("close-request", self.close)
        self.connect("realize", self.on_realize)
        self.file_selection_view.connect("files-selected", lambda obj, files: self.on_files_selected(files))
        self.watch_view.connect("toggle-fullscreen-requested", lambda *args: self.on_toggle_fullscreen())
        self.watch_view.connect("window-resize-requested", self.on_window_resize_requested)
        self.connect("notify::fullscreened", lambda object, spec: self.on_fullscreened(object.get_property(spec.name)))

        self.export_view.props.view_stack = self.view_stack
        self.export_view.connect("video-export-requested", lambda obj, restore_directory_or_file: self.on_video_export_requested(restore_directory_or_file))
        self.export_view.connect("shutdown-confirmation-requested", lambda *args: self.present())
        self.watch_view.props.view_stack = self.view_stack
        self.realtime_view.props.view_stack = self.view_stack
        self.realtime_view.connect("toggle-fullscreen-requested", lambda *args: self.on_toggle_fullscreen())
        self.realtime_view.connect("window-resize-requested", self.on_window_resize_requested)

        # Single-direction playback-position hand-off between Watch and Realtime tabs.
        # Connected after the views so their own visible-child-name handlers (pause leaving,
        # init entering) run first; we then read the leaving view's position and seek the
        # entering one to it once its pipeline is ready. Pipelines stay independent.
        self._previous_view_name = self.view_stack.props.visible_child_name
        self.view_stack.connect("notify::visible-child-name", self._on_view_changed_handoff)

        self.window_focused = False

    def on_video_export_requested(self, restore_directory_or_file: Gio.File):
        self.stack.props.visible_child_name = "main"
        self.view_stack.props.visible_child_name = "export"
        def run():
            self.watch_view.close(block=True)
            GLib.idle_add(lambda: self.export_view.start_export(restore_directory_or_file))
        threading.Thread(target=run).start()

    def on_files_selected(self, files: list[Gio.File]):
        self.stack.props.visible_child_name = "main"
        if self._config.initial_view in ("realtime", "watch", "export"):
            self.view_stack.props.visible_child_name = self._config.initial_view
        else:
            self.view_stack.props.visible_child_name = "realtime"
        self.watch_view.add_files(files)
        self.realtime_view.add_files(files)
        self.export_view.add_files(files)
        if self.view_stack.props.visible_child_name == "watch":
            self.watch_view.play_file(0)
        elif self.view_stack.props.visible_child_name == "realtime":
            self.realtime_view.play_file(0)
        self._previous_view_name = self.view_stack.props.visible_child_name

    def _handoff_view(self, name: str):
        if name == "watch":
            return self.watch_view
        elif name == "realtime":
            return self.realtime_view
        return None

    def _on_view_changed_handoff(self, obj, spec):
        new_name = obj.get_property(spec.name)
        old_name = self._previous_view_name
        self._previous_view_name = new_name

        from_view = self._handoff_view(old_name)
        to_view = self._handoff_view(new_name)
        if from_view is None or to_view is None or from_view is to_view:
            return

        # Read the leaving view's current playback position.
        if not getattr(from_view, "_video_preview_init_done", False) or from_view.pipeline_manager is None:
            return
        position_ns = from_view.pipeline_manager.get_position_ns()
        if position_ns is None or position_ns < 0:
            return

        # Apply it to the entering view, waiting for its pipeline to finish (re)initializing.
        # First-ever entry kicks off play_file(0) asynchronously, so poll until ready.
        attempts = {"n": 0}
        def apply_seek():
            attempts["n"] += 1
            if self.view_stack.props.visible_child_name != new_name:
                return False  # user switched away again; abandon
            if to_view._video_preview_init_done and to_view.pipeline_manager is not None and not to_view.seek_in_progress:
                to_view.seek_video(position_ns)
                return False
            if attempts["n"] > 100:  # ~10s safety cap
                return False
            return True
        GLib.timeout_add(100, apply_seek)

    def on_fullscreened(self, fullscreened: bool):
        if self.stack.props.visible_child_name == "main":
            if self.view_stack.props.visible_child_name == "watch":
                self.watch_view.on_fullscreened(fullscreened)
            elif self.view_stack.props.visible_child_name == "realtime":
                self.realtime_view.on_fullscreened(fullscreened)

    def on_toggle_fullscreen(self):
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def on_window_resize_requested(self, obj, paintable: Gdk.Paintable):
        if self.is_visible():
            self._resize_window(paintable)
        else:
            self.connect("map", self._resize_window, paintable, True)

    def on_realize(self, *_args) -> None:
        surface = self.get_surface()
        if not isinstance(surface, Gdk.Toplevel):
            return

        surface.connect("notify::state", self._on_toplevel_state_changed)

        # First-run guided TRT engine build. Deferred via idle_add so the window
        # paints its first frame before the dialog appears. No-op when engines
        # already exist / TRT disabled / no fp16 CUDA GPU (decided inside).
        if self._config is not None:
            def _show_trt_setup():
                from lada.gui.trt_setup_dialog import maybe_show_trt_setup_dialog
                maybe_show_trt_setup_dialog(self, self._config)
                return False
            GLib.idle_add(_show_trt_setup)


    def _on_toplevel_state_changed(self, toplevel: Gdk.Toplevel, *_args) -> None:
        focused = bool(toplevel.get_state() & Gdk.ToplevelState.FOCUSED)
        if focused == self.window_focused:
            return

        self.window_focused = focused
        if self.stack.props.visible_child_name == "main":
            if self.view_stack.props.visible_child_name == "watch":
                self.watch_view.on_window_focused(focused)
            elif self.view_stack.props.visible_child_name == "realtime":
                self.realtime_view.on_window_focused(focused)

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("ui", "UI")
        def switch_views(child_name):
            if self.stack.props.visible_child_name == "main":
                self.view_stack.set_visible_child_name(child_name)
        self._shortcuts_manager.add("ui", "show-export-view", "e", lambda *args: switch_views('export'), _("Switch to Export View"))
        self._shortcuts_manager.add("ui", "show-watch-view", "p", lambda *args: switch_views('watch'), _("Switch to Watch View"))

    def close(self, *args):
        self.watch_view.close()
        self.realtime_view.close()
        self.export_view.close()

    def _resize_window(self, paintable: Gdk.Paintable, initial: bool | None = False) -> None:
        # SPDX-SnippetBegin
        # SPDX-License-Identifier: GPL-3.0-or-later AND AGPL-3.0
        # SPDX-FileCopyrightText: Copyright 2024-2025 kramo
        # Code vendored from: https://gitlab.gnome.org/GNOME/showtime/-/blob/3c940ff2a4128a50c559985a04fb6beb7e9292e6/showtime/widgets/window.py

        # For large enough monitors, occupy 40% of the screen area
        # when opening a window with a video
        DEFAULT_OCCUPY_SCREEN = 0.4

        # Screens with this resolution or smaller are handled as small
        SMALL_SCREEN_AREA = 1280 * 1024

        # For small monitors, occupy 80% of the screen area
        SMALL_OCCUPY_SCREEN = 0.8

        SMALL_SIZE_CHANGE = 10

        logger.debug("Resizing window…")

        if initial:
            self.disconnect_by_func(self._resize_window)

        if not (video_width := paintable.get_intrinsic_width()) or not (
                video_height := paintable.get_intrinsic_height()
        ):
            return

        if not (surface := self.get_surface()):
            logger.error("Could not get GdkSurface to resize window")
            return

        if not (monitor := self.props.display.get_monitor_at_surface(surface)):
            logger.error("Could not get GdkMonitor to resize window")
            return

        video_area = video_width * video_height
        init_width, init_height = self.get_default_size()

        if initial:
            # Algorithm copied from Loupe
            # https://gitlab.gnome.org/GNOME/loupe/-/blob/4ca5f9e03d18667db5d72325597cebc02887777a/src/widgets/image/rendering.rs#L151

            hidpi_scale = surface.props.scale_factor

            monitor_rect = monitor.props.geometry

            monitor_width = monitor_rect.width
            monitor_height = monitor_rect.height

            monitor_area = monitor_width * monitor_height
            logical_monitor_area = monitor_area * pow(hidpi_scale, 2)

            occupy_area_factor = (
                SMALL_OCCUPY_SCREEN
                if logical_monitor_area <= SMALL_SCREEN_AREA
                else DEFAULT_OCCUPY_SCREEN
            )

            size_scale = sqrt(monitor_area / video_area * occupy_area_factor)

            target_scale = min(1, size_scale)
            nat_width = video_width * target_scale
            nat_height = video_height * target_scale

            # margin is estimated space for Dock or Taskbar. In some OS these can also be placed left/right of the monitor so use it for both width/height
            margin = 100
            max_width = monitor_width - margin * hidpi_scale
            if nat_width > max_width:
                nat_width = max_width
                nat_height = video_height * nat_width / video_width

            max_height = monitor_height - margin * hidpi_scale
            if nat_height > max_height:
                nat_height = max_height
                nat_width = video_width * nat_height / video_height

        else:
            prev_area = init_width * init_height

            if video_width > video_height:
                ratio = video_width / video_height
                nat_width = int(sqrt(prev_area * ratio))
                nat_height = int(nat_width / ratio)
            else:
                ratio = video_height / video_width
                nat_width = int(sqrt(prev_area / ratio))
                nat_height = int(nat_width * ratio)

            if (abs(init_width - nat_width) < SMALL_SIZE_CHANGE) and (
                    abs(init_height - nat_height) < SMALL_SIZE_CHANGE
            ):
                return

        nat_width = round(nat_width)
        nat_height = round(nat_height)

        for prop, init, target in (
                ("default-width", init_width, nat_width),
                ("default-height", init_height, nat_height),
        ):
            anim = Adw.TimedAnimation.new(
                self, init, target, 500, Adw.PropertyAnimationTarget.new(self, prop)
            )
            anim.props.easing = Adw.Easing.EASE_OUT_EXPO
            (anim.skip if initial else anim.play)()
            logger.debug("Resized window to %ix%i", nat_width, nat_height)

        # SPDX-SnippetEnd