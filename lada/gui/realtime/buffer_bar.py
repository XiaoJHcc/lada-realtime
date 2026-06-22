# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
BufferBar: a horizontal strip visualizing the realtime AI buffer window.

The strip represents the frame window [playhead, playhead + window_frames). The AI restorer
produces frames sequentially, so the "ready" set is a single contiguous interval
[ready_start_frame, ready_end_frame) (ready_end = the restorer's output/production frontier).
The bar fills the part of the window covered by that interval with the accent color (ready)
and leaves the rest dark (not ready). Because ready_end advances even while playback is
paused, the ready segment keeps growing toward the window end when paused — a direct view of
the buffer filling up.

Two-state colouring: ready = accent, not-ready = dark track (the cold-start original-frame
lead-in [playhead, playhead+clip) is simply "not ready" here, no separate colour).

Drawing mirrors watch/timeline.py (Gtk.Snapshot + Graphene.Rect + Gsk.RoundedRect + theme
colours from Adw.StyleManager).
"""

from gi.repository import Gtk, GObject, Gdk, Graphene, Gsk, Adw


class BufferBar(Gtk.Widget):
    __gtype_name__ = 'BufferBar'

    @GObject.Property(type=Adw.StyleManager)
    def style_manager(self):
        return self._style_manager

    @style_manager.setter
    def style_manager(self, value):
        self._style_manager = value
        self.queue_draw()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._style_manager: Adw.StyleManager | None = None
        # window_frames <= 0 means "no data yet" -> draw empty track
        self._window_frames = 0
        self._playhead_frame = 0
        self._ready_start_frame = 0
        self._ready_end_frame = 0
        self.set_hexpand(True)
        # Min width so the bar stays usable as an AdwActionRow suffix; hexpand lets it grow
        # into the right half of the row. Height matches the other rows' suffix labels.
        self.set_size_request(160, 20)

    def update_buffer(self, window_frames: int, playhead_frame: int,
                      ready_start_frame: int, ready_end_frame: int):
        """Set the window + ready interval (all in frame numbers) and repaint. ready_end is the
        restorer's production frontier; ready_start its start frame. None-safe via int()."""
        self._window_frames = max(0, int(window_frames or 0))
        self._playhead_frame = int(playhead_frame or 0)
        self._ready_start_frame = int(ready_start_frame or 0)
        self._ready_end_frame = int(ready_end_frame or 0)
        self.queue_draw()

    def do_snapshot(self, s: Gtk.Snapshot):
        allocation = self.get_allocation()
        width = allocation.width
        height = allocation.height
        if width <= 0 or height <= 0:
            return

        track_color, ready_color = self._get_colors()
        border_radius = 6

        # rounded track background = "not ready"
        clip_rect = Graphene.Rect().init(0, 0, width, height)
        rounded = Gsk.RoundedRect()
        rounded.init_from_rect(clip_rect, border_radius)
        s.push_rounded_clip(rounded)
        s.append_color(track_color, clip_rect)

        # ready segment = window ∩ [ready_start, ready_end), mapped to [playhead, playhead+window)
        if self._window_frames > 0:
            win_start = self._playhead_frame
            win_end = self._playhead_frame + self._window_frames
            seg_start = max(win_start, self._ready_start_frame)
            seg_end = min(win_end, self._ready_end_frame)
            if seg_end > seg_start:
                x0 = (seg_start - win_start) / self._window_frames * width
                x1 = (seg_end - win_start) / self._window_frames * width
                ready_rect = Graphene.Rect().init(x0, 0, max(1.0, x1 - x0), height)
                s.append_color(ready_color, ready_rect)

        s.pop()

    def _get_colors(self) -> tuple[Gdk.RGBA, Gdk.RGBA]:
        if self._style_manager:
            accent = self._style_manager.get_accent_color()
            uses_dark_scheme = bool(self._style_manager.get_dark())
        else:
            accent = Adw.AccentColor.BLUE
            uses_dark_scheme = False

        # libadwaita API change (see timeline.py): to_rgba() may or may not take itself
        try:
            ready_color = accent.to_rgba()
        except TypeError:
            ready_color = accent.to_rgba(accent)

        track_color = Gdk.RGBA()
        if uses_dark_scheme:
            track_color.parse("#ffffff1a")
        else:
            track_color.parse("#0000001a")
        return track_color, ready_color


GObject.type_register(BufferBar)
