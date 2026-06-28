# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
Baseline presentation test: play a video through a VANILLA GStreamer pipeline
(filesrc/decodebin -> videoconvert -> gtk4paintablesink) in a plain GTK4 window, with the
SAME presentation-layer instrumentation lada's realtime path uses (GdkFrameClock tick +
paintable invalidate-contents). NO appsrc, NO AI pipeline, NO clock-driven push.

Purpose: isolate whether the on-screen judder measured in the realtime view is caused by
lada's pipeline (appsrc bursts / in-process AI contention) or by the gtk4paintablesink + GTK
render path on this machine. If THIS baseline presents the same 30fps file smoothly at the
monitor refresh (clean tick rate, one dominant "refreshes held per frame"), the judder is
lada-specific. If it juders too, it's the sink/GTK/platform render path.

Mirrors the realtime manager's Windows workaround (#62): no glsinkbin on win32.

Usage:
    .venv/Scripts/python.exe scripts/gtk_sink_baseline.py <video> [seconds]

Writes the same trace files as the realtime tracer (present/ticks/sinkstats + summary) to
realtime_trace_baseline/ via LADA_REALTIME_TRACE.
"""

import os
import sys
import time

os.environ.setdefault("LADA_REALTIME_TRACE", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "realtime_trace_baseline")))

# import lada.gui FIRST: prepare_windows_gui_environment() wires the GTK/Gst DLLs onto PATH at
# import time. Importing gi.repository before this fails with "DLL load failed importing _gi".
import lada.gui  # noqa: F401,E402
from lada.gui.realtime import realtime_trace  # noqa: E402

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gtk, Gst, GLib, Gdk  # noqa: E402

Gst.init(None)


def main():
    video = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 70.0
    if not video or not os.path.exists(video):
        print("usage: gtk_sink_baseline.py <video> [seconds]")
        return 2

    tracer = realtime_trace.get_tracer()
    if tracer is None:
        print("tracing disabled (set LADA_REALTIME_TRACE)")
        return 2
    tracer.set_fps_target(29.97)  # test_video is 29.97; only used for the summary's target line

    app = Gtk.Application(application_id="io.github.ladaapp.lada.baseline")

    state = {}

    def on_activate(a):
        win = Gtk.ApplicationWindow(application=a)
        win.set_default_size(1280, 720)
        picture = Gtk.Picture()
        # LADA_BASELINE_SPINNER=1: add a continuously-spinning sibling widget. A spinner animates
        # at the display refresh rate, which keeps the whole window's GdkFrameClock at 150Hz (vs
        # the ~30Hz it self-throttles to when only the 30fps video is animating). This reproduces
        # the realtime view's condition to test whether a refresh-rate sibling animation alone
        # turns the smooth baseline into the realtime judder.
        if realtime_trace._env_truthy(os.environ.get("LADA_BASELINE_SPINNER")):
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            sp = Gtk.Spinner()
            sp.start()
            sp.set_size_request(32, 32)
            box.append(sp)
            picture.set_vexpand(True)
            box.append(picture)
            win.set_child(box)
        else:
            win.set_child(picture)

        pipeline = Gst.Pipeline.new("baseline")
        src = Gst.ElementFactory.make("filesrc", None)
        src.set_property("location", video)
        dbin = Gst.ElementFactory.make("decodebin", None)
        conv = Gst.ElementFactory.make("videoconvert", None)
        gtksink = Gst.ElementFactory.make("gtk4paintablesink", None)
        paintable = gtksink.get_property("paintable")
        picture.set_paintable(paintable)

        for e in (src, dbin, conv, gtksink):
            pipeline.add(e)
        src.link(dbin)
        conv.link(gtksink)

        # LADA_BASELINE_AUDIO=1: also play the file's audio through autoaudiosink. The audio sink
        # then becomes the pipeline clock provider and the video sink slaves to it -- reproducing
        # the realtime view's clock situation (audio present) WITHOUT the AI pipeline, to test
        # whether slaving video to the audio clock is what makes frame release uneven (31/47ms).
        want_audio = realtime_trace._env_truthy(os.environ.get("LADA_BASELINE_AUDIO"))
        if want_audio:
            aconv = Gst.ElementFactory.make("audioconvert", None)
            ares = Gst.ElementFactory.make("audioresample", None)
            asink = Gst.ElementFactory.make("autoaudiosink", None)
            for e in (aconv, ares, asink):
                pipeline.add(e)
            aconv.link(ares); ares.link(asink)
            state["asink_pad"] = aconv.get_static_pad("sink")

        def on_pad_added(_dbin, pad):
            caps = pad.get_current_caps() or pad.query_caps()
            name = caps.get_structure(0).get_name() if caps else ""
            if name.startswith("video"):
                pad.link(conv.get_static_pad("sink"))
            elif want_audio and name.startswith("audio") and not state["asink_pad"].is_linked():
                pad.link(state["asink_pad"])
        dbin.connect("pad-added", on_pad_added)

        state["pipeline"] = pipeline
        state["sink"] = gtksink

        # identical instrumentation to lada's realtime view
        paintable.connect("invalidate-contents",
                          lambda *_a: tracer.record_newframe(time.perf_counter_ns()))

        def on_tick(widget, frame_clock):
            tracer.record_tick(frame_clock.get_frame_time())
            return GLib.SOURCE_CONTINUE
        picture.add_tick_callback(on_tick)

        def poll_stats():
            try:
                st = gtksink.get_property("stats")
                def g(name, d=0):
                    try:
                        v = st.get_value(name)
                        return v if v is not None else d
                    except Exception:
                        return d
                tracer.record_sink_stats(time.perf_counter_ns(), int(g("rendered", 0)),
                                         int(g("dropped", 0)), float(g("average-rate", 0.0)))
            except Exception:
                pass
            tracer.maybe_log_rolling_summary()
            return True
        GLib.timeout_add(500, poll_stats)

        win.present()
        pipeline.set_state(Gst.State.PLAYING)

        def quit_now():
            tracer.dump()
            pipeline.set_state(Gst.State.NULL)
            a.quit()
            return False
        GLib.timeout_add(int(seconds * 1000), quit_now)

    app.connect("activate", on_activate)
    return app.run([])


if __name__ == "__main__":
    sys.exit(main())
