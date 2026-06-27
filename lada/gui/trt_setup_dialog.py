# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
#
# First-run guided dialog for building the BasicVSR++ TensorRT acceleration
# engines. The engines must be compiled locally on first use (a one-time,
# multi-minute, uncancellable torch_tensorrt.compile of 6 sub-engines). This
# dialog moves that compile from "the moment the user opens a video" to "right
# after the app starts", and lets a multi-GPU user pick which card to build for
# (that card also becomes the inference device).
#
# Shown from MainWindow.on_realize. Returns immediately (showing nothing) when
# TRT is disabled, the model isn't basicvsrpp, there's no fp16 CUDA GPU, or the
# engines already exist — so a user who has compiled once never sees it again.
import logging
import pathlib
import threading

import torch
from gi.repository import Adw, Gtk, GLib

from lada import LOG_LEVEL, ModelFiles
from lada.gui import utils
from lada.gui.utils import get_available_gpus

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

here = pathlib.Path(__file__).parent.resolve()


def _cuda_gpus() -> list[tuple[str, str]]:
    return [(dev, name) for dev, name in get_available_gpus() if dev.startswith("cuda:")]


def maybe_show_trt_setup_dialog(parent_window, config) -> None:
    """Show the first-run TRT build dialog if (and only if) it's warranted.

    No-op (returns without showing anything) when:
    - TRT is disabled via LADA_BASICVSRPP_TRT=0,
    - the restoration model isn't a basicvsrpp model (only those have engines),
    - fp16 is off or no CUDA GPU is present (TRT path needs both),
    - the engines for the current device already exist (compiled before).

    The no-fp16 / no-GPU cases still show a brief informational notice so the
    user understands acceleration is unavailable; everything else is silent.
    """
    from lada.restorationpipeline import BASICVSRPP_TRT_ENABLED, BASICVSRPP_TRT_MAX_CLIP_SIZE
    from lada.restorationpipeline.trt_engine_paths import all_basicvsrpp_sub_engines_exist

    if not BASICVSRPP_TRT_ENABLED:
        return

    model_name = config.mosaic_restoration_model
    if not model_name or not model_name.startswith("basicvsrpp"):
        return

    modelfile = ModelFiles.get_restoration_model_by_name(model_name)
    if modelfile is None:
        return
    model_path = modelfile.path

    cuda_gpus = _cuda_gpus()
    has_cuda = bool(cuda_gpus) and torch.cuda.is_available()
    fp16 = bool(config.fp16_enabled)

    # Engines already built for the configured device? Then never bother the user.
    if has_cuda and fp16:
        try:
            device = torch.device(config.device)
        except Exception:
            device = torch.device(cuda_gpus[0][0])
        if device.type == "cuda" and all_basicvsrpp_sub_engines_exist(model_path, True, BASICVSRPP_TRT_MAX_CLIP_SIZE, device):
            return

    dialog = TrtSetupDialog(config=config, model_path=model_path,
                            cuda_gpus=cuda_gpus, has_cuda=has_cuda, fp16=fp16,
                            max_clip_size=BASICVSRPP_TRT_MAX_CLIP_SIZE)
    dialog.present(parent_window)


@Gtk.Template(string=utils.translate_ui_xml(here / 'trt_setup_dialog.ui'))
class TrtSetupDialog(Adw.Dialog):
    __gtype_name__ = 'TrtSetupDialog'

    stack: Gtk.Stack = Gtk.Template.Child()
    label_prompt: Gtk.Label = Gtk.Template.Child()
    label_progress: Gtk.Label = Gtk.Template.Child()
    drop_down_gpu: Gtk.DropDown = Gtk.Template.Child()
    box_prompt_buttons: Gtk.Box = Gtk.Template.Child()
    button_build: Gtk.Button = Gtk.Template.Child()
    button_later: Gtk.Button = Gtk.Template.Child()
    button_ok: Gtk.Button = Gtk.Template.Child()

    def __init__(self, *, config, model_path: str, cuda_gpus: list[tuple[str, str]],
                 has_cuda: bool, fp16: bool, max_clip_size: int, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._model_path = model_path
        self._cuda_gpus = cuda_gpus
        self._max_clip_size = max_clip_size

        if not has_cuda:
            # No Nvidia GPU: inform and offer only OK (closes -> PyTorch path).
            self._make_notice(_("No Nvidia GPU detected. GPU acceleration is unavailable; the slower PyTorch path will be used."))
        elif not fp16:
            self._make_notice(_("TensorRT acceleration requires FP16, which is currently disabled. The PyTorch path will be used. Enable FP16 in settings to build acceleration engines."))
        elif len(cuda_gpus) > 1:
            self._make_build_prompt(multi=True)
        else:
            self._make_build_prompt(multi=False)

    # ── prompt construction ──────────────────────────────────────────────
    def _make_notice(self, text: str):
        self.label_prompt.set_text(text)
        self.box_prompt_buttons.set_visible(False)
        self.button_ok.set_visible(True)

    def _make_build_prompt(self, *, multi: bool):
        if multi:
            self.label_prompt.set_text(_(
                "Select the GPU to accelerate, then build its engines. This one-time "
                "setup may take several minutes and cannot be cancelled. The selected "
                "GPU will also be used for processing."))
            strings = Gtk.StringList()
            self.drop_down_gpu.set_model(strings)
            selected_idx = 0
            for i, (device, name) in enumerate(self._cuda_gpus):
                strings.append(name)
                if device == self._config.device:
                    selected_idx = i
            self.drop_down_gpu.set_selected(selected_idx)
            self.drop_down_gpu.set_visible(True)
        else:
            device, name = self._cuda_gpus[0]
            self.label_prompt.set_text(_(
                "Build GPU acceleration engines for {gpu}? Only needs to be built once, "
                "may take several minutes, and cannot be interrupted.").format(gpu=name))

    # ── selected device ──────────────────────────────────────────────────
    def _selected_device(self) -> str:
        if len(self._cuda_gpus) > 1:
            idx = self.drop_down_gpu.get_selected()
            if 0 <= idx < len(self._cuda_gpus):
                return self._cuda_gpus[idx][0]
        return self._cuda_gpus[0][0]

    # ── button callbacks ─────────────────────────────────────────────────
    @Gtk.Template.Callback()
    def button_ok_clicked_callback(self, _button):
        self.close()

    @Gtk.Template.Callback()
    def button_later_clicked_callback(self, _button):
        # Skip for now: PyTorch path this session; the dialog reappears next
        # startup because the engines still won't exist (until built once).
        self.close()

    @Gtk.Template.Callback()
    def button_build_clicked_callback(self, _button):
        device_str = self._selected_device()
        # The chosen card is both the compile target and the inference device.
        if self._config.device != device_str:
            self._config.device = device_str
        self._start_build(device_str)

    # ── compilation ──────────────────────────────────────────────────────
    def _start_build(self, device_str: str):
        from lada.restorationpipeline.progress import set_load_progress_callback

        # Lock the dialog into the uncancellable progress state.
        self.set_can_close(False)
        self.stack.set_visible_child_name("progress")

        set_load_progress_callback(
            lambda msg: GLib.idle_add(lambda: self.label_progress.set_text(msg))
        )

        def run():
            ok = False
            err = None
            try:
                from lada.restorationpipeline.basicvsrpp_trt_compilation import basicvsrpp_startup_policy
                ok = basicvsrpp_startup_policy(
                    restoration_model_path=self._model_path,
                    device=torch.device(device_str), fp16=True,
                    compile_basicvsrpp=True, max_clip_size=self._max_clip_size,
                    optimization_level=5,
                )
            except Exception as e:  # noqa: BLE001 - report to user, never crash GUI
                err = e
                logger.warning("TRT engine build failed: %s", e)
            GLib.idle_add(lambda: self._on_build_done(ok, err))

        threading.Thread(target=run, daemon=True).start()

    def _on_build_done(self, ok: bool, err):
        from lada.restorationpipeline.progress import clear_load_progress_callback
        clear_load_progress_callback()
        self.set_can_close(True)
        if ok:
            # Engines ready; opening a video will load them without recompiling.
            self.close()
        else:
            msg = _("Building acceleration engines failed. The slower PyTorch path will be used.")
            if err is not None:
                msg = _("Building acceleration engines failed ({error}). The slower PyTorch path will be used.").format(error=err)
            self.stack.set_visible_child_name("prompt")
            self._make_notice(msg)
        return False

    def present(self, parent):
        super().present(parent)
