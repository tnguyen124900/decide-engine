"""Shared File-panel behaviour for the model editors (Markov, PSM, Microsim,
ABM, DES, DynaTrans, NMA), matching the Decision Tree Editor:

* Save     -- writes the current model straight to <folder>/<file name>,
              overwriting without asking (like any desktop app).
* Save as  -- a browser download (works on a server too).
* Open     -- from a "Saved models" list of the chosen folder.
* Autosave -- a background copy in autosave/<prefix>-autosave.json (5
              rotating backups) and a Restore banner when the app is
              reopened with unsaved work from an earlier session.

Each editor supplies three small callbacks:
``serialize() -> str | None`` (the file text, None when there is no model
yet), ``open_text(text) -> str | None`` (load a file's text into the
editor's session state, returning an error message on failure) and
``default_filename``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
from typing import Callable

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOSAVE_DIR = os.path.join(APP_DIR, "autosave")
MODELS_DIR = os.path.join(APP_DIR, "models")
AUTOSAVE_KEEP = 5


def _fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# DECIDE model files are saved as ``.decide`` (the content is still plain JSON).
# Legacy ``.json`` model files are still listed and opened.
MODEL_EXT = ".decide"
MODEL_EXTS = (".decide", ".json")

# Set by the desktop launcher: File commands come from the native menu bar, so the
# sidebar Save/Open panel is not drawn.
DESKTOP_MODE = os.environ.get("DECIDE_DESKTOP") == "1"


class ModelFiles:
    def __init__(
        self,
        prefix: str,
        label: str,
        default_filename: str | Callable[[], str],
        serialize: Callable[[], str | None],
        open_text: Callable[[str], str | None],
    ) -> None:
        self.prefix = prefix
        self.label = label
        self._default_filename = default_filename
        self._serialize = serialize
        self._open_text = open_text
        self.autosave_file = os.path.join(AUTOSAVE_DIR, f"{prefix}-autosave.json")
        self.settings_file = os.path.join(AUTOSAVE_DIR, f"{prefix}-settings.json")
        self.default_folder = os.path.join(MODELS_DIR, prefix)

    # -- session keys ---------------------------------------------------
    def _k(self, name: str) -> str:
        return f"{self.prefix}_files_{name}"

    def default_filename(self) -> str:
        return self._default_filename() if callable(self._default_filename) else self._default_filename

    @staticmethod
    def safe_filename(name: str, fallback: str) -> str:
        base = os.path.basename((name or "").strip().replace("\\", "/")) or fallback
        base = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base)
        low = base.lower()
        if low.endswith(MODEL_EXT):
            return base
        if low.endswith(".json"):
            base = base[: -len(".json")]
        return base + MODEL_EXT

    # -- folder ---------------------------------------------------------
    def _models_dir(self) -> str:
        try:
            with open(self.settings_file, encoding="utf-8") as handle:
                saved = json.load(handle).get("models_dir")
            if saved:
                return str(saved)
        except (OSError, ValueError, AttributeError):
            pass
        return self.default_folder

    def _remember_folder(self, folder: str) -> None:
        try:
            os.makedirs(AUTOSAVE_DIR, exist_ok=True)
            with open(self.settings_file, "w", encoding="utf-8") as handle:
                json.dump({"models_dir": folder}, handle)
        except OSError:
            pass

    def _folder(self) -> str:
        return os.path.expanduser((st.session_state.get(self._k("folder")) or self._models_dir()).strip())

    def _list_models(self) -> list[str]:
        try:
            names = [n for n in os.listdir(self._folder()) if n.lower().endswith(MODEL_EXTS) and not n.startswith(".")]
            return sorted(names, key=lambda n: os.path.getmtime(os.path.join(self._folder(), n)), reverse=True)
        except OSError:
            return []

    def _target_path(self) -> str:
        name = self.safe_filename(st.session_state.get(self._k("filename"), ""), self.default_filename())
        return os.path.join(self._folder(), name)

    # -- callbacks ------------------------------------------------------
    def _write(self, path: str) -> None:
        text = self._serialize()
        if text is None:
            st.session_state[self._k("message")] = ("error", "There is no model to save yet.")
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        except OSError as exc:
            st.session_state[self._k("message")] = ("error", f"Could not save to {path}: {exc}")
            return
        self._remember_folder(os.path.dirname(path))
        st.session_state[self._k("current")] = path
        st.session_state[self._k("filename")] = os.path.basename(path)
        st.session_state[self._k("saved_fp")] = _fingerprint(text)
        st.session_state[self._k("message")] = ("success", f"Saved to {path}")

    def _cb_save(self) -> None:
        current = st.session_state.get(self._k("current"))
        self._write(current if current else self._target_path())

    def _cb_open(self) -> None:
        choice = st.session_state.get(self._k("choice"))
        if not choice:
            return
        self.open_path(os.path.join(self._folder(), choice))

    # -- public helpers (used by the desktop menu bar) --------------------
    def save_current(self) -> None:
        self._cb_save()

    def save_to(self, path: str) -> None:
        self._write(path)

    def open_path(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            st.session_state[self._k("message")] = ("error", f"Could not open {path}: {exc}")
            return
        error = self._open_text(text)
        if error:
            st.session_state[self._k("message")] = ("error", f"Could not open {path}: {error}")
            return
        self._remember_folder(os.path.dirname(path))
        st.session_state[self._k("current")] = path
        st.session_state[self._k("filename")] = os.path.basename(path)
        st.session_state[self._k("saved_pending")] = True  # saved_fp is taken at the next autosave_step
        st.session_state[self._k("message")] = ("success", f"Opened {path}")

    # -- panel ----------------------------------------------------------
    def render_panel(self, *, with_save_as: bool = True) -> None:
        """Save button, file name, Save as (download), Folder, Saved models."""

        if DESKTOP_MODE:
            return  # File > Open / Save / Save As live in the native menu bar

        st.session_state.setdefault(self._k("filename"), self.default_filename())
        st.session_state.setdefault(self._k("folder"), self._models_dir())
        message = st.session_state.pop(self._k("message"), None)
        current = st.session_state.get(self._k("current"))
        st.caption(f"Current file: {current}" if current else "Current file: not saved yet")
        st.button(
            "Save", type="primary", use_container_width=True, key=self._k("save_button"),
            icon=":material/save:", on_click=self._cb_save,
            help="Overwrites the current file. A new model is saved as <folder>/<file name> below.",
        )
        st.text_input("Filename", key=self._k("filename"))
        if with_save_as:
            text = self._serialize()
            if text is not None:
                st.download_button(
                    "Save as", text, self.safe_filename(st.session_state.get(self._k("filename"), ""), self.default_filename()),
                    "application/json", use_container_width=True, key=self._k("save_as_button"),
                    icon=":material/download:",
                    help="Downloads a copy to your computer (choose the name and folder in the browser's save dialog).",
                )
        st.text_input("Folder", key=self._k("folder"), help="Where Save and Open look. Remembered for next time.")
        saved = self._list_models()
        if saved:
            st.selectbox("Saved models", saved, key=self._k("choice"))
            st.button("Open saved model", use_container_width=True, key=self._k("open_button"), on_click=self._cb_open)
        if message:
            kind, text_ = message
            (st.success if kind == "success" else st.error)(text_)

    # -- autosave -------------------------------------------------------
    def _autosave_write(self, text: str) -> None:
        try:
            os.makedirs(AUTOSAVE_DIR, exist_ok=True)
            if os.path.exists(self.autosave_file):
                for index in range(AUTOSAVE_KEEP - 1, 0, -1):
                    older = os.path.join(AUTOSAVE_DIR, f"{self.prefix}-autosave-{index}.json")
                    if os.path.exists(older):
                        shutil.move(older, os.path.join(AUTOSAVE_DIR, f"{self.prefix}-autosave-{index + 1}.json"))
                shutil.move(self.autosave_file, os.path.join(AUTOSAVE_DIR, f"{self.prefix}-autosave-1.json"))
            tmp = self.autosave_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, self.autosave_file)
        except OSError:
            pass

    def autosave_step(self) -> None:
        """Call once per run (after the model was edited): writes the model to
        autosave/ when it changed since the last write, and shows the status
        line. The untouched starting state is never autosaved."""

        text = self._serialize()
        if text is None:
            return
        fingerprint = _fingerprint(text)
        initial = st.session_state.setdefault(self._k("initial_fp"), fingerprint)
        if st.session_state.pop(self._k("saved_pending"), False):
            st.session_state[self._k("saved_fp")] = fingerprint
        if st.session_state.get(self._k("autosave_fp")) != fingerprint:
            st.session_state[self._k("autosave_fp")] = fingerprint
            if fingerprint != initial:
                self._autosave_write(text)
                st.session_state[self._k("autosave_time")] = datetime.datetime.now().strftime("%H:%M:%S")
        if st.session_state.get(self._k("autosave_time")):
            st.caption(f"Autosaved {st.session_state[self._k('autosave_time')]} (autosave/{self.prefix}-autosave.json)")

    def is_dirty(self) -> bool:
        """True when the model differs from the last saved / opened file (or from
        the untouched starting model when it was never saved)."""
        current = st.session_state.get(self._k("autosave_fp"))
        if current is None:
            return False
        base = st.session_state.get(self._k("saved_fp"), st.session_state.get(self._k("initial_fp")))
        return current != base

    def _read_autosave(self) -> str | None:
        try:
            with open(self.autosave_file, encoding="utf-8") as handle:
                text = handle.read()
            json.loads(text)
            return text
        except (OSError, ValueError):
            return None

    def restore_banner(self) -> None:
        """Once per session offer the autosave of an earlier session."""

        if not st.session_state.get(self._k("checked")):
            st.session_state[self._k("checked")] = True
            text = self._read_autosave()
            current = self._serialize()
            if text and (current is None or _fingerprint(text) != _fingerprint(current)):
                st.session_state[self._k("pending")] = text
        pending = st.session_state.get(self._k("pending"))
        if not pending:
            return
        try:
            when = datetime.datetime.fromtimestamp(os.path.getmtime(self.autosave_file)).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            when = ""
        if DESKTOP_MODE:  # sidebar is hidden in the desktop app: show the offer as a centred dialog
            _bk = f"div.st-key-{self._k('banner')}"
            st.markdown(
                f"""<style>
                section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
                    transform: none !important; will-change: auto !important; filter: none !important; contain: none !important; }}
                {_bk} {{ position: fixed !important; top: 50% !important; left: 50% !important;
                    transform: translate(-50%, -50%) !important; width: min(640px, 92vw) !important;
                    z-index: 1000002 !important; visibility: visible !important; background: #ffffff; padding: 12px 20px !important; box-sizing: border-box !important;
                    box-shadow: 0 0 0 100vmax rgba(15, 30, 60, 0.45), 0 12px 40px rgba(0,0,0,0.35); }}
                </style>""",
                unsafe_allow_html=True,
            )
        with st.container(border=True, key=self._k("banner")):
            st.markdown(f"**:material/restore: Unsaved {self.label} work from the last session was found** · {when}")
            restore_col, discard_col, _ = st.columns([1, 1, 3])
            with restore_col:
                if st.button("Restore", type="primary", icon=":material/restore:", key=self._k("restore"),
                             use_container_width=True):
                    error = self._open_text(pending)
                    st.session_state[self._k("pending")] = None
                    if error:
                        st.error(f"Could not restore: {error}")
                    else:
                        st.session_state[self._k("current")] = None
                        restored = self._serialize()
                        if restored is not None:
                            st.session_state[self._k("autosave_fp")] = _fingerprint(restored)
                        st.rerun()
            with discard_col:
                if st.button("Discard", icon=":material/delete:", key=self._k("discard"), use_container_width=True):
                    try:
                        os.replace(self.autosave_file, os.path.join(AUTOSAVE_DIR, f"{self.prefix}-autosave-discarded.json"))
                    except OSError:
                        pass
                    st.session_state[self._k("pending")] = None
                    st.rerun()
