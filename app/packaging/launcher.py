"""DECIDE desktop launcher (demo).

Starts the Streamlit server in a child process and shows it in a native
window (pywebview) with a menu bar. The bundled app files are copied to a
user-writable folder (~/DECIDE/app) so autosave/ and models/ work; user
data in ~/DECIDE/app/models and ~/DECIDE/app/autosave is never overwritten.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_NAME = "DECIDE"
USER_HOME_DIR = Path.home() / "DECIDE"
KEEP_USER_DIRS = ("models", "autosave")
COMMAND_FILE = USER_HOME_DIR / "command.json"
STATE_FILE = USER_HOME_DIR / "state.json"
GITHUB_URL = "https://github.com/tnguyen124900/decide-engine"


def bundle_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def payload_dir() -> Path:
    """Folder holding app.py inside the bundle (or the repo when run from source)."""
    if getattr(sys, "frozen", False):
        return bundle_dir() / "app_payload"
    return Path(__file__).resolve().parent.parent


def sync_payload() -> Path:
    """Copy bundled app files to ~/DECIDE/app, keeping the user's models/autosave."""
    src = payload_dir()
    if not getattr(sys, "frozen", False):
        return src  # running from source: use the repo directly
    dst = USER_HOME_DIR / "app"
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.name in KEEP_USER_DIRS and target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return dst


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_streamlit(app_dir: str, port: int) -> None:
    """Child process: run the Streamlit server in its main thread."""
    os.chdir(app_dir)
    sys.path.insert(0, app_dir)
    sys.path.insert(0, os.path.join(app_dir, "src"))
    from streamlit.web import bootstrap

    flags = {
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
        "server.fileWatcherType": "none",
    }
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(os.path.join(app_dir, "app.py"), False, [], flags)


def wait_for_port(port: int, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.4)
    return False


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        os.system(f'open "{path}"')
    elif sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        os.system(f'xdg-open "{path}"')


WINDOWS: dict = {}  # window id ("w1", "w2", ...) -> pywebview window; one Streamlit session each


def command_file(wid: str) -> Path:
    return COMMAND_FILE.with_name(f"command-{wid}.json")


def state_file(wid: str) -> Path:
    return STATE_FILE.with_name(f"state-{wid}.json")


def active_wid() -> str | None:
    """The window (tab) the user is looking at: visible + focused, else visible, else any."""
    visible = None
    for wid, win in list(WINDOWS.items()):
        try:
            seen = str(win.evaluate_js("document.visibilityState + ',' + document.hasFocus()"))
        except Exception:  # noqa: BLE001 - window closing or not loaded yet
            continue
        if seen == "visible,true":
            return wid
        if seen.startswith("visible") and visible is None:
            visible = wid
    return visible or next(iter(WINDOWS), None)


def send_command(cmd: str, path: str | None = None, model: str | None = None, wid: str | None = None) -> None:
    """Hand a menu command to the Streamlit session of one window (the active one by default)."""
    wid = wid or active_wid()
    if wid is None:
        return
    payload = {"cmd": cmd}
    if path:
        payload["path"] = path
    if cmd in ("goto", "setting", "language"):
        payload["model"] = model
    target = command_file(wid)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, target)


def read_state(wid: str | None = None) -> dict:
    wid = wid or active_wid()
    try:
        return json.loads(state_file(wid).read_text(encoding="utf-8")) if wid else {}
    except (OSError, ValueError):
        return {}



# Windows / Linux keyboard shortcuts: injected into each page (see make_window).
SHORTCUT_HANDLERS: dict = {}
KEYS_JS = """
(function () {
  if (window.__decideKeys) { return; }
  window.__decideKeys = true;
  var isLinux = /Linux/.test(navigator.platform) && !/Android/.test(navigator.userAgent);
  document.addEventListener('keydown', function (e) {
    if (!e.ctrlKey || e.altKey || e.metaKey) { return; }
    var k = (e.key || '').toLowerCase(), name = null;
    if (k === 's') { name = e.shiftKey ? 'save_as' : 'save'; }
    else if (!e.shiftKey && k === 'o') { name = 'open'; }
    else if (!e.shiftKey && k === 'n') { name = 'new'; }
    else if (!e.shiftKey && k === 'w') { name = 'close'; }
    else if (!e.shiftKey && k === 'q' && isLinux) { name = 'quit'; }
    if (!name) { return; }
    e.preventDefault(); e.stopPropagation();
    if (window.pywebview && window.pywebview.api && window.pywebview.api.shortcut) {
      window.pywebview.api.shortcut(name);
    }
  }, true);
})();
"""

FORCE_CLOSE: set = set()  # windows allowed to close without asking
ASKING: set = set()  # windows with a "save changes?" question open


def confirm_save() -> str:
    """Native Yes / No / Cancel question. Returns "yes", "no" or "cancel"."""
    text = "Do you want to save the changes you made to this model?"
    detail = "Your changes will be lost if you don't save them."
    try:
        if sys.platform == "darwin":
            script = (
                f'display dialog "{text}" & return & "{detail}" with title "DECIDE" '
                'buttons {"Cancel", "No", "Yes"} default button "Yes" cancel button "Cancel" with icon caution'
            )
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if out.returncode != 0:
                return "cancel"
            return "yes" if "Yes" in out.stdout else "no"
        if sys.platform.startswith("win"):
            import ctypes

            res = ctypes.windll.user32.MessageBoxW(0, f"{text}\n{detail}", "DECIDE", 0x03 | 0x30 | 0x10000)
            return {6: "yes", 7: "no"}.get(res, "cancel")
    except Exception:  # noqa: BLE001
        pass
    return "cancel"


def dialog_kind(name: str):
    import webview

    enum = getattr(webview, "FileDialog", None)
    return getattr(enum, name) if enum is not None else getattr(webview, name + "_DIALOG")


def models_dir() -> str:
    d = USER_HOME_DIR / "app" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def ask_and_save(wid: str) -> bool:
    """Ask about unsaved changes of one window. True = it may be closed now."""
    win = WINDOWS.get(wid)
    if win is None:
        return True
    if not read_state(wid).get("dirty"):
        return True
    choice = confirm_save()
    if choice == "cancel":
        return False
    if choice == "no":
        return True
    state = read_state(wid)
    if state.get("current"):
        send_command("save", wid=wid)
    else:
        kind = state.get("model_type") or "model"
        picked = win.create_file_dialog(
            dialog_kind("SAVE"), directory=models_dir(), save_filename=f"{kind}-model.decide",
            file_types=("DECIDE models (*.decide)",),
        )
        if not picked:
            return False  # cancelled the Save dialog: back to the model
        path = str(picked[0] if isinstance(picked, (list, tuple)) else picked)
        if not path.lower().endswith(".decide"):
            path += ".decide"
        send_command("save_as", path, wid=wid)
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.3)
        if not read_state(wid).get("dirty"):
            return True
    return False  # could not confirm the save: keep the tab open rather than lose work


WIN_TITLES: dict = {}
PLACEHOLDER = "DECIDE New Model"


def free_placeholder_title() -> str:
    """'DECIDE New Model', then 'DECIDE New Model_1', _2 ... (lowest unused)."""
    used = set(WIN_TITLES.values())
    if PLACEHOLDER not in used:
        return PLACEHOLDER
    n = 1
    while f"{PLACEHOLDER}_{n}" in used:
        n += 1
    return f"{PLACEHOLDER}_{n}"


def desired_title(wid: str) -> str | None:
    cur = read_state(wid).get("current")
    if cur:
        return Path(str(cur)).stem or None
    return None


def title_watcher() -> None:
    """Once a model has a file name, the window/tab title follows it; an unsaved
    model keeps its 'DECIDE New Model[_n]' placeholder."""
    while True:
        time.sleep(1.0)
        for wid, win in list(WINDOWS.items()):
            try:
                want = desired_title(wid)
                if want is None:
                    want = WIN_TITLES.get(wid) if WIN_TITLES.get(wid, "").startswith(PLACEHOLDER) else None
                    if want is None:
                        continue
                if WIN_TITLES.get(wid) != want:
                    win.set_title(want)
                    WIN_TITLES[wid] = want
            except Exception:  # noqa: BLE001 - cosmetic only
                pass


def close_flow(wid: str) -> None:
    if wid in ASKING:
        return
    ASKING.add(wid)
    try:
        if ask_and_save(wid):
            FORCE_CLOSE.add(wid)
            win = WINDOWS.get(wid)
            if win is not None:
                win.destroy()
    finally:
        ASKING.discard(wid)


def on_closing(wid: str):
    """pywebview 'closing' hook: returning False keeps the window open."""
    if wid in FORCE_CLOSE or not read_state(wid).get("dirty"):
        return True
    threading.Thread(target=close_flow, args=(wid,), daemon=True).start()
    return False


def main() -> None:
    multiprocessing.freeze_support()
    app_dir = sync_payload()
    # Every spawned child (the Streamlit server below, and any parallel
    # PSA worker ProcessPoolExecutor spawns from it) inherits this, so a
    # frozen worker can put the app's modules back on its import path --
    # see psa_worker_init().
    os.environ["DECIDE_APP_DIR"] = str(app_dir)
    port = free_port()
    USER_HOME_DIR.mkdir(parents=True, exist_ok=True)
    for stale in list(USER_HOME_DIR.glob("command*.json")) + list(USER_HOME_DIR.glob("state*.json")):
        stale.unlink(missing_ok=True)
    os.environ["DECIDE_DESKTOP"] = "1"
    os.environ["DECIDE_COMMAND_FILE"] = str(COMMAND_FILE)
    os.environ["DECIDE_STATE_FILE"] = str(STATE_FILE)
    server = multiprocessing.Process(target=run_streamlit, args=(str(app_dir), port), daemon=True)
    server.start()

    import webview
    from webview.menu import Menu, MenuAction, MenuSeparator

    url = f"http://127.0.0.1:{port}"
    if not wait_for_port(port):
        server.terminate()
        raise SystemExit("DECIDE server did not start")

    from urllib.parse import quote

    counter = {"n": 0}

    def make_window(open_path: str | None = None):
        counter["n"] += 1
        wid = f"w{counter['n']}"
        target = f"{url}/?w={wid}" + (f"&open={quote(open_path)}" if open_path else "")
        title = free_placeholder_title()
        win = webview.create_window(title, target, width=1440, height=900, min_size=(900, 600))
        WINDOWS[wid] = win
        WIN_TITLES[wid] = title
        win.events.closed += lambda wid=wid: (WINDOWS.pop(wid, None), WIN_TITLES.pop(wid, None))

        def _on_closing(*_a, wid=wid):
            return on_closing(wid)

        win.events.closing += _on_closing

        if sys.platform != "darwin":
            # Windows / Linux: the native menu has no accelerator keys, so the page
            # itself catches Ctrl+N/O/S/W (and Ctrl+Shift+S, Ctrl+Q on Linux) and
            # calls back into Python. macOS gets real menu key equivalents instead.
            def shortcut(name: str, _wid=wid) -> None:
                fn = SHORTCUT_HANDLERS.get("fn")
                if fn:
                    fn(name)

            win.expose(shortcut)

            def _install_keys(_win=win) -> None:
                try:
                    _win.evaluate_js(KEYS_JS)
                except Exception:  # noqa: BLE001 - shortcuts are a convenience only
                    pass

            win.events.loaded += _install_keys
        return wid, win

    _, window = make_window()
    threading.Thread(target=title_watcher, daemon=True).start()

    def cur():
        wid = active_wid()
        return WINDOWS.get(wid, window)

    def models_dir() -> str:
        d = USER_HOME_DIR / "app" / "models"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def dialog_kind(name: str):
        enum = getattr(webview, "FileDialog", None)  # pywebview >= 6
        if enum is not None:
            return getattr(enum, name)
        return getattr(webview, name + "_DIALOG")

    def open_model():
        picked = cur().create_file_dialog(
            dialog_kind("OPEN"), directory=models_dir(), allow_multiple=False,
            file_types=("DECIDE models (*.decide;*.json)", "All files (*.*)"),
        )
        if picked:
            path = str(picked[0] if isinstance(picked, (list, tuple)) else picked)
            wid = active_wid()
            if wid and not read_state(wid).get("model_type"):
                send_command("open_any", path, wid=wid)  # current tab is empty (model picker): reuse it
            else:
                new_tab(path)  # never overwrite the model in the current tab

    def save_model_as():
        kind = read_state().get("model_type") or "model"
        picked = cur().create_file_dialog(
            dialog_kind("SAVE"), directory=models_dir(), save_filename=f"{kind}-model.decide",
            file_types=("DECIDE models (*.decide)",),
        )
        if picked:
            path = str(picked[0] if isinstance(picked, (list, tuple)) else picked)
            if not path.lower().endswith(".decide"):
                path += ".decide"
            send_command("save_as", path)

    def save_model():
        if read_state().get("current"):
            send_command("save")
        else:
            save_model_as()  # never saved yet: ask where

    def reload_app():
        wid = active_wid()
        if wid in WINDOWS:
            WINDOWS[wid].load_url(f"{url}/?w={wid}")

    def quit_app():
        def work() -> None:
            for wid in list(WINDOWS):
                if not ask_and_save(wid):
                    return
            for wid, win in list(WINDOWS.items()):
                FORCE_CLOSE.add(wid)
                win.destroy()

        threading.Thread(target=work, daemon=True).start()

    def close_tab():
        wid = active_wid()
        if wid:
            threading.Thread(target=close_flow, args=(wid,), daemon=True).start()

    def js(code: str):
        return lambda: cur().evaluate_js(code)

    def paste():
        cur().evaluate_js("navigator.clipboard.readText().then(t=>document.execCommand('insertText',false,t))")

    def go(model: str | None):
        return lambda: send_command("goto", model=model)

    def setting(panel: str):
        return lambda: send_command("setting", model=panel)

    def new_tab(open_path: str | None = None):
        """File > New model: a new window (own Streamlit session, starts at the
        model-type picker). Other windows keep their models untouched. On macOS
        the new window is attached as a native tab of the current window."""
        _, new_win = make_window(open_path)
        if sys.platform == "darwin":
            def attach() -> None:
                time.sleep(1.5)
                try:
                    from AppKit import NSApp
                    from PyObjCTools import AppHelper

                    def work() -> None:
                        wins = [w for w in NSApp.windows() if str(w.title()) in set(WIN_TITLES.values()) and w.isVisible()]
                        if len(wins) >= 2:
                            for w in wins:
                                w.setTabbingMode_(2)  # NSWindowTabbingModePreferred
                            wins[0].addTabbedWindow_ordered_(wins[-1], 1)
                            wins[-1].makeKeyAndOrderFront_(None)

                    AppHelper.callAfter(work)
                except Exception:  # noqa: BLE001 - falls back to a separate window
                    pass

            import threading
            threading.Thread(target=attach, daemon=True).start()
        return new_win

    def run_shortcut(name: str) -> None:
        actions = {"new": new_tab, "open": open_model, "save": save_model, "save_as": save_model_as,
                   "close": close_tab, "quit": quit_app}
        action = actions.get(name)
        if action:
            action()

    SHORTCUT_HANDLERS["fn"] = run_shortcut

    file_menu = Menu("File", [
        MenuAction("New model", lambda: new_tab()),
        MenuAction("Clear current model...", lambda: send_command("new")),
        MenuAction("Open...", open_model),
        MenuSeparator(),
        MenuAction("Save", save_model),
        MenuAction("Save As...", save_model_as),
        MenuAction("Export report...", lambda: send_command("report")),
        MenuAction("Batch run...", lambda: send_command("batch")),
        MenuSeparator(),
        MenuAction("Open data folder", lambda: open_folder(USER_HOME_DIR)),
        MenuAction("Reload", reload_app),
        MenuSeparator(),
        MenuAction("Close tab", close_tab),
        MenuAction("Quit", quit_app),
    ])
    edit_menu = Menu("Edit", [
        MenuAction("Undo", js("document.execCommand('undo')")),
        MenuAction("Redo", js("document.execCommand('redo')")),
        MenuSeparator(),
        MenuAction("Cut", js("document.execCommand('cut')")),
        MenuAction("Copy", js("document.execCommand('copy')")),
        MenuAction("Paste", paste),
        MenuAction("Select All", js("document.execCommand('selectAll')")),
    ])
    view_menu = Menu("View", [
        MenuAction("Zoom in", js("document.body.style.zoom=(parseFloat(document.body.style.zoom||1)+0.1)")),
        MenuAction("Zoom out", js("document.body.style.zoom=(parseFloat(document.body.style.zoom||1)-0.1)")),
        MenuAction("Actual size", js("document.body.style.zoom=1")),
        MenuSeparator(),
        MenuAction("Show / hide node numbers", setting("node_numbers")),
        MenuSeparator(),
        MenuAction("Full screen", lambda: cur().toggle_fullscreen()),
    ])
    go_menu = Menu("Go", [
        MenuAction("Home (choose model type)", go(None)),
        MenuSeparator(),
        MenuAction("Decision Tree", go("tree")),
        MenuAction("Markov Model", go("markov")),
        MenuAction("Partitioned Survival Model", go("psm")),
        MenuAction("Microsimulation", go("microsim")),
        MenuAction("Network Meta-Analysis", go("nma")),
        MenuSeparator(),
        MenuAction("DECIDE Manual", go("manual")),
    ])
    window_menu = Menu("Window", [MenuAction("Minimize", lambda: window.minimize())])
    languages = [("English", "en"), ("Français", "fr"), ("Tiếng Việt", "vi"), ("Español", "es"),
                 ("中文", "zh"), ("한국어", "ko"), ("日本語", "ja")]
    language_menu = Menu("Language", [
        MenuAction(label, (lambda c=code: send_command("language", model=c))) for label, code in languages
    ])
    setting_menu = Menu("Setting", [
        language_menu,
        MenuSeparator(),
        MenuAction("Model properties...", setting("properties")),
        MenuAction("Model calibration...", setting("calibration")),
        MenuAction("Analysis...", setting("analysis")),
        MenuSeparator(),
        MenuAction("Export code...", setting("export")),
        MenuSeparator(),
        MenuAction("NMA settings...", setting("nma")),
    ])
    bayes_menu = Menu("Bayes revision", [
        MenuAction("Diagnostic test (binary)...", setting("tree_bayes")),
        MenuAction("Several states / results (matrix)...", setting("tree_bayes_matrix")),
    ])
    tree_tools_menu = Menu("Tree", [
        bayes_menu,
        MenuAction("Flip tree...", setting("tree_flip")),
        MenuAction("Check formulas...", setting("tree_check")),
        MenuAction("Reuse subtree (clone)...", setting("tree_clone")),
        MenuAction("Trackers & Group Monte Carlo...", setting("tree_trackers")),
    ])
    psm_tools_menu = Menu("PSM", [
        MenuAction("Extrapolation & hazard...", setting("psm_extrap")),
        MenuAction("Curve library...", setting("psm_library")),
        MenuAction("Plausibility checks & background mortality...", setting("psm_plaus")),
    ])
    # On macOS the View menu is the system's own, which pywebview cannot add items to, so the switch is also here.
    tools_menu = Menu("Tools", [
        MenuAction("Show / hide node numbers", setting("node_numbers")),
        MenuSeparator(),
        tree_tools_menu, psm_tools_menu,
    ])
    help_menu = Menu("Help", [
        MenuAction("About DECIDE", lambda: send_command("about")),
        MenuAction("DECIDE Manual", go("manual")),
        MenuAction("Home (choose model type)", go(None)),
        MenuSeparator(),
        MenuAction("DECIDE on GitHub", lambda: webbrowser.open(GITHUB_URL)),
    ])
    if sys.platform == "darwin":
        # macOS already provides native Edit and View menus (working Cmd+C/V/X/Z);
        # reorder_mac_menu() puts everything in File, Edit, View, Setting, Help order.
        menu = [file_menu, tools_menu, setting_menu, help_menu]
    else:
        menu = [file_menu, edit_menu, view_menu, tools_menu, setting_menu, help_menu]

    def reorder_mac_menu() -> None:
        """Order the menu bar: app menu, File, Edit, View, Go, Window, Help."""
        if sys.platform != "darwin":
            return
        time.sleep(1.5)  # let pywebview finish building its menus
        try:
            from AppKit import NSApp
            from PyObjCTools import AppHelper

            def work() -> None:
                main = NSApp.mainMenu()
                wanted = ["File", "Edit", "View", "Tools", "Setting", "Help"]
                position = 1  # index 0 is the application menu
                for title in wanted:
                    for i in range(main.numberOfItems()):
                        item = main.itemAtIndex_(i)
                        if str(item.title()) == title:
                            main.removeItemAtIndex_(i)
                            main.insertItem_atIndex_(item, min(position, main.numberOfItems()))
                            position += 1
                            break

            AppHelper.callAfter(work)

            def shortcuts() -> None:
                cmd, shift = 1 << 20, 1 << 17
                keys = {"New model": ("n", cmd), "Open...": ("o", cmd), "Save": ("s", cmd),
                        "Save As...": ("s", cmd | shift), "Close tab": ("w", cmd), "Quit": ("q", cmd)}
                main = NSApp.mainMenu()
                for i in range(main.numberOfItems()):
                    top = main.itemAtIndex_(i)
                    if str(top.title()) != "File" or top.submenu() is None:
                        continue
                    for sub in top.submenu().itemArray():
                        spec = keys.get(str(sub.title()))
                        if spec:
                            sub.setKeyEquivalent_(spec[0])
                            sub.setKeyEquivalentModifierMask_(spec[1])

            AppHelper.callAfter(shortcuts)
        except Exception:  # noqa: BLE001 - cosmetic only; keep default order if anything fails
            pass

    try:
        # pywebview blocks file downloads by default, so the Word/Excel report buttons and the canvas camera
        # (PNG) did nothing or navigated the page to the blob. Allow them: the OS shows its own Save dialog.
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True
        except Exception:  # noqa: BLE001 - older pywebview without this setting
            pass
        webview.start(reorder_mac_menu, menu=menu)
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
