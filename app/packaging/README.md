# DECIDE desktop app

Native window (pywebview) around the Streamlit app, built with PyInstaller.

## Build
GitHub -> Actions -> **Release desktop app** -> Run workflow, enter a version (for example `0.1.0`). A draft release with the macOS `.dmg` and the Windows `.zip` appears under Releases; review it and click **Publish release**.

Local (on the target OS), from the `app/` folder: `pip install -r packaging/requirements-build.txt` then `python packaging/build.py` (output in `dist/`). The build bundles the engine from `../src`.

## Run the launcher from source (no packaging)
`pip install pywebview` then `python packaging/launcher.py`

## First launch (the app is not code-signed)
- macOS: drag DECIDE to Applications, right-click -> Open -> Open.
- Windows: unzip, run DECIDE.exe; if SmartScreen appears: More info -> Run anyway. Needs the Microsoft WebView2 runtime (included in Windows 11).

## Data
The app copies itself to `~/DECIDE/app` on first run; models and autosave live in `~/DECIDE/app/models` and `~/DECIDE/app/autosave` and are never overwritten by updates.
