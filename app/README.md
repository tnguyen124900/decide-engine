# DECIDE app

The Streamlit user interface of DECIDE. It uses the calculation engine in `../src` (package `heor_tree_engine`).

Models available in this release: decision tree, Markov cohort, partitioned survival (PSM), microsimulation and network meta-analysis (experimental).

## Run from source

```bash
cd decide-engine
pip install -e ".[docx]"
pip install -r app/requirements.txt
cd app
python -m streamlit run app.py
```

Python 3.12 is recommended (the pinned versions in `requirements.txt` are the ones the app was tested with).

## Build the desktop app

See `packaging/README.md`. Prebuilt apps for macOS (Apple silicon) and Windows are attached to the releases of this repository.
