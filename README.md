# Spectra Match 
This project converts the original desktop/Gradio workflow into a stateless FastAPI application for Vercel.

## Project layout

```text
api/index.py                 FastAPI entrypoint and matching API
public/index.html            Browser UI
preprocessing_utils.py       Pure preprocessing/math utilities
data/reference_data.pkl      Bundled read-only reference library
requirements.txt             Python dependencies installed by Vercel
vercel.json                  Function duration, bundled data, and rewrites
.vercelignore                Excludes notebooks and desktop-only scripts
```

## Local test

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload
```

Open `http://127.0.0.1:8000`.

## Deploy

1. Put this folder in a GitHub repository.
2. Import the repository in Vercel.
3. Keep the Framework Preset as **Other** or let Vercel detect Python/FastAPI.
4. Deploy. No environment variable is required for the bundled reference library.

## Why the original files were not Vercel-compatible

- `gradio_app.py` starts a persistent Gradio server. Vercel expects an HTTP application exported as `app` and executes it as a function.
- `tkinter` opens a folder picker on the server machine, not in the user's browser. A cloud function cannot choose a folder on the user's computer.
- Local output folders are not durable. Serverless filesystems are ephemeral, so this version sends JSON to the browser and lets the browser generate CSV/JSON downloads.
- `input()` and `webbrowser.open()` require an interactive desktop session and cannot be used inside a web request.
- Relative `joblib.load("reference_data.pkl")` is fragile. The new version resolves an absolute path from the source file and explicitly bundles the pickle with `includeFiles`.
- Nested Python loops for similarity were replaced with matrix multiplication to reduce execution time.
- The Vercel request/response limit is 4.5 MB, so this app accepts files up to 4 MB and returns compressed visual summaries rather than all raw/intermediate matrices.

## Important operational limits

The 24 MB reference pickle is small enough to include as application data, but scientific Python dependencies make the function bundle much larger. Vercel currently allows a larger uncompressed limit for Python functions, but deployment size and cold-start time should still be monitored.

For uploads larger than 4 MB or for full ZIP archives containing all raw, corrected, and interpolated spectra, use direct browser uploads to Vercel Blob and store generated results in Blob, or host the analysis backend on a container platform and keep only the frontend on Vercel.
