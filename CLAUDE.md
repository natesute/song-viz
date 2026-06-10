# song-viz

Personal project (GitHub: `natesute/song-viz`). 3D visualization of a song's MERT latent space:
Python preprocessing pipeline → static browser viewer. **README.md has the full setup and
pipeline — read it before working here.**

## Workflow
- Python: venv + `requirements.txt`; `preprocess.py` (whole song), `preprocess_stems.py`
  (Demucs-split stems)
- Viewer: static files in `viewer/`, no build step
- All `*.npy` embeddings, `stems/`, audio files, and `viewer/*.json` are generated artifacts and
  gitignored — regenerate them, never commit them.

## Gotchas
- MERT-v1-330M downloads from Hugging Face on first run; preprocessing is slow and
  memory-hungry. This is an 8 GB machine — close idle Claude sessions before running it.
