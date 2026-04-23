# song-viz

3D visualization of a song's MERT embedding space. Each ~1-second audio chunk is embedded with [MERT](https://huggingface.co/m-a-p/MERT-v1-330M), reduced to 3D via UMAP, and rendered as a point cloud in the browser. A playhead snaps through the chunks in sync with audio playback, lighting up the current moment and its self-similar echoes — so when a chorus returns, the prior instances of that chorus glow across the cloud.

## What you see

- **Cloud**: 1241 points (for a 2-minute song at 0.1s hop), color by song-time, size by loudness, saturation/lightness by spectral brightness.
- **Playhead**: snaps 10×/sec to the exact embedded chunk at the current audio time.
- **Comet trail**: recent playhead history, fading.
- **Self-similarity echoes**: the 6 nearest chunks in embedding space (excluding temporal neighbors) light up with each playhead step, with exponential rank falloff.
- **Spectral-flux onset pulse**: a brief flare on the playhead at transients.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Put your audio file somewhere — by default `preprocess.py` looks for `~/Desktop/dream.wav`. Edit `SRC_AUDIO` to point elsewhere.
2. Preprocess:
   ```bash
   python preprocess.py
   ```
   First run downloads MERT-v1-330M (~1.3GB) and embeds every chunk. Subsequent runs reuse the cached embeddings, so tweaking UMAP / neighbor settings is near-instant.
3. Serve the viewer:
   ```bash
   cd viewer && python3 -m http.server 8000
   ```
4. Open http://localhost:8000.

## Tunable knobs

- **`MODEL_ID`** — swap between `m-a-p/MERT-v1-95M` (faster) and `m-a-p/MERT-v1-330M` (richer).
- **`HOP_S`** — chunk step size. 0.1s gives a 10Hz playhead; 0.5s is coarser but faster to embed.
- **`UMAP_N_NEIGHBORS`, `UMAP_MIN_DIST`, `UMAP_SPREAD`** — shape the cloud. Lower neighbors + higher min_dist/spread = more separation between clusters.
- **`NN_K`, `NN_EXCLUDE`** — number of self-similarity echoes per chunk and the temporal exclusion window.

Viewer aesthetics (point size, active-boost, color saturation, decay half-life) are all tunable directly in `viewer/index.html`.
