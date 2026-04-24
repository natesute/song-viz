# song-viz

3D visualization of a song's MERT latent space. Each ~1-second audio chunk is embedded with [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) (all 13 transformer layers averaged), reduced to 3D via UMAP, and rendered as a triangle cloud in the browser. A per-stem playhead snaps through the chunks in sync with playback — concussion-flashing chunks, firing nearest-neighbor echoes, and spawning particles. Supports whole-song mode, stem-split mode (vocals/drums/bass/other + mix via [Demucs](https://github.com/facebookresearch/demucs)), a cinematic follow camera, and direct-to-MP4 recording.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For stem-split mode, also install Demucs:
```bash
pip install demucs
```

## Preprocess a song

```bash
# whole-song mode
python preprocess.py /path/to/song.mp3

# stem-split mode (vocals/drums/bass/other + mix)
python -m demucs /path/to/song.mp3 -o stems
python preprocess_stems.py "Song Title" "stems/htdemucs/Song Title"
```

Preprocessing runs MERT-v1-330M on CPU (slow — budget ~15 min per song for whole-mix, ~60 min for 4-stem split). Embeddings are cached per `(model, version, song, stem)` so re-runs with different UMAP/neighbor params are instant.

## Serve and view

```bash
cd viewer && python3 -m http.server 8000
# open http://localhost:8000/?song=<slug>
```

Slugs are derived from the input filename (`flume-free.mp3` → `flume-free`). Stems live at `<slug>-stems`. Multiple songs coexist in the same `viewer/` folder; pick one via the `?song=` query param.

## Controls

- **Space / play button** — transport
- **Scrub bar** — seek (visuals freeze while seek buffers, resume cleanly)
- **Drag / scroll** — orbit & zoom
- **F** — toggle follow-camera mode
- **1..4 / 0 / 5** — focus a specific stem (multi-stem mode only)
- **[ / ←** — advance visuals (fire earlier)
- **] / →** — delay visuals (fire later) — hold Shift for 5ms fine-tune
- **R** — record the canvas + song audio to MP4 (Safari) or WebM (other browsers). Downloads automatically when you press R again to stop.

## What's in a frame

- Reuleaux-triangle sprites (curved-edged equilateral triangles), randomly rotated.
- Per-chunk loudness drives activation flash size & intensity.
- Hits trigger white burst → brief random R/G/B concussion flash → long white decay, per the shader's three-phase color profile.
- Top-6 self-similar neighbors (cosine on MERT embeddings, excluding ±10 temporal chunks) light up with exponential rank-weighted falloff.
- Particle emitter at each hit — count scales with loudness, ~1.3s life.
- Post chain: MSAA 4× → Bokeh DOF (focus tracks cloud center) → UnrealBloom → radial chromatic aberration.
- Starfield background, 7000 dim stars.
- Follow camera: Gaussian-smoothed weighted centroid over stem leads, quaternion-slerped orientation, adaptive radius (spread-aware and amplitude-reactive).

## Tunable knobs

### preprocess.py
- `MODEL_ID` — `m-a-p/MERT-v1-330M` (default) vs `m-a-p/MERT-v1-95M` (faster).
- `EMB_VERSION` — bump to invalidate caches when embedding aggregation changes.
- `HOP_S`, `WINDOW_S` — chunking.
- `UMAP_N_NEIGHBORS`, `UMAP_MIN_DIST`, `UMAP_SPREAD` — UMAP shape.
- `NN_K`, `NN_EXCLUDE` — self-similarity echo count + temporal exclusion window.
- `compute_onset_indices` uses librosa's superflux.

### viewer/index.html
Near the top, find:
- `MULTISTEM_SCALE` — how tight the 4 stem clouds pack around origin.
- `STEM_SIZE_MULT`, `STEM_LEAD`, `MIX_LEAD` — per-stem size / lead prominence.
- `VISUAL_LEAD` — default visual-time offset (press `[` / `]` to adjust live).
- `LOUD_FLOOR` — chunks below this loudness skip firing.

In the shader:
- `loudBoost` — size dynamic range across loudness.
- `hitMag` (in JS) — brightness / residue deposit per hit.
- Particle count formula in `fireHit`.
- Bloom strength, Bokeh aperture, chromatic aberration strength.
