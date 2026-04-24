#!/usr/bin/env python3
"""Build a multi-stem visualization JSON.

Takes 4 stem WAV files (vocals/drums/bass/other from Demucs), embeds each
with MERT, fits a JOINT UMAP on all stem embeddings combined so they share
one 3D space, then writes a single JSON with per-stem trajectories.

Usage:
    python preprocess_stems.py "Flume - Free" /path/to/stems_folder

Where stems_folder contains `vocals.mp3|wav`, `drums.mp3|wav`, etc.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import umap
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# Reuse helpers from preprocess.py
from preprocess import (
    MODEL_ID, TARGET_SR, WINDOW_S, HOP_S, BATCH,
    NN_K, NN_EXCLUDE, MEL_BANDS, MEL_FMIN, MEL_FMAX, EMB_VERSION,
    load_audio, robust_norm01, compute_chunk_features, top_k_neighbors,
    compute_mel_vectors, compute_onset_indices, pick_device, slugify,
)

HERE = Path(__file__).parent
VIEWER_DIR = HERE / "viewer"
_model_slug = MODEL_ID.replace("/", "_") + f"_{EMB_VERSION}"
STEM_NAMES = ("vocals", "drums", "bass", "other")


def find_stem(stems_dir: Path, name: str) -> Path:
    """Find vocals.(wav|mp3) etc. in the stems folder."""
    for ext in (".wav", ".mp3", ".flac"):
        p = stems_dir / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No stem found for '{name}' in {stems_dir}")


def embed_stem(audio: np.ndarray, stem_name: str, song_slug: str,
               processor, model, device):
    """Returns (starts, embeddings). Caches per (song, stem, model)."""
    window_n = int(WINDOW_S * TARGET_SR)
    hop_n   = int(HOP_S * TARGET_SR)
    starts  = list(range(0, len(audio) - window_n + 1, hop_n))

    emb_cache    = HERE / f"embeddings_{_model_slug}_{song_slug}_{stem_name}.npy"
    starts_cache = HERE / f"starts_{_model_slug}_{song_slug}_{stem_name}.npy"

    if emb_cache.exists() and starts_cache.exists():
        cached_starts = np.load(starts_cache)
        if len(cached_starts) == len(starts) and np.array_equal(cached_starts, np.array(starts)):
            print(f"  [{stem_name}] reusing cache")
            return starts, np.load(emb_cache)

    print(f"  [{stem_name}] embedding {len(starts)} chunks")
    out = []
    with torch.no_grad():
        for i in range(0, len(starts), BATCH):
            batch_starts = starts[i:i + BATCH]
            chunks = [audio[s:s + window_n] for s in batch_starts]
            inputs = processor(chunks, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hs = torch.stack(outputs.hidden_states, dim=0)
            emb = hs.mean(dim=2).mean(dim=0).cpu().numpy()
            out.append(emb)
            done = min(i + BATCH, len(starts))
            if done % 160 == 0 or done == len(starts):
                print(f"    {done}/{len(starts)}")
    embeddings = np.concatenate(out, axis=0)
    np.save(emb_cache, embeddings)
    np.save(starts_cache, np.array(starts))
    return starts, embeddings


# Per-stem tint (baseline color for the stem's idle/flash, in sRGB 0..1).
# Kept distinct enough that 4 overlapping clouds read as separate tracks.
STEM_COLORS = {
    "vocals": [0.95, 0.95, 1.00],   # near-white, slight cool tint
    "drums":  [1.00, 0.35, 0.30],   # warm red
    "bass":   [0.35, 0.55, 1.00],   # blue
    "other":  [0.55, 0.95, 0.55],   # green
}


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: preprocess_stems.py <song-title> <stems-dir>")
    song_title = sys.argv[1]
    stems_dir  = Path(sys.argv[2])
    song_slug  = slugify(song_title)
    out_slug   = f"{song_slug}-stems"

    device = pick_device()
    print(f"device: {device}  song: {song_title}  out-slug: {out_slug}")

    # Load MERT once, reuse for all stems.
    print(f"loading model: {MODEL_ID}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()

    # Collect each stem's audio + embed
    stems = {}
    duration = None
    for name in STEM_NAMES:
        path = find_stem(stems_dir, name)
        print(f"stem {name}: {path}")
        audio = load_audio(path, TARGET_SR)
        if duration is None:
            duration = len(audio) / TARGET_SR
        starts, emb = embed_stem(audio, name, song_slug, processor, model, device)
        print(f"  embeddings: {emb.shape}")

        # Per-chunk features (loudness, brightness, mel-bands) from this stem's audio
        window_n = int(WINDOW_S * TARGET_SR)
        chunks = np.stack([audio[s:s + window_n] for s in starts])
        rms, centroid = compute_chunk_features(chunks, TARGET_SR)
        mel = compute_mel_vectors(chunks, TARGET_SR)
        onsets = compute_onset_indices(chunks, TARGET_SR)
        print(f"  [{name}] {len(onsets)} onsets")
        stems[name] = {
            "starts":     starts,
            "embeddings": emb,
            "loudness":   robust_norm01(rms),
            "brightness": robust_norm01(centroid),
            "mel":        mel,
            "onsets":     onsets,
        }

    # Separate UMAP per stem — each stem is fit independently and normalized
    # into its own [-1, 1] cube. No cross-stem acoustic relationships are
    # preserved; each stem uses the full 3D volume to show its own trajectory.
    per_stem_len = len(stems[STEM_NAMES[0]]["starts"])
    for name in STEM_NAMES:
        print(f"UMAP for {name}")
        reducer = umap.UMAP(n_components=3, n_neighbors=6, min_dist=0.95, spread=3.0, random_state=42)
        pts = reducer.fit_transform(stems[name]["embeddings"])
        mn, mx = pts.min(0), pts.max(0)
        center = (mn + mx) / 2
        scale  = float((mx - mn).max()) / 2
        pts = (pts - center) / scale
        stems[name]["points"] = pts.tolist()

    # Per-stem nearest neighbors (within the stem's own embeddings)
    for name in STEM_NAMES:
        print(f"NN for {name}")
        idx, score = top_k_neighbors(stems[name]["embeddings"], NN_K, NN_EXCLUDE)
        stems[name]["neighbors"] = {
            "idx":   idx.tolist(),
            "score": [[round(float(s), 4) for s in row] for row in score],
        }

    # Copy the source mix into the viewer folder for playback (if not already present)
    # Use the same audio file the mono-stem version used so selection is easy.
    mix_candidates = list((HERE / "viewer").glob(f"{song_slug}.*"))
    audio_file_name = mix_candidates[0].name if mix_candidates else f"{song_slug}.mp3"

    out = {
        "slug":      out_slug,
        "title":     f"{song_title} — stems",
        "audioFile": audio_file_name,
        "hopS":      HOP_S,
        "windowS":   WINDOW_S,
        "duration":  duration,
        "melBands": MEL_BANDS,
        "melFmin":  MEL_FMIN,
        "melFmax":  MEL_FMAX,
        "stems": {
            name: {
                "color":      STEM_COLORS[name],
                "points":     stems[name]["points"],
                "loudness":   [round(float(v), 4) for v in stems[name]["loudness"]],
                "brightness": [round(float(v), 4) for v in stems[name]["brightness"]],
                "neighbors":  stems[name]["neighbors"],
                "melVectors": [round(float(v), 4) for v in stems[name]["mel"].flatten().tolist()],
                "onsets":     stems[name]["onsets"],
            } for name in STEM_NAMES
        },
    }
    out_path = VIEWER_DIR / f"{out_slug}.json"
    out_path.write_text(json.dumps(out))
    n = per_stem_len
    print(f"wrote {out_path}  ({n} chunks × 4 stems = {4*n} points)")
    print(f"open: http://localhost:8000/?song={out_slug}")


if __name__ == "__main__":
    main()
