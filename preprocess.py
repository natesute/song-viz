#!/usr/bin/env python3
"""Embed dream.wav with MERT-v1-330M, reduce to 3D via UMAP, export for viewer."""
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import umap
from transformers import AutoModel, Wav2Vec2FeatureExtractor

HERE = Path(__file__).parent
SRC_AUDIO = Path.home() / "Desktop" / "dream.wav"
VIEWER_DIR = HERE / "viewer"
VIEWER_AUDIO = VIEWER_DIR / "dream.wav"
OUTPUT_JSON = VIEWER_DIR / "data.json"
MODEL_ID = "m-a-p/MERT-v1-330M"

# Cache keyed by model so swapping between 95M/330M doesn't re-embed.
_model_slug = MODEL_ID.replace("/", "_")
EMB_CACHE = HERE / f"embeddings_{_model_slug}.npy"
STARTS_CACHE = HERE / f"starts_{_model_slug}.npy"
TARGET_SR = 24000
WINDOW_S = 1.0
HOP_S = 0.1
BATCH = 16

UMAP_N_NEIGHBORS = 6
UMAP_MIN_DIST = 0.95
UMAP_SPREAD = 3.0

# Self-similarity neighbors: for each chunk, find top-K most similar *non-adjacent* chunks.
NN_K = 6
NN_EXCLUDE = 10  # exclude ±N chunks around self in time (=N*HOP_S seconds)


def robust_norm01(a):
    """Map values into [0, 1] using 5th/95th percentile clipping so outliers don't flatten the range."""
    a = np.asarray(a, dtype=np.float64)
    lo = float(np.percentile(a, 5))
    hi = float(np.percentile(a, 95))
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def compute_chunk_features(chunks, sr):
    """Per-chunk RMS (loudness) and spectral centroid (brightness)."""
    chunks = np.asarray(chunks, dtype=np.float64)  # (N, L)
    rms = np.sqrt(np.mean(chunks ** 2, axis=1))
    spec = np.abs(np.fft.rfft(chunks, axis=1))
    freqs = np.fft.rfftfreq(chunks.shape[1], d=1.0 / sr)
    total = spec.sum(axis=1)
    # Avoid divide-by-zero on silent frames.
    centroid = np.where(total > 0, (spec * freqs[None, :]).sum(axis=1) / np.maximum(total, 1e-12), 0.0)
    return rms, centroid


def top_k_neighbors(embeddings, k, exclude):
    """Cosine-similarity top-k per row, ignoring self and ±exclude temporal neighbors."""
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-8)
    sim = emb_n @ emb_n.T  # (n, n)
    for i in range(n):
        lo = max(0, i - exclude)
        hi = min(n, i + exclude + 1)
        sim[i, lo:hi] = -np.inf
    idx = np.argsort(-sim, axis=1)[:, :k]
    score = np.take_along_axis(sim, idx, axis=1)
    score = np.clip(score, 0.0, 1.0)
    return idx.astype(np.int32), score.astype(np.float32)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    # MERT has custom ops that can be flaky on MPS; CPU is safest on Mac.
    return "cpu"


def main():
    device = pick_device()
    print(f"device: {device}")

    if not VIEWER_AUDIO.exists():
        print(f"copying {SRC_AUDIO} -> {VIEWER_AUDIO}")
        shutil.copyfile(SRC_AUDIO, VIEWER_AUDIO)

    print(f"loading audio: {SRC_AUDIO}")
    data_np, sr = sf.read(str(SRC_AUDIO), always_2d=True)  # (samples, channels)
    mono = data_np.mean(axis=1).astype(np.float32)  # downmix to mono
    waveform = torch.from_numpy(mono).unsqueeze(0)  # (1, samples)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    audio = waveform.squeeze().numpy().astype(np.float32)
    duration = len(audio) / TARGET_SR
    print(f"audio: {duration:.1f}s @ {TARGET_SR}Hz, {len(audio)} samples")

    window_n = int(WINDOW_S * TARGET_SR)
    hop_n = int(HOP_S * TARGET_SR)
    starts = list(range(0, len(audio) - window_n + 1, hop_n))

    # Per-chunk audio features (always recomputed — cheap).
    print("computing per-chunk loudness + brightness")
    _chunks = np.stack([audio[s:s + window_n] for s in starts])
    rms, centroid = compute_chunk_features(_chunks, TARGET_SR)
    loudness = robust_norm01(rms)
    brightness = robust_norm01(centroid)

    if EMB_CACHE.exists() and STARTS_CACHE.exists():
        cached_starts = np.load(STARTS_CACHE)
        if len(cached_starts) == len(starts) and np.array_equal(cached_starts, np.array(starts)):
            print(f"reusing cached embeddings: {EMB_CACHE}")
            embeddings = np.load(EMB_CACHE)
        else:
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print(f"loading model: {MODEL_ID} (downloads ~380MB the first time)")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()

        print(f"chunks: {len(starts)} ({WINDOW_S}s window, {HOP_S}s hop)")
        out = []
        with torch.no_grad():
            for i in range(0, len(starts), BATCH):
                batch_starts = starts[i:i + BATCH]
                chunks = [audio[s:s + window_n] for s in batch_starts]
                inputs = processor(chunks, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                out.append(emb)
                done = min(i + BATCH, len(starts))
                print(f"  {done}/{len(starts)}")
        embeddings = np.concatenate(out, axis=0)
        np.save(EMB_CACHE, embeddings)
        np.save(STARTS_CACHE, np.array(starts))
        print(f"cached embeddings -> {EMB_CACHE}")

    print(f"embeddings: {embeddings.shape}")

    print(f"UMAP -> 3D  (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, spread={UMAP_SPREAD})")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        random_state=42,
    )
    pts = reducer.fit_transform(embeddings)

    # normalize into [-1, 1] cube, centered
    mn, mx = pts.min(0), pts.max(0)
    center = (mn + mx) / 2
    scale = float((mx - mn).max()) / 2
    pts = (pts - center) / scale

    print(f"self-similarity neighbors  (k={NN_K}, exclude ±{NN_EXCLUDE} chunks)")
    nn_idx, nn_score = top_k_neighbors(embeddings, NN_K, NN_EXCLUDE)

    data = {
        "points": pts.tolist(),
        "times": [s / TARGET_SR for s in starts],
        "hopS": HOP_S,
        "windowS": WINDOW_S,
        "duration": duration,
        "loudness":  [round(float(v), 4) for v in loudness],
        "brightness": [round(float(v), 4) for v in brightness],
        "neighbors": {
            "idx":   nn_idx.tolist(),
            "score": [[round(float(s), 4) for s in row] for row in nn_score],
        },
    }
    OUTPUT_JSON.write_text(json.dumps(data))
    print(f"wrote {OUTPUT_JSON}  ({len(pts)} points, {NN_K} neighbors each)")


if __name__ == "__main__":
    main()
