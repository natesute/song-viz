#!/usr/bin/env python3
"""Embed an audio file with MERT, reduce to 3D via UMAP, export for viewer.

Usage:
    python preprocess.py                            # defaults to ~/Desktop/dream.wav
    python preprocess.py path/to/song.mp3          # any WAV / FLAC / MP3 / M4A
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import umap
from transformers import AutoModel, Wav2Vec2FeatureExtractor

HERE = Path(__file__).parent
VIEWER_DIR = HERE / "viewer"
DEFAULT_AUDIO = Path.home() / "Desktop" / "dream.wav"

MODEL_ID = "m-a-p/MERT-v1-330M"
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

SF_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}  # formats soundfile handles natively
# Bump EMB_VERSION when the embedding extraction method changes so old
# caches are not silently reused. v2 = all-layer mean pool.
EMB_VERSION = "v2"
_model_slug = MODEL_ID.replace("/", "_") + f"_{EMB_VERSION}"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "song"


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 at `target_sr`. Uses soundfile for native formats and ffmpeg for the rest."""
    suffix = path.suffix.lower()
    if suffix in SF_EXTS:
        data, sr = sf.read(str(path), always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        if sr != target_sr:
            waveform = torch.from_numpy(mono).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            mono = waveform.squeeze().numpy().astype(np.float32)
        return mono

    # Shell out to ffmpeg for mp3, m4a, aac, etc. — decode straight to float32 mono at target_sr.
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(target_sr),
            "-",
        ],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


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
    centroid = np.where(total > 0, (spec * freqs[None, :]).sum(axis=1) / np.maximum(total, 1e-12), 0.0)
    return rms, centroid


MEL_BANDS = 32
MEL_FMIN = 80.0
MEL_FMAX = 8000.0


def compute_mel_vectors(chunks, sr, n_bands=MEL_BANDS, fmin=MEL_FMIN, fmax=MEL_FMAX):
    """Per-chunk L2-normalized log-spectrogram bands for live similarity matching.

    Returns (N, n_bands) float32 array, each row unit-length so cosine similarity
    against a runtime-computed live-audio vector of the same shape is just a dot product.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    spec  = np.abs(np.fft.rfft(chunks, axis=1))  # (N, L/2 + 1)
    freqs = np.fft.rfftfreq(chunks.shape[1], d=1.0 / sr)
    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    bands = np.zeros((chunks.shape[0], n_bands), dtype=np.float64)
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if mask.any():
            bands[:, b] = spec[:, mask].mean(axis=1)
    bands = np.log1p(bands * 10.0)
    norms = np.linalg.norm(bands, axis=1, keepdims=True)
    return (bands / np.maximum(norms, 1e-8)).astype(np.float32)


def compute_onset_indices(chunks, sr, hop_s=None):
    """librosa superflux onsets mapped onto our chunk grid.

    Concatenates the chunks back into a continuous signal, runs librosa's
    superflux onset-strength + peak-pick, then maps each onset time to its
    nearest chunk index. Much more accurate than our hand-rolled spectral
    flux, especially for percussive transients.
    """
    import librosa  # imported lazily so non-match-mode runs don't pay the cost
    # Reconstruct a continuous signal from non-overlapping chunk starts.
    # For overlapping chunks (hop < window), just load the first chunk's
    # audio plus the hop-overlapping tails.
    if hop_s is None:
        hop_s = HOP_S
    window_n = chunks.shape[1]
    hop_n = int(hop_s * sr)
    n_chunks = len(chunks)
    total = (n_chunks - 1) * hop_n + window_n
    y = np.zeros(total, dtype=np.float32)
    y[:window_n] = chunks[0]
    for i in range(1, n_chunks):
        y[i * hop_n : i * hop_n + window_n] = chunks[i]

    # librosa onset detection — hop matches a short internal window for
    # fine-grained onset strength; we post-process back to our chunk grid.
    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, units="time", backtrack=False,
        hop_length=512,
    )
    onset_chunks = np.round(onset_times / hop_s).astype(int)
    onset_chunks = np.clip(onset_chunks, 0, n_chunks - 1)
    return sorted(set(onset_chunks.tolist()))


def top_k_neighbors(embeddings, k, exclude):
    """Cosine-similarity top-k per row, ignoring self and ±exclude temporal neighbors."""
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-8)
    sim = emb_n @ emb_n.T
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
    return "cpu"  # MERT's custom ops are flaky on MPS.


def main():
    src = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    slug = slugify(src.stem)
    viewer_audio = VIEWER_DIR / f"{slug}{src.suffix.lower()}"
    output_json  = VIEWER_DIR / f"{slug}.json"
    emb_cache    = HERE / f"embeddings_{_model_slug}_{slug}.npy"
    starts_cache = HERE / f"starts_{_model_slug}_{slug}.npy"

    device = pick_device()
    print(f"device: {device}   slug: {slug}")

    if not viewer_audio.exists():
        print(f"copying {src} -> {viewer_audio}")
        shutil.copyfile(src, viewer_audio)

    print(f"loading audio: {src}")
    audio = load_audio(src, TARGET_SR)
    duration = len(audio) / TARGET_SR
    print(f"audio: {duration:.1f}s @ {TARGET_SR}Hz, {len(audio)} samples")

    window_n = int(WINDOW_S * TARGET_SR)
    hop_n = int(HOP_S * TARGET_SR)
    starts = list(range(0, len(audio) - window_n + 1, hop_n))

    print("computing per-chunk loudness + brightness + mel-vectors + onsets")
    _chunks = np.stack([audio[s:s + window_n] for s in starts])
    rms, centroid = compute_chunk_features(_chunks, TARGET_SR)
    loudness = robust_norm01(rms)
    brightness = robust_norm01(centroid)
    mel_vectors = compute_mel_vectors(_chunks, TARGET_SR)
    onset_indices = compute_onset_indices(_chunks, TARGET_SR)
    print(f"  {len(onset_indices)} onsets detected ({len(onset_indices)/duration:.1f}/sec)")

    embeddings = None
    if emb_cache.exists() and starts_cache.exists():
        cached_starts = np.load(starts_cache)
        if len(cached_starts) == len(starts) and np.array_equal(cached_starts, np.array(starts)):
            print(f"reusing cached embeddings: {emb_cache}")
            embeddings = np.load(emb_cache)

    if embeddings is None:
        print(f"loading model: {MODEL_ID}")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()

        print(f"chunks: {len(starts)} ({WINDOW_S}s window, {HOP_S}s hop, all-layer pool)")
        out = []
        with torch.no_grad():
            for i in range(0, len(starts), BATCH):
                batch_starts = starts[i:i + BATCH]
                chunks = [audio[s:s + window_n] for s in batch_starts]
                inputs = processor(chunks, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                # Multi-layer aggregation: stack every transformer hidden state,
                # mean across time, then mean across layers. Captures timbre
                # (early layers), harmony (mid), and structure (late) together.
                hs = torch.stack(outputs.hidden_states, dim=0)  # (L+1, B, T, D)
                emb = hs.mean(dim=2).mean(dim=0).cpu().numpy()  # (B, D)
                out.append(emb)
                done = min(i + BATCH, len(starts))
                print(f"  {done}/{len(starts)}")
        embeddings = np.concatenate(out, axis=0)
        np.save(emb_cache, embeddings)
        np.save(starts_cache, np.array(starts))
        print(f"cached embeddings -> {emb_cache}")

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

    # normalize into [-1, 1] cube
    mn, mx = pts.min(0), pts.max(0)
    center = (mn + mx) / 2
    scale = float((mx - mn).max()) / 2
    pts = (pts - center) / scale

    print(f"self-similarity neighbors  (k={NN_K}, exclude ±{NN_EXCLUDE} chunks)")
    nn_idx, nn_score = top_k_neighbors(embeddings, NN_K, NN_EXCLUDE)

    data = {
        "slug": slug,
        "title": src.stem,
        "audioFile": viewer_audio.name,
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
        "melBands":   MEL_BANDS,
        "melFmin":    MEL_FMIN,
        "melFmax":    MEL_FMAX,
        "melVectors": [round(float(v), 4) for v in mel_vectors.flatten().tolist()],
        # Sparse list of chunk indices where an onset was detected.
        "onsets":     list(onset_indices),
    }
    output_json.write_text(json.dumps(data))
    print(f"wrote {output_json}  ({len(pts)} points, {NN_K} neighbors each)")
    print(f"open: http://localhost:8000/?song={slug}")


if __name__ == "__main__":
    main()
