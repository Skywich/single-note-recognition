import warnings

import numpy as np
from scipy.io import wavfile


def preprocess(audio_path, frame_ms=25.0, hop_ms=10.0) -> dict:

#---校验---
    if frame_ms <=0 or hop_ms <=0:
        raise ValueError("frame_ms and hop_ms must be positive")
    if hop_ms > frame_ms:
        raise ValueError("hop_ms cannot be greater than frame_ms")
#---读取---
    rate, data = wavfile.read(audio_path)
    rate = int(rate)

#---转换格式，转换成单声道---

    y = data.astype(np.float64)

    if y.ndim == 2:
        y = y.mean(axis=1)
    elif y.ndim != 1:
        raise ValueError(f"not support ndim {y.ndim}")

    if data.dtype == np.uint8:
        y = y - 128.0

    if y.size == 0  :
        raise ValueError(f"empty audio: {audio_path}")
    if not np.all(np.isfinite(y) ):
        raise ValueError(f":audio include NaN/Inf {audio_path}")

#---除最大值幅度归一化---

    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y_raw = y / peak
    else:
        y_raw = y.copy()
        warnings.warn("empty audio, skip it", RuntimeWarning, stacklevel=2)

#---求帧长帧移---

    frame_len = max(1, int(np.floor(frame_ms * rate / 1000.0 + 0.5)))
    hop_len = max(1, int(np.floor(hop_ms * rate / 1000.0 + 0.5)))

#---分帧---

    L = y_raw.size

    if L >= frame_len:
        n_frames = 1 + (L - frame_len) // hop_len
        view = np.lib.stride_tricks.sliding_window_view(y_raw, frame_len)
        frames = np.ascontiguousarray(view[::hop_len][:n_frames])

    else:
        padded = np.zeros(frame_len, dtype=np.float64)
        padded[:L] = y_raw
        frames = padded.reshape(1, frame_len)

#---返回值---

    return {
        "y_raw": y_raw,
        "frames": frames,
        "sr": rate,
        "frame_len": frame_len,
        "hop_len": hop_len,
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python preprocess_m.py <audio_path>")
        raise SystemExit(1)

    audio_path = sys.argv[1]
    ctx = preprocess(audio_path)
    for key, value in ctx.items():
        if isinstance(value, np.ndarray):
            print(f"{key:10s} shape={value.shape} dtype={value.dtype}")
        else:
            print(f"{key:10s} {value}")
    print(f"{'峰值':10s} {float(np.max(np.abs(ctx['y_raw'])))}")