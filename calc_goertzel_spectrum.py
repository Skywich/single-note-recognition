# -*- coding: utf-8 -*-
"""M3：Goertzel 频谱 + L2 能量归一化（论文对齐版，26 音）。

输入一个单音的一维信号，输出 B3–C6 共 26 个标准音基频上的归一化幅度谱。
模板矩阵 T 的形状为 (26, 26)，第 i 行对应 NOTE_NAMES[i]。
接口版本：v2.0（2026-11-09，按论文表 1 与 M=26 冻结）。
"""
import numpy as np

#---频率网格---
NOTE_NAMES = [
    "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4",
    "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5",
    "G5", "G#5", "A5", "A#5", "B5", "C6",
]
MIDI_B3 = 59
MIDI_C6 = 84
A4_HZ = 440.0


def midi_to_freq(midi, a4=A4_HZ):
    """MIDI 音高 → 频率（Hz），A4 = 440 Hz 对应 MIDI 69。"""
    return a4 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def note_frequencies():
    """B3–C6 共 26 个半音的基频，单位 Hz。"""
    return midi_to_freq(np.arange(MIDI_B3, MIDI_C6 + 1))


TARGET_FREQS = note_frequencies()


#---单频点 Goertzel---
def goertzel_power(x, sr, freq):
    """返回 |X(e^{jω0})|²，x 应已加窗；ω0 = 2π·freq/sr。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("x is empty")

    w0 = 2.0 * np.pi * float(freq) / float(sr)
    coeff = 2.0 * np.cos(w0)

    s1 = 0.0
    s2 = 0.0
    for v in x:
        s0 = v + coeff * s1 - s2
        s2 = s1
        s1 = s0

    p = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return float(max(p, 0.0))


#---谱向量---
def calc_goertzel_spectrum(note_signal, sr, target_freqs):
    """单音（一维）→ 26 维 L2 归一化幅度谱（一维，范数 1）。"""
    #---校验---
    x = np.asarray(note_signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"note_signal must be 1-D, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("note_signal is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("note_signal contains NaN/Inf")
    if float(sr) <= 0:
        raise ValueError("sr must be positive")

    freqs = np.asarray(target_freqs, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("target_freqs must be a non-empty 1-D array")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("target_freqs contains NaN/Inf")
    if np.any(freqs <= 0) or np.any(freqs >= float(sr) / 2.0):
        raise ValueError("target_freqs must satisfy 0 < f < sr/2")

    #---加窗---
    xw = x * np.hamming(x.size)

    #---逐频点 Goertzel---
    power = np.array([goertzel_power(xw, float(sr), f) for f in freqs])

    #---L2 归一化---
    mag = np.sqrt(power)
    norm = float(np.linalg.norm(mag))
    if norm == 0.0:
        return np.zeros_like(mag)
    return mag / norm


#---模板矩阵---
def build_templates(note_signals, sr, target_freqs):
    """26 个标准音 → (26, 26) 模板矩阵，第 i 行对应 NOTE_NAMES[i]。"""
    rows = [calc_goertzel_spectrum(s, sr, target_freqs) for s in note_signals]
    return np.vstack(rows)


#---自检---
if __name__ == "__main__":
    fs = 44100
    t = np.arange(int(0.5 * fs)) / fs
    x = 0.8 * np.sin(2 * np.pi * 261.63 * t) + 0.4 * np.sin(2 * np.pi * 523.25 * t)
    S = calc_goertzel_spectrum(x, fs, TARGET_FREQS)
    top = np.argsort(S)[::-1][:3]
    print(f"shape={S.shape}  norm={np.linalg.norm(S):.6f}")
    for i in top:
        print(f"  {NOTE_NAMES[i]:4s} {S[i]:.4f}")