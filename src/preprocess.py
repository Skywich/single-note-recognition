# -*- coding: utf-8 -*-
"""
preprocess.py —— 音频前期处理（模块 M1）

这个文件只做三件事：
    1. 读取 wav 文件；
    2. 把多声道混成单声道，并做幅度归一化（除以最大绝对值）；
    3. 按帧长 / 帧移分帧，返回下游要用的全部上下文。

对外只有一个函数：
    preprocess(audio_path, frame_ms=25.0, hop_ms=10.0) -> dict

返回的字典严格包含 5 个键：
    y_raw      : 一维 float64 数组，归一化后的整段信号（切音用这个）
    frames     : 二维 float64 数组，形状 (帧数, 帧长)
    sr         : 采样率（int）
    frame_len  : 帧长对应的采样点数（int）
    hop_len    : 帧移对应的采样点数（int）

依据：
    论文《基于频域匹配滤波的音频信号分析与识别》第 2 节；
    《分工初步》模块 1；
    《音频前期处理脚本计划》v1.0。
"""

import warnings

import numpy as np
from scipy.io import wavfile


def preprocess(audio_path: str, frame_ms: float = 25.0, hop_ms: float = 10.0) -> dict:
    """读取 wav，归一化，分帧，返回下游需要的全部上下文。

    参数
    ----
    audio_path : str
        wav 文件路径。
    frame_ms : float
        帧长（毫秒），默认 25。
    hop_ms : float
        帧移（毫秒），默认 10。

    返回
    ----
    dict
        严格包含 y_raw、frames、sr、frame_len、hop_len 五个键。
    """

    # ---------- Step 0：参数校验 ----------
    # 帧长、帧移必须是正数；帧移不能大于帧长，否则相邻帧之间会留空隙。
    if frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("frame_ms 和 hop_ms 必须为正数")
    if hop_ms > frame_ms:
        raise ValueError("hop_ms 不能大于 frame_ms（帧间会留空隙）")

    # ---------- Step 1：读取 wav ----------
    # wavfile.read 返回两个值：采样率 sr 和采样数据 data。
    sr, data = wavfile.read(audio_path)
    sr = int(sr)

    # ---------- Step 2：转 float64 + 单声道化 ----------
    # 先转 float64 再取绝对值：int16 的 -32768 在整数域取绝对值会溢出。
    y = data.astype(np.float64)

    if y.ndim == 2:
        # 形状 (采样点数, 声道数)。axis=1 表示"把各声道平均掉"，每个采样点只留一个值。
        y = y.mean(axis=1)
    elif y.ndim != 1:
        raise ValueError(f"不支持的音频维度：{y.ndim}")

    if data.dtype == np.uint8:
        # 8-bit PCM 是无符号的（0~255，中点 128），先减掉中点偏置。
        y = y - 128.0

    if y.size == 0:
        raise ValueError(f"音频为空：{audio_path}")
    if not np.all(np.isfinite(y)):
        raise ValueError("音频含 NaN/Inf")

    # ---------- Step 3：幅度归一化（除以最大绝对值） ----------
    # 整个文件只算一个峰值，全段共用，不做逐帧归一化。
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y_raw = y / peak
    else:
        # 全零 / 全静音：不能做 0/0，原样返回全零，让模块 2 去判断"没有音符"。
        y_raw = y.copy()
        warnings.warn("音频全为零，跳过归一化", RuntimeWarning, stacklevel=2)

    # ---------- Step 4：帧长 / 帧移换算 ----------
    # floor(x + 0.5) 等价于正数的四舍五入：
    # 44.1kHz × 25ms = 1102.5，四舍五入后是 1103（和 MATLAB round 一致）。
    frame_len = max(1, int(np.floor(frame_ms * sr / 1000.0 + 0.5)))
    hop_len = max(1, int(np.floor(hop_ms * sr / 1000.0 + 0.5)))

    # ---------- Step 5：分帧 ----------
    L = y_raw.size

    if L >= frame_len:
        # 正常情况：只保留完整帧，不足一帧的尾部丢弃（与 MATLAB enframe 一致）。
        n_frames = 1 + (L - frame_len) // hop_len

        # sliding_window_view 把一维数组变成"很多个连续小窗口"的二维视图，
        # 每个窗口长度是 frame_len；再按 hop_len 每隔几帧取一个。
        view = np.lib.stride_tricks.sliding_window_view(y_raw, frame_len)

        # ascontiguousarray：把视图复制成一块连续、可写的正常数组。
        frames = np.ascontiguousarray(view[::hop_len][:n_frames])
    else:
        # 超短音频：只补 frames，不补 y_raw，保证 y_raw 长度不变。
        padded = np.zeros(frame_len, dtype=np.float64)
        padded[:L] = y_raw
        frames = padded.reshape(1, frame_len)

    # ---------- Step 6：组装返回值（严格 5 个键） ----------
    return {
        "y_raw": y_raw,
        "frames": frames,
        "sr": sr,
        "frame_len": frame_len,
        "hop_len": hop_len,
    }


# 这一块只有"直接运行本文件"时才会执行，被 import 时不会执行。
# 用法：py src\preprocess.py data\demo_tone.wav
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法：py src\\preprocess.py <wav文件路径>")
        raise SystemExit(1)

    ctx = preprocess(sys.argv[1])
    for key, value in ctx.items():
        if isinstance(value, np.ndarray):
            print(f"{key:10s} shape={value.shape} dtype={value.dtype}")
        else:
            print(f"{key:10s} {value}")
    print(f"{'峰值':10s} {float(np.max(np.abs(ctx['y_raw'])))}")
