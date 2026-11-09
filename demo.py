# -*- coding: utf-8 -*-
r"""基于频域匹配滤波的钢琴单音 / 乐句识别 —— 可完整运行的 Demo。

复现论文：
    赵凌览, 李晓峰. 基于频域匹配滤波的音频信号分析与识别[J].
    实验科学与技术, 2018, 16(5): 34-38.

处理链（严格对应《信号与系统课程项目计划 v1.0.6》第 4、5 节）：
    音频 ──► preprocess ──► segment_notes ──► calc_goertzel_spectrum
         ──► match_filter_decision ──► 音名 / MIDI / 频率

四个板块的公开接口（下游代码只依赖这些签名）：
    preprocess(audio_path, frame_ms=25, hop_ms=10) -> dict
        {"y_raw", "frames", "sr", "frame_len", "hop_len"}
    segment_notes(data_dict) -> list[np.ndarray]
    calc_goertzel_spectrum(note_signal, sr, target_freqs) -> np.ndarray
    match_filter_decision(S_i, templates) -> (best_note_idx, max_corr_value)

运行（Windows，项目内已带 Windows 虚拟环境 .venv）：
    .venv\Scripts\python.exe demo.py
    .venv\Scripts\python.exe demo.py --save-audio demo_audio
    .venv\Scripts\python.exe demo.py --wav my_phrase.wav
    .venv\Scripts\python.exe demo.py --wav my.wav --templates data\templates

也可以先激活虚拟环境：
    .venv\Scripts\activate.bat
    python demo.py

依赖：numpy。WAV 读取使用 Python 标准库解析，支持 PCM 8/16/24/32 bit 与 IEEE float。

音符范围：B3–C6，共 26 个半音（论文表 1 的 26 个音，A4 = 440 Hz，十二平均律）。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - 仅负责给出友好提示
    raise SystemExit(
        "demo.py 需要 numpy。\n"
        "请先安装：python -m pip install numpy\n"
        "本仓库内也可直接使用：.venv\\Scripts\\python.exe demo.py"
    ) from exc


# =============================================================================
# 0. 标准音网格与全局参数
# =============================================================================

NOTE_NAMES_12 = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# 论文表 1：B3 = MIDI 59，C6 = MIDI 84，闭区间共 26 个音。
RANGE_START_MIDI = 59
RANGE_END_MIDI = 84
A4_HZ = 440.0


def midi_to_name(midi: int) -> str:
    """MIDI 编号 -> 科学音高记号，例如 60 -> C4。"""
    return f"{NOTE_NAMES_12[midi % 12]}{midi // 12 - 1}"


def midi_to_freq(midi: int, a4: float = A4_HZ) -> float:
    """十二平均律：f = A4 · 2^((midi-69)/12)。"""
    return float(a4 * 2.0 ** ((midi - 69) / 12.0))


def build_note_grid(
    start_midi: int = RANGE_START_MIDI,
    end_midi: int = RANGE_END_MIDI,
    a4: float = A4_HZ,
) -> tuple[list[str], np.ndarray, list[int]]:
    """返回 (音名列表, 基频数组, MIDI 列表)，长度 M = 26。"""
    midis = list(range(start_midi, end_midi + 1))
    names = [midi_to_name(m) for m in midis]
    freqs = np.array([midi_to_freq(m, a4) for m in midis], dtype=np.float64)
    return names, freqs, midis


NOTE_NAMES, TARGET_FREQS, NOTE_MIDIS = build_note_grid()
NOTE_INDEX = {name: i for i, name in enumerate(NOTE_NAMES)}

#: 模板数 M 与频谱向量长度 N（本项目中恰好都是 26）
M = len(NOTE_NAMES)
N = len(TARGET_FREQS)
assert M == N == 26, "B3–C6 应为 26 个音"


@dataclass(frozen=True)
class Params:
    """计划 4.2 节建议参数的集中管理。"""

    frame_ms: float = 25.0
    hop_ms: float = 10.0
    min_note_ms: float = 50.0
    merge_gap_ms: float = 50.0
    pre_margin_ms: float = 10.0
    post_margin_ms: float = 20.0
    min_score: float = 0.35
    a4: float = A4_HZ


DEFAULT_PARAMS = Params()


@dataclass(frozen=True)
class Segment:
    """端点检测得到的一个音符区间（采样点，左闭右开）。"""

    start: int
    end: int
    sr: int

    @property
    def start_time(self) -> float:
        return self.start / float(self.sr)

    @property
    def end_time(self) -> float:
        return self.end / float(self.sr)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0 / self.sr


@dataclass
class RecognitionResult:
    """一个单音的识别结果。idx/name 为 None 表示低于置信度阈值。"""

    idx: int | None
    name: str | None
    midi: int | None
    freq_hz: float | None
    score: float
    scores: np.ndarray

    @property
    def confident(self) -> bool:
        return self.idx is not None


# =============================================================================
# 1. 音频读写（WAV，标准库）
# =============================================================================


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """读取 WAV，返回 (单声道 float64 波形, 采样率)。

    支持 PCM 8/16/24/32 bit 和 IEEE float 32/64 bit；多声道取平均。
    不支持 MP3/AAC；遇到压缩 WAV 会给出明确错误。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到音频文件：{p}")
    if p.suffix.lower() in {".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
        raise ValueError(f"只支持 WAV，请先把 {p.suffix} 转成 44.1 kHz / 16 bit / 单声道 WAV")

    raw = p.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{p} 不是合法的 RIFF/WAVE 文件")

    fmt: bytes | None = None
    data: bytes | None = None
    pos = 12
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        body_start = pos + 8
        body_end = min(body_start + chunk_size, len(raw))
        body = raw[body_start:body_end]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            data = body
        # chunk 按偶数字节对齐
        pos = body_start + chunk_size + (chunk_size & 1)

    if fmt is None or len(fmt) < 16:
        raise ValueError(f"{p} 缺少 fmt chunk")
    if data is None:
        raise ValueError(f"{p} 缺少 data chunk")

    audio_format = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    sr = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")

    # WAVE_FORMAT_EXTENSIBLE：真正的格式在 SubFormat GUID 的前两字节。
    if audio_format == 0xFFFE and len(fmt) >= 40:
        audio_format = int.from_bytes(fmt[24:26], "little")

    if channels < 1:
        raise ValueError(f"{p} 声道数非法：{channels}")
    if sr <= 0:
        raise ValueError(f"{p} 采样率非法：{sr}")
    if bits <= 0:
        raise ValueError(f"{p} 位深非法：{bits}")

    if audio_format == 1:  # PCM
        if bits == 8:
            x = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif bits == 16:
            x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
        elif bits == 24:
            b = np.frombuffer(data, dtype=np.uint8)
            b = b[: (b.size // 3) * 3].reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float64) / 8388608.0
        elif bits == 32:
            x = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"{p} 不支持 {bits} bit PCM")
    elif audio_format == 3:  # IEEE float
        if bits == 32:
            x = np.frombuffer(data, dtype="<f4").astype(np.float64)
        elif bits == 64:
            x = np.frombuffer(data, dtype="<f8").astype(np.float64)
        else:
            raise ValueError(f"{p} 不支持 {bits} bit float")
    else:
        raise ValueError(
            f"{p} 的 WAV 编码格式为 {audio_format}（0x{audio_format:04X}），"
            "仅支持 PCM(1) 或 IEEE float(3)"
        )

    usable = (x.size // channels) * channels
    if usable == 0:
        raise ValueError(f"{p} 没有采样点")
    x = x[:usable].reshape(-1, channels)
    if channels > 1:
        x = x.mean(axis=1)
    else:
        x = x[:, 0]
    return np.ascontiguousarray(x, dtype=np.float64), sr


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    """把 [-1, 1] 浮点波形写成 16 bit / 单声道 WAV。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.clip(y, -1.0, 1.0)
    pcm = np.round(y * 32767.0).astype("<i2")
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


# =============================================================================
# 2. 模块 1：预处理（计划中的 preprocess.py）
# =============================================================================


def preprocess(
    audio_path: str,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> dict:
    """读取音频 → 单声道 → 幅度归一化 → 分帧。

    参数
    ----
    audio_path : str
        WAV 文件路径。
    frame_ms : float
        帧长（毫秒），默认 25。
    hop_ms : float
        帧移（毫秒），默认 10。

    返回
    ----
    dict
        y_raw     : 1D np.ndarray，按 max(|x|) 归一化后的完整波形（切音符用这个）
        frames    : 2D np.ndarray，形状 (帧数, frame_len)
        sr        : int，采样率
        frame_len : int，帧长（采样点）
        hop_len   : int，帧移（采样点）
    """
    if frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("frame_ms 和 hop_ms 必须为正数")

    y, sr = read_wav(audio_path)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    # 严格按计划：除以最大绝对值；全静音则保持全零，避免 0/0。
    y_raw = y / peak if peak > 0.0 else np.zeros_like(y)

    frame_len = int(round(frame_ms * sr / 1000.0))
    hop_len = int(round(hop_ms * sr / 1000.0))
    frame_len = max(frame_len, 1)
    hop_len = max(hop_len, 1)

    if y_raw.size == 0:
        raise ValueError(f"{audio_path} 没有采样点")

    # 尾帧补零，保证最后不足一帧的音频也能进入端点检测。
    n_frames = 1 + max(0, int(math.ceil((y_raw.size - frame_len) / hop_len)))
    need = (n_frames - 1) * hop_len + frame_len
    if need > y_raw.size:
        y_pad = np.pad(y_raw, (0, need - y_raw.size))
    else:
        y_pad = y_raw

    # 重叠分帧；返回的是只读视图，下游只读即可，长音频也不会复制内存。
    frames = np.lib.stride_tricks.sliding_window_view(y_pad, frame_len)[::hop_len]
    return {
        "y_raw": y_raw,
        "frames": frames,
        "sr": int(sr),
        "frame_len": int(frame_len),
        "hop_len": int(hop_len),
    }


# =============================================================================
# 3. 模块 2：时域端点检测与单音分割（计划中的 segment_notes.py）
# =============================================================================


def _frame_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐帧短时能量 STE 与短时过零率 ZCR。"""
    if frames.ndim != 2 or frames.shape[1] == 0:
        raise ValueError("frames 必须是形状 (帧数, 帧长) 的二维数组")
    x = np.asarray(frames, dtype=np.float64)
    energy = np.mean(x * x, axis=1)
    if x.shape[1] < 2:
        zcr = np.zeros(x.shape[0], dtype=np.float64)
    else:
        # 相邻采样点异号即一次过零
        zcr = np.mean((x[:, 1:] * x[:, :-1]) < 0.0, axis=1)
    return energy, zcr


def _detect_segments(data_dict: dict, params: Params = DEFAULT_PARAMS) -> list[Segment]:
    """端点检测内部实现，额外返回时间信息；公开接口仍是 segment_notes。"""
    y_raw = np.asarray(data_dict["y_raw"], dtype=np.float64).reshape(-1)
    frames = np.asarray(data_dict["frames"])
    sr = int(data_dict["sr"])
    frame_len = int(data_dict["frame_len"])
    hop_len = int(data_dict["hop_len"])

    if y_raw.size == 0 or frames.size == 0:
        return []

    energy, zcr = _frame_features(frames)
    max_energy = float(np.max(energy))
    if max_energy <= 1e-20:
        return []  # 全静音

    # 噪声能量：取能量最低的 10% 帧。
    # 若整段都在发声（没有静音），10% 分位数可能接近峰值；此时把高阈值封顶在
    # 0.5*max、低阈值再压到高阈值的 0.8 倍，保证仍能起振且低阈值低于高阈值。
    noise_energy = float(np.percentile(energy, 10.0))
    high_th = min(max(noise_energy * 4.0, max_energy * 0.01, 1e-20), max_energy * 0.5)
    low_th = min(max(noise_energy * 2.0, max_energy * 0.005, 1e-20), high_th * 0.8)

    # 高过零率通常是宽带噪声/摩擦声；钢琴乐音即使含高次谐波也远低于此门限。
    zcr_gate = 0.45

    raw_intervals: list[tuple[int, int]] = []
    i = 0
    n_frame = len(energy)
    while i < n_frame:
        if energy[i] >= high_th and zcr[i] < zcr_gate:
            j = i + 1
            # 低阈值滞后延伸，避免音符衰减段被截断
            while j < n_frame and energy[j] >= low_th:
                j += 1
            raw_intervals.append((i, j))  # [start_frame, end_frame)
            i = j
        else:
            i += 1

    if not raw_intervals:
        return []

    pre = int(round(params.pre_margin_ms * sr / 1000.0))
    post = int(round(params.post_margin_ms * sr / 1000.0))
    min_len = int(round(params.min_note_ms * sr / 1000.0))
    merge_gap = int(round(params.merge_gap_ms * sr / 1000.0))

    # 帧区间 -> 采样点区间，并为起止各留一点余量
    intervals: list[list[int]] = []
    for start_f, end_f in raw_intervals:
        start = max(0, start_f * hop_len - pre)
        # end_f 是第一个低于低阈值的帧，最后一个有效帧是 end_f-1
        end = min(y_raw.size, (end_f - 1) * hop_len + frame_len + post)
        if end > start:
            intervals.append([start, end])

    # 合并间隔小于 50 ms 的相邻片段（同音连奏 / 短停顿）
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # 过滤短于 50 ms 的毛刺
    return [
        Segment(start=int(start), end=int(end), sr=sr)
        for start, end in merged
        if end - start >= min_len
    ]


def segment_notes(data_dict: dict) -> list[np.ndarray]:
    """短时能量 + 过零率端点检测，从 y_raw 中切出所有单音。

    参数
    ----
    data_dict : dict
        模块 1 的输出字典。

    返回
    ----
    list[np.ndarray]
        每个元素是一个单音的一维 NumPy 数组；没有检测到音符时返回 []。
    """
    y_raw = np.asarray(data_dict["y_raw"], dtype=np.float64)
    return [y_raw[seg.start : seg.end].copy() for seg in _detect_segments(data_dict)]


# =============================================================================
# 4. 模块 3：Goertzel 离散谱线 + L2 能量归一化
# =============================================================================


def calc_goertzel_spectrum(
    note_signal: np.ndarray,
    sr: int,
    target_freqs: np.ndarray,
) -> np.ndarray:
    """在 target_freqs 各频点上做 Goertzel，返回 L2 归一化后的 |X(ω)|。

    递推式（计划 1.4 节）：
        s[n] = x[n] + 2cos(ω0)s[n-1] - s[n-2],  s[-1]=s[-2]=0
        |X(ω0)|² = s[N-1]² + s[N-2]² - 2cos(ω0)s[N-1]s[N-2]

    参数
    ----
    note_signal : 1D np.ndarray
        单个音符的时域波形。
    sr : int
        采样率。
    target_freqs : 1D np.ndarray
        26 个标准音基频。

    返回
    ----
    np.ndarray
        长度 26 的归一化离散频谱序列；满足 ||S_i||_2 = 1（全零输入除外）。
    """
    x = np.asarray(note_signal, dtype=np.float64).reshape(-1)
    freqs = np.asarray(target_freqs, dtype=np.float64).reshape(-1)
    sr = int(sr)

    if sr <= 0:
        raise ValueError("sr 必须为正整数")
    if x.size == 0:
        raise ValueError("note_signal 不能为空")
    if freqs.size == 0:
        return np.zeros(0, dtype=np.float64)
    if np.any(freqs <= 0.0) or np.any(freqs >= sr / 2.0):
        raise ValueError(f"target_freqs 必须位于 (0, sr/2) = (0, {sr / 2}) 内")

    # Hamming 窗减小截断泄漏；L2 归一化会抵消窗函数带来的整体幅度变化。
    window = np.hamming(x.size)
    xw = x * window

    omega = 2.0 * math.pi * freqs / sr
    coeff = 2.0 * np.cos(omega)

    # 一次递推同时算 26 个频点：循环长度 N，状态向量长度 M。
    s_prev = np.zeros(freqs.size, dtype=np.float64)
    s_prev2 = np.zeros(freqs.size, dtype=np.float64)
    for sample in xw:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    power = s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
    mag = np.sqrt(np.maximum(power, 0.0))

    norm = float(np.linalg.norm(mag))
    if norm <= 1e-12:
        return np.zeros(freqs.size, dtype=np.float64)
    return mag / norm


# =============================================================================
# 5. 模块 4：匹配滤波（相关）与最大值判决
# =============================================================================


def match_filter_scores(S_i: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """计算输入谱向量与 M 个模板的相关值（匹配滤波器在 n=N-1 处采样）。

    公开接口 match_filter_decision 只返回 (idx, max)，本函数是同一板块的
    辅助函数，方便 GUI / 报告显示 top-K。
    """
    s = np.asarray(S_i, dtype=np.float64).reshape(-1)
    t = np.asarray(templates, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError("templates 必须是形状 (M, N) 的二维数组")
    if s.size != t.shape[1]:
        raise ValueError(f"维度不匹配：S_i 长度 {s.size}，templates 列数 {t.shape[1]}")

    # 理论上模块 3 已归一化；这里再做一次防御性归一化，
    # 保证 r_ij <= 1 = r_ii 的 Cauchy–Schwarz 前提成立。
    norm = float(np.linalg.norm(s))
    if norm > 1e-12:
        s = s / norm

    row_norm = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.divide(t, row_norm, out=np.zeros_like(t), where=row_norm > 1e-12)

    return t @ s


def match_filter_decision(
    S_i: np.ndarray,
    templates: np.ndarray,
) -> tuple[int, float]:
    """把输入谱向量并行送入 M 路匹配滤波器，取最大输出。

    参数
    ----
    S_i : 1D np.ndarray
        模块 3 输出的归一化频谱序列，长度 N=26。
    templates : 2D np.ndarray
        标准音模板矩阵，形状 (M, N) = (26, 26)，每行一个归一化模板。

    返回
    ----
    (best_note_idx, max_corr_value) : (int, float)
        best_note_idx 为 0–25；max_corr_value 为最大相关值（置信度）。
    """
    scores = match_filter_scores(S_i, templates)
    if scores.size == 0:
        raise ValueError("templates 至少需要一行")
    best = int(np.argmax(scores))
    return best, float(scores[best])


def recognize_single(
    note_signal: np.ndarray,
    sr: int,
    templates: np.ndarray,
    target_freqs: np.ndarray = TARGET_FREQS,
    min_score: float = DEFAULT_PARAMS.min_score,
) -> RecognitionResult:
    """单音识别：归一化 → Goertzel → 相关 → 阈值判决。"""
    x = np.asarray(note_signal, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.0:
        x = x / peak  # 计划伪代码：输入也要归一化

    spectrum = calc_goertzel_spectrum(x, sr, target_freqs)
    scores = match_filter_scores(spectrum, templates)
    idx = int(np.argmax(scores))
    best = float(scores[idx])
    if best < min_score:
        return RecognitionResult(
            idx=None, name=None, midi=None, freq_hz=None, score=best, scores=scores
        )
    midi = NOTE_MIDIS[idx]
    return RecognitionResult(
        idx=idx,
        name=NOTE_NAMES[idx],
        midi=midi,
        freq_hz=float(TARGET_FREQS[idx]),
        score=best,
        scores=scores,
    )


# =============================================================================
# 6. 模板构建（标准音 WAV → 预处理 → 端点检测 → Goertzel → 模板矩阵）
# =============================================================================


def _template_candidates(directory: Path, name: str) -> Iterable[Path]:
    """生成候选文件名：支持 C#4.wav、C4#.wav 以及大小写差异。"""
    stems = [name]
    if "#" in name:  # C#4 -> C4#
        stems.append(name.replace("#", "") + "#")
    for stem in stems:
        yield directory / f"{stem}.wav"
        yield directory / f"{stem}.WAV"


def build_template_matrix(
    template_dir: str | Path,
    params: Params = DEFAULT_PARAMS,
    target_freqs: np.ndarray = TARGET_FREQS,
) -> np.ndarray:
    """读取 26 个标准音 WAV，构建 (M, N) 归一化模板矩阵。

    每个标准音单独一个文件，文件名为 B3.wav、C4.wav、C#4.wav … C6.wav。
    每个文件走一遍 preprocess → segment_notes（取最长片段）→
    calc_goertzel_spectrum，得到一行模板。
    """
    directory = Path(template_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"模板目录不存在：{directory}")

    rows: list[np.ndarray] = []
    missing: list[str] = []
    for name in NOTE_NAMES:
        path = next((p for p in _template_candidates(directory, name) if p.is_file()), None)
        if path is None:
            missing.append(f"{name}.wav")
            continue

        data = preprocess(str(path), params.frame_ms, params.hop_ms)
        segments = segment_notes(data)
        if not segments:
            raise ValueError(f"模板 {path.name} 中没有检测到音符，请检查录音")
        note = max(segments, key=len)  # 单个标准音文件；若有毛刺取最长段
        rows.append(calc_goertzel_spectrum(note, data["sr"], target_freqs))

    if missing:
        raise FileNotFoundError(
            "模板目录缺少以下文件：" + "、".join(missing) + f"\n目录：{directory}"
        )

    templates = np.vstack(rows)
    if templates.shape != (M, N):
        raise ValueError(f"模板矩阵形状应为 {(M, N)}，实际为 {templates.shape}")
    if np.any(np.linalg.norm(templates, axis=1) <= 1e-12):
        raise ValueError("存在全零模板，请检查对应标准音录音")
    return templates


# =============================================================================
# 7. 内置合成钢琴音（无需外部数据即可跑通）
# =============================================================================


def synth_note(
    freq: float,
    duration: float,
    sr: int,
    *,
    amps: Sequence[float],
    inharmonicity: float = 0.0,
    detune_cents: float = 0.0,
    decay: float = 0.55,
    attack: float = 0.006,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """合成一个带泛音的钢琴式单音，峰值归一化到 0.5（约 -6 dBFS）。"""
    n = max(1, int(round(duration * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)
    f0 = freq * 2.0 ** (detune_cents / 1200.0)

    out = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate(amps, start=1):
        # 轻微非谐性：真实钢琴的泛音不是严格整数倍
        fk = k * f0 * math.sqrt(1.0 + inharmonicity * k * k)
        if fk >= sr / 2.0:
            break
        phase = rng.uniform(0.0, 2.0 * math.pi)
        # 高次泛音衰减更快，模拟钢琴音色
        env = np.exp(-t / (decay / math.sqrt(k)))
        out += amp * env * np.sin(2.0 * math.pi * fk * t + phase)

    attack_n = max(1, min(n, int(round(attack * sr))))
    attack_env = np.ones(n, dtype=np.float64)
    attack_env[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=True)
    out *= attack_env

    peak = float(np.max(np.abs(out)))
    if peak > 0.0:
        out /= peak
    if noise > 0.0:
        out += noise * rng.standard_normal(n)
    return 0.5 * out


def add_silence(x: np.ndarray, pre: float = 0.3, post: float = 0.3, sr: int = 44100) -> np.ndarray:
    """前后补静音，模拟计划 4.5 节的录音规范。"""
    return np.concatenate(
        [
            np.zeros(max(0, int(round(pre * sr))), dtype=np.float64),
            np.asarray(x, dtype=np.float64).reshape(-1),
            np.zeros(max(0, int(round(post * sr))), dtype=np.float64),
        ]
    )


# 模板音色（“标准振荡器”）：基频突出，泛音依次减弱
TEMPLATE_AMPS = (1.0, 0.35, 0.18, 0.10, 0.06, 0.035, 0.02)
# 测试音色（另一 take）：泛音结构、音准、包络都不同，用于跨 take 测试
TEST_AMPS = (1.0, 0.48, 0.30, 0.16, 0.10, 0.06, 0.04)
TEST_INHARMONICITY = 2.0e-4
TEST_DETUNE_CENTS = 3.0


def make_template_audio(name: str, sr: int, seed: int) -> np.ndarray:
    """生成一个“标准音”录音波形（含前后 0.3 s 静音）。"""
    freq = float(TARGET_FREQS[NOTE_INDEX[name]])
    note = synth_note(
        freq,
        0.75,
        sr,
        amps=TEMPLATE_AMPS,
        inharmonicity=0.0,
        detune_cents=0.0,
        decay=0.60,
        seed=seed,
    )
    return add_silence(note, 0.3, 0.3, sr)


def make_test_audio(name: str, sr: int, seed: int) -> np.ndarray:
    """生成一个“另一 take”的测试录音（含前后 0.3 s 静音）。"""
    freq = float(TARGET_FREQS[NOTE_INDEX[name]])
    note = synth_note(
        freq,
        0.55,
        sr,
        amps=TEST_AMPS,
        inharmonicity=TEST_INHARMONICITY,
        detune_cents=TEST_DETUNE_CENTS,
        decay=0.45,
        noise=5.0e-4,
        seed=seed,
    )
    return add_silence(note, 0.3, 0.3, sr)


MARY_MELODY = ("E4", "D4", "C4", "D4", "E4", "E4", "E4")


def make_mary_phrase(sr: int, seed: int = 2026) -> np.ndarray:
    """合成《Mary Had a Little Lamb》前 7 个音：E4 D4 C4 D4 E4 E4 E4。"""
    parts = [np.zeros(int(0.3 * sr), dtype=np.float64)]
    for i, name in enumerate(MARY_MELODY):
        freq = float(TARGET_FREQS[NOTE_INDEX[name]])
        note = synth_note(
            freq,
            0.45,
            sr,
            amps=TEST_AMPS,
            inharmonicity=TEST_INHARMONICITY,
            detune_cents=TEST_DETUNE_CENTS,
            decay=0.40,
            noise=5.0e-4,
            seed=seed + 100 + i,
        )
        parts.append(note)
        parts.append(np.zeros(int(0.12 * sr), dtype=np.float64))
    parts.append(np.zeros(int(0.3 * sr), dtype=np.float64))
    return np.concatenate(parts)


# =============================================================================
# 8. 演示流程与输出
# =============================================================================


def _print_banner(params: Params) -> None:
    print("=" * 78)
    print(" 基于频域匹配滤波的钢琴单音 / 乐句识别 Demo")
    print(" Goertzel 离散谱线 + L2 能量归一化 + 匹配滤波相关判决")
    print("=" * 78)
    print(
        f"音域 B3-C6：{M} 个标准音（A4 = {params.a4:.1f} Hz）  |  "
        f"帧长 {params.frame_ms:g} ms / 帧移 {params.hop_ms:g} ms  |  "
        f"判决阈值 {params.min_score:.2f}"
    )
    print(f"模板矩阵 S：({M} x {N})，每一行 ||S_j||_2 = 1")
    print()


def _prepare_workdir(save_audio: Path | None) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if save_audio is None:
        tmp = tempfile.TemporaryDirectory(prefix="sigsys_demo_")
        return Path(tmp.name), tmp
    save_audio.mkdir(parents=True, exist_ok=True)
    return save_audio, None


def _write_synthetic_dataset(
    workdir: Path,
    sr: int,
    *,
    need_test: bool,
    save_audio: bool,
) -> tuple[Path, dict[str, Path], Path | None]:
    """生成标准音 / 测试音 / 乐句 WAV，返回模板目录、测试文件表、乐句文件。"""
    template_dir = workdir / "standard"
    test_dir = workdir / "test"
    template_dir.mkdir(parents=True, exist_ok=True)
    if need_test:
        test_dir.mkdir(parents=True, exist_ok=True)

    test_paths: dict[str, Path] = {}
    for i, name in enumerate(NOTE_NAMES):
        tpl_path = template_dir / f"{name}.wav"
        write_wav(tpl_path, make_template_audio(name, sr, seed=1000 + i), sr)
        if need_test:
            test_path = test_dir / f"{name}.wav"
            write_wav(test_path, make_test_audio(name, sr, seed=2000 + i), sr)
            test_paths[name] = test_path

    phrase_path: Path | None = None
    if need_test:
        phrase_path = workdir / "phrase_mary.wav"
        write_wav(phrase_path, make_mary_phrase(sr), sr)

    if save_audio:
        # 写一份 metadata，便于报告/复现实验
        meta_path = workdir / "metadata.csv"
        with meta_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "note", "midi", "freq_hz", "file"])
            for name, midi, freq in zip(NOTE_NAMES, NOTE_MIDIS, TARGET_FREQS):
                writer.writerow(["standard", name, midi, f"{freq:.2f}", f"standard/{name}.wav"])
            if need_test:
                for name, midi, freq in zip(NOTE_NAMES, NOTE_MIDIS, TARGET_FREQS):
                    writer.writerow(["test", name, midi, f"{freq:.2f}", f"test/{name}.wav"])
                writer.writerow(["phrase", "MaryHadALittleLamb", "", "", "phrase_mary.wav"])

    return template_dir, test_paths, phrase_path


def _load_or_make_templates(
    args: argparse.Namespace,
    params: Params,
    workdir: Path,
    *,
    need_test: bool,
) -> tuple[np.ndarray, dict[str, Path], Path | None]:
    """有 --templates 就读真实模板；否则生成内置合成模板。"""
    if args.templates is not None:
        templates = build_template_matrix(args.templates, params)
        if need_test:
            # 真实模板 + 合成测试音：仍然可以检查流程；音色不同，准确率仅作参考
            test_dir = workdir / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            test_paths: dict[str, Path] = {}
            for i, name in enumerate(NOTE_NAMES):
                p = test_dir / f"{name}.wav"
                write_wav(p, make_test_audio(name, args.sr, seed=2000 + i), args.sr)
                test_paths[name] = p
            phrase_path = workdir / "phrase_mary.wav"
            write_wav(phrase_path, make_mary_phrase(args.sr), args.sr)
            return templates, test_paths, phrase_path
        return templates, {}, None

    need_test = need_test or args.save_audio is not None
    template_dir, test_paths, phrase_path = _write_synthetic_dataset(
        workdir,
        args.sr,
        need_test=need_test,
        save_audio=args.save_audio is not None,
    )
    return build_template_matrix(template_dir, params), test_paths, phrase_path


def _run_self_test(
    args: argparse.Namespace,
    params: Params,
    workdir: Path,
    templates: np.ndarray,
    test_paths: dict[str, Path],
    phrase_path: Path | None,
) -> int:
    print(f"[1] 模板：{M} 个标准音，矩阵 {templates.shape}，"
          f"行范数 min/max = {np.linalg.norm(templates, axis=1).min():.6f}/"
          f"{np.linalg.norm(templates, axis=1).max():.6f}")
    print()

    # ---------------- 单音测试 ----------------
    print("[2] 单音测试：26 个音，测试 take 与模板 take 的泛音结构/音准/包络不同")
    header = f"{'编号':>4}  {'期望':<4}  {'识别':<4}  {'相关值':>7}  {'次优':<10}  {'段数':>4}  结果"
    print(header)
    print("-" * 78)
    single_ok = 0
    for i, name in enumerate(NOTE_NAMES):
        data = preprocess(str(test_paths[name]), params.frame_ms, params.hop_ms)
        segments = segment_notes(data)
        if not segments:
            print(f"{i + 1:>4}  {name:<4}  {'--':<4}  {'--':>7}  {'--':<10}  {0:>4}  未检测到")
            continue
        note = max(segments, key=len)
        result = recognize_single(note, data["sr"], templates, TARGET_FREQS, params.min_score)
        order = np.argsort(result.scores)[::-1]
        second_idx = int(order[1]) if order.size > 1 else int(order[0])
        second = f"{NOTE_NAMES[second_idx]}/{result.scores[second_idx]:.3f}"
        got = result.name if result.name is not None else "不确定"
        ok = result.name == name
        single_ok += int(ok)
        print(
            f"{i + 1:>4}  {name:<4}  {got:<4}  {result.score:>7.3f}  "
            f"{second:<10}  {len(segments):>4}  {'OK' if ok else 'XX'}"
        )

    single_acc = single_ok / M
    print("-" * 78)
    print(f"单音准确率：{single_ok}/{M} = {single_acc * 100:.2f}%")
    print()

    # ---------------- 乐句测试 ----------------
    phrase_ok = 0
    recognized_sequence: list[str] = []
    if phrase_path is not None:
        print("[3] 乐句测试：Mary Had a Little Lamb 前 7 音（E4 D4 C4 D4 E4 E4 E4）")
        data = preprocess(str(phrase_path), params.frame_ms, params.hop_ms)
        segments = _detect_segments(data, params)
        segs = [data["y_raw"][s.start : s.end] for s in segments]
        header2 = (
            f"{'序号':>4}  {'起止/s':>15}  {'时长/ms':>7}  "
            f"{'期望':<4}  {'识别':<4}  {'相关值':>7}"
        )
        print(header2)
        print("-" * 78)
        for k, (seg, note) in enumerate(zip(segments, segs), start=1):
            result = recognize_single(note, data["sr"], templates, TARGET_FREQS, params.min_score)
            expected = MARY_MELODY[k - 1] if k <= len(MARY_MELODY) else "--"
            got = result.name if result.name is not None else "rest"
            recognized_sequence.append(got)
            ok = got == expected
            phrase_ok += int(ok)
            print(
                f"{k:>4}  {seg.start_time:>6.3f}-{seg.end_time:<6.3f}  "
                f"{seg.duration_ms:>7.1f}  {expected:<4}  {got:<4}  {result.score:>7.3f}"
            )
        expected_all = list(MARY_MELODY)
        if len(segments) != len(expected_all):
            print(f"注意：检测到 {len(segments)} 段，期望 {len(expected_all)} 段。")
        phrase_acc = phrase_ok / len(MARY_MELODY)
        print("-" * 78)
        print(f"乐句准确率：{phrase_ok}/{len(MARY_MELODY)} = {phrase_acc * 100:.2f}%")
        print(f"识别序列：{' '.join(recognized_sequence)}")
        print()

    # ---------------- 结论 ----------------
    print("[4] 结论")
    print(f"  · 单音（合成模板 + 另一 take 测试）：{single_acc * 100:.2f}%")
    if phrase_path is not None:
        print(f"  · 乐句（7 个音，含 120 ms 停顿）：{phrase_acc * 100:.2f}%")
    if args.save_audio is not None:
        print(f"  · 已导出模板/测试音频、乐句与 metadata.csv：{workdir}")
    else:
        print("  · 未导出音频；需要试听/写报告时加 --save-audio <目录>。")
    print("  · 说明：以上为内置合成音频的自检结果；真实录音请用 --wav/--templates 复测。")
    print("=" * 78)
    return 0


def _run_wav(args: argparse.Namespace, params: Params, workdir: Path) -> int:
    if args.templates is not None:
        templates = build_template_matrix(args.templates, params)
        template_src = str(args.templates)
    else:
        template_dir = workdir / "standard"
        template_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(NOTE_NAMES):
            write_wav(template_dir / f"{name}.wav", make_template_audio(name, args.sr, 1000 + i), args.sr)
        templates = build_template_matrix(template_dir, params)
        template_src = "内置合成标准音"

    print(f"[1] 模板：{template_src}，矩阵 {templates.shape}")
    data = preprocess(str(args.wav), params.frame_ms, params.hop_ms)
    segments = _detect_segments(data, params)
    duration = len(data["y_raw"]) / data["sr"]
    print(
        f"[2] 输入：{args.wav}  |  sr = {data['sr']} Hz  |  时长 {duration:.3f} s  |  "
        f"检测到 {len(segments)} 段"
    )
    print()

    if not segments:
        print("未检测到音符：请确认音频不是全静音，或降低 --min-score / 检查录音电平。")
        return 0

    header = (
        f"{'序号':>4}  {'起止/s':>15}  {'时长/ms':>7}  "
        f"{'识别':<4}  {'MIDI':>4}  {'频率/Hz':>8}  {'相关值':>7}"
    )
    print(header)
    print("-" * 78)
    sequence: list[str] = []
    for k, seg in enumerate(segments, start=1):
        note = data["y_raw"][seg.start : seg.end]
        result = recognize_single(note, data["sr"], templates, TARGET_FREQS, params.min_score)
        name = result.name if result.name is not None else "rest"
        sequence.append(name)
        midi = str(result.midi) if result.midi is not None else "--"
        freq = f"{result.freq_hz:.2f}" if result.freq_hz is not None else "--"
        print(
            f"{k:>4}  {seg.start_time:>6.3f}-{seg.end_time:<6.3f}  "
            f"{seg.duration_ms:>7.1f}  {name:<4}  {midi:>4}  {freq:>8}  {result.score:>7.3f}"
        )
    print("-" * 78)
    print(f"识别序列：{' '.join(sequence)}")
    print("=" * 78)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="频域匹配滤波钢琴音识别 Demo（Goertzel + L2 归一化 + 相关判决）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（Windows）：\n"
            "  .venv\\Scripts\\python.exe demo.py\n"
            "  .venv\\Scripts\\python.exe demo.py --save-audio demo_audio\n"
            "  .venv\\Scripts\\python.exe demo.py --wav my_phrase.wav\n"
            "  .venv\\Scripts\\python.exe demo.py --wav my.wav --templates data\\templates\n"
        ),
    )
    parser.add_argument("--wav", type=Path, help="待识别的 WAV；省略则运行内置单音+乐句自检")
    parser.add_argument(
        "--templates",
        type=Path,
        help="26 个标准音 WAV 所在目录（B3.wav … C6.wav）；省略则用内置合成标准音",
    )
    parser.add_argument(
        "--save-audio",
        type=Path,
        help="把内置合成的标准音/测试音/乐句 WAV 与 metadata.csv 保存到该目录",
    )
    parser.add_argument("--sr", type=int, default=44100, help="内置合成音频采样率（默认 44100）")
    parser.add_argument("--frame-ms", type=float, default=25.0, help="帧长 ms（默认 25）")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="帧移 ms（默认 10）")
    parser.add_argument("--min-score", type=float, default=0.35, help="单音判决阈值（默认 0.35）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sr <= 0:
        print("[错误] --sr 必须为正整数", file=sys.stderr)
        return 2
    params = Params(frame_ms=args.frame_ms, hop_ms=args.hop_ms, min_score=args.min_score)

    _print_banner(params)
    workdir, tmp = _prepare_workdir(args.save_audio)
    try:
        if args.wav is not None:
            return _run_wav(args, params, workdir)

        templates, test_paths, phrase_path = _load_or_make_templates(
            args, params, workdir, need_test=True
        )
        return _run_self_test(args, params, workdir, templates, test_paths, phrase_path)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
