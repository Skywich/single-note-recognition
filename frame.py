#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《信号与系统（1）》课程项目最终成品框架 —— 频域匹配滤波钢琴音识别。

本文件是给 5 人小组合作使用的“成品骨架”：
  · 五个板块的公开函数/类已经按《信号与系统课程项目计划 v1.0.6》第 5 节
    规定的输入、输出接口写好签名和 docstring；
  · 每个板块的函数体保留为 NotImplementedError，由对应成员填空；
  · 主流程、参数、26 音网格、CLI 已经接好，填完各板块后即可联调；
  · 不在本文件中放算法实现，避免与 demo.py 重复、也避免“框架”变成某一个人的答案。

对应关系（计划第 5 节）：
    板块 1  音频前期处理        preprocess()              成员 1 / Skywich
    板块 2  音频时域处理        segment_notes()           成员 2 / banS
    板块 3  音频频域处理        calc_goertzel_spectrum()  成员 3 / 天启
    板块 4  匹配滤波器与判决    match_filter_decision()   成员 4 / saku
    板块 5  GUI 与标准音数据    RecognitionGUI / launch_gui()  成员 5 / FeO

运行方式：
    python frame.py --check
        打印五个板块的接口清单和填充进度。

    python frame.py --templates data/templates --audio data/test.wav
        按最终成品流程跑一遍：模板 → 预处理 → 分割 → Goertzel → 判决。

    python frame.py --gui
        启动最终 GUI（待成员 5 实现）。

依赖：numpy。GUI 建议使用标准库 tkinter（无需额外安装）。

可运行的参考实现见同目录 demo.py；本文件只保留接口和 TODO，成员应按 docstring
独立实现，不要把 demo.py 的算法直接复制进来（否则验收答辩时讲不清）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "frame.py 需要 numpy。\n"
        "请先安装：python -m pip install numpy\n"
        "本仓库内也可直接使用：.venv/bin/python frame.py --check"
    ) from exc


# =============================================================================
# 0. 全组共用配置：音域、音名、频率、参数
#    这部分已经实现，不需要成员修改；四个人都从这里取 TARGET_FREQS。
# =============================================================================

NOTE_NAMES_12 = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# 论文表 1 / 计划 §5：识别范围 B3~C6，共 26 个音。
RANGE_START_MIDI = 59  # B3
RANGE_END_MIDI = 84    # C6
A4_HZ = 440.0


def midi_to_name(midi: int) -> str:
    """MIDI -> 科学音高记号，例如 60 -> C4。"""
    return f"{NOTE_NAMES_12[midi % 12]}{midi // 12 - 1}"


def midi_to_freq(midi: int, a4: float = A4_HZ) -> float:
    """十二平均律：f = A4 · 2^((midi - 69) / 12)。"""
    return float(a4 * 2.0 ** ((midi - 69) / 12.0))


def build_note_grid(
    start_midi: int = RANGE_START_MIDI,
    end_midi: int = RANGE_END_MIDI,
    a4: float = A4_HZ,
) -> tuple[list[str], np.ndarray, list[int]]:
    """生成 (音名列表, 基频数组, MIDI 列表)。"""
    midis = list(range(start_midi, end_midi + 1))
    names = [midi_to_name(m) for m in midis]
    freqs = np.array([midi_to_freq(m, a4) for m in midis], dtype=np.float64)
    return names, freqs, midis


NOTE_NAMES, TARGET_FREQS, NOTE_MIDIS = build_note_grid()
NOTE_INDEX = {name: i for i, name in enumerate(NOTE_NAMES)}

#: 模板数 M
M = len(NOTE_NAMES)
#: 离散频谱序列长度 N（本项目 N = M = 26）
N = len(TARGET_FREQS)
assert M == N == 26, "B3~C6 应为 26 个音；如改音域，请同步修改模板数量和报告"


@dataclass(frozen=True)
class PipelineParams:
    """计划 4.2 节的建议参数。成员可在此集中调参，不要散落在各文件里。"""

    frame_ms: float = 25.0
    hop_ms: float = 10.0
    min_note_ms: float = 50.0
    merge_gap_ms: float = 50.0
    pre_margin_ms: float = 10.0
    post_margin_ms: float = 20.0
    min_score: float = 0.35
    a4: float = A4_HZ


DEFAULT_PARAMS = PipelineParams()


@dataclass(frozen=True)
class NoteSegment:
    """板块 2 的可选扩展：带时间信息的音符片段。"""

    start_sample: int
    end_sample: int
    sr: int

    @property
    def start_time(self) -> float:
        return self.start_sample / float(self.sr)

    @property
    def end_time(self) -> float:
        return self.end_sample / float(self.sr)

    @property
    def duration_ms(self) -> float:
        return (self.end_sample - self.start_sample) * 1000.0 / self.sr


@dataclass
class RecognitionResult:
    """板块 4 / 主流程共用的单音识别结果。"""

    idx: int | None          # 0-25；None 表示低于阈值，判为静音/不确定
    name: str | None         # 例如 "C4"；None 表示 rest
    midi: int | None
    freq_hz: float | None
    score: float             # 最大相关值，0~1
    scores: np.ndarray       # 长度 M 的全部相关值，供 GUI 显示 top-K
    start_time: float | None = None
    end_time: float | None = None

    @property
    def confident(self) -> bool:
        return self.idx is not None


# =============================================================================
# 板块 1：音频前期处理（成员 1 / Skywich，对应计划中的 preprocess.py）
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
        WAV 文件路径。建议 44.1 kHz、16 bit、单声道；双声道取平均或左声道。
    frame_ms : float
        帧长（毫秒），默认 25。
    hop_ms : float
        帧移（毫秒），默认 10。

    返回（字典的键名和类型不可改，下游三个板块直接依赖）
    ----
    dict:
        "y_raw"     : 1D np.ndarray，按 max(|x|) 归一化后的完整波形。
                      注意：音符分割最终切的是这个数组，不是 frames。
        "frames"    : 2D np.ndarray，形状 (帧数, frame_len)。
        "sr"        : int，采样率。
        "frame_len" : int，帧长对应的采样点数。
        "hop_len"   : int，帧移对应的采样点数。

    实现要点
    --------
    1. 用 wave / scipy.io.wavfile 读取，多声道转单声道；
    2. y_raw = y / (max(abs(y)) + eps)，严格按“除以最大绝对值”归一化；
    3. frame_len = round(frame_ms * sr / 1000)，hop_len 同理；
    4. frames 每行是一帧原始采样（不要在这里加窗，加窗属于板块 3）；
    5. 最后一帧不足时补零，保证 frames 形状永远是二维。

    验收
    ----
    · data["frames"].shape == (帧数, data["frame_len"])；
    · 全静音文件不崩溃（可返回全零 y_raw 和全零 frames）；
    · 同一文件重复调用结果一致。
    """
    # TODO(成员1 / Skywich):
    #   在这里实现音频读取、单声道化、幅度归一化和分帧。
    #   建议先写 read_wav() 私有辅助函数，再组装下面的返回字典。
    raise NotImplementedError("板块 1 preprocess() 尚未实现（成员 1 / Skywich）")


# =============================================================================
# 板块 2：音频时域处理（成员 2 / banS，对应计划中的 segment_notes.py）
# =============================================================================


def segment_notes(data_dict: dict) -> list[np.ndarray]:
    """根据分帧矩阵计算短时能量 STE 和短时过零率 ZCR，切出所有单音。

    参数
    ----
    data_dict : dict
        板块 1 的输出字典，必须包含 y_raw / frames / sr / frame_len / hop_len。

    返回
    ----
    list[np.ndarray]
        列表，每个元素是一个单音的一维 NumPy 数组；没有音符时返回 []。
        数组必须从 data_dict["y_raw"] 中切片，不要对 frames 切片。

    实现要点（计划 4.2 节）
    ----------------------
    1. 逐帧 STE = mean(frames[i] ** 2)；
    2. 逐帧 ZCR = mean(frames[i, 1:] * frames[i, :-1] < 0)；
    3. 高阈值 = max(噪声能量 * 4, max(STE) * 0.01)，用于触发起点；
       低阈值 = max(噪声能量 * 2, max(STE) * 0.005)，用于滞后延伸；
       噪声能量可用最低 10% 帧的能量估计；
    4. 起止帧要换算回采样点：sample = frame_index * hop_len，
       结束点还要加 frame_len；起止各留 10/20 ms 余量；
    5. 最短音符 50 ms，间隔小于 50 ms 的片段合并；
    6. 建议用 ZCR 门限（例如 < 0.45）排除宽带噪声。

    验收
    ----
    · 单音文件返回长度 1 的列表；7 音旋律返回 7 个数组；
    · 每个数组长度 >= 50 ms * sr；
    · 不修改 data_dict 中的任何数组。
    """
    # TODO(成员2 / banS):
    #   实现 _frame_features() 和端点检测状态机，再按 y_raw[start:end] 切片。
    raise NotImplementedError("板块 2 segment_notes() 尚未实现（成员 2 / banS）")


def segment_notes_detailed(data_dict: dict) -> list[NoteSegment]:
    """【可选扩展】与 segment_notes 相同，但额外返回每个音符的起止采样点。

    GUI 需要显示“第 1 个音 0.27–0.785 s”时使用本函数；
    如果成员 2 只实现 segment_notes，可暂时不实现本函数，GUI 只显示序号。

    返回
    ----
    list[NoteSegment]，start_sample/end_sample 为 y_raw 的下标，左闭右开。
    """
    # TODO(成员2 / banS，可选): 建议让 segment_notes 调用本函数，再统一映射为数组，
    #   这样两个接口不会出现两套阈值逻辑。
    raise NotImplementedError("板块 2 segment_notes_detailed() 尚未实现（可选）")


# =============================================================================
# 板块 3：音频频域处理（成员 3 / 天启，对应 calc_goertzel_spectrum.py）
# =============================================================================


def calc_goertzel_spectrum(
    note_signal: np.ndarray,
    sr: int,
    target_freqs: np.ndarray,
) -> np.ndarray:
    """用 Goertzel 计算单音在 26 个基频处的离散谱线，并做 L2 能量归一化。

    参数
    ----
    note_signal : 1D np.ndarray
        单个音符的时域波形（来自板块 2）。
    sr : int
        采样率。
    target_freqs : 1D np.ndarray
        26 个标准音基频，长度 M=26，由 TARGET_FREQS 提供。

    返回
    ----
    np.ndarray
        长度 N=26 的归一化离散频谱序列；
        必须满足 ||S_i||_2 = 1（全零输入可返回全零向量）。

    公式（计划 1.4 节）
    ------------------
        ω0 = 2π f0 / fs
        s[n] = x[n] + 2cos(ω0) s[n-1] - s[n-2],  s[-1]=s[-2]=0
        |X(ω0)|² = s[N-1]² + s[N-2]² - 2cos(ω0) s[N-1] s[N-2]
        S_i = |X(ω0)| / || |X(ω0)| ||_2

    实现要点
    --------
    1. 建议先对 note_signal 加 Hamming 窗，减小截断泄漏；
    2. 对 26 个频点分别递推即可；本项目数据量小，不需要 FFT；
    3. 输入也要归一化（主流程会先除以峰值，板块 3 再做 L2 归一化）；
    4. 目标频率必须满足 0 < f < sr/2。

    验收
    ----
    · 返回 shape == (26,)；dtype 为 float；
    · np.linalg.norm(S_i) 约等于 1；
    · 与 np.fft.fft 在对应频点的幅度对比，相对误差 < 1e-6（加窗后）。
    """
    # TODO(成员3 / 天启):
    #   1. 参数检查；2. 加 Hamming 窗；3. 26 路 Goertzel 递推；
    #   4. 功率 -> 幅度；5. L2 归一化。
    raise NotImplementedError("板块 3 calc_goertzel_spectrum() 尚未实现（成员 3 / 天启）")


# =============================================================================
# 板块 4：匹配滤波器与判决（成员 4 / saku，对应 match_filter_decision.py）
# =============================================================================


def match_filter_decision(
    S_i: np.ndarray,
    templates: np.ndarray,
) -> tuple[int, float]:
    """把输入谱向量并行送入 M 路匹配滤波器，在 n=N-1 处采样并取最大值。

    参数
    ----
    S_i : 1D np.ndarray
        板块 3 输出的归一化频谱序列，长度 N=26。
    templates : 2D np.ndarray
        标准音模板矩阵，形状 (M, N) = (26, 26)；每一行是一个归一化模板 S_j。

    返回
    ----
    (best_note_idx, max_corr_value) : (int, float)
        best_note_idx ∈ [0, 25]；max_corr_value 为最大相关值，用作置信度。

    原理（计划 1.4 节）
    ------------------
        h_j[n] = S_j[N-1-n]
        y_j[N-1] = Σ_k S_i[k] h_j[N-1-k] = Σ_k S_i[k] S_j[k] = r_ij[0]
    因此不需要真的做卷积，直接做点积即可。归一化后由 Cauchy–Schwarz：
        0 <= r_ij <= 1 = r_ii
    所以取 argmax 即可。

    实现要点
    --------
    1. 校验 S_i 长度等于 templates.shape[1]；
    2. 模板行和输入向量的 L2 范数都应为 1（可做防御性归一化）；
    3. 用 templates @ S_i 一次算出 M 个相关值；
    4. 返回 np.argmax 和对应值，类型分别为 int、float。

    验收
    ----
    · 输入自己的模板时 max_corr_value 接近 1，且 best_note_idx 正确；
    · 输出值在 [0, 1] 内（数值误差允许 1e-9）。
    """
    # TODO(成员4 / saku):
    #   实现相关值计算、最大值判决、置信度返回。
    raise NotImplementedError("板块 4 match_filter_decision() 尚未实现（成员 4 / saku）")


def match_filter_scores(S_i: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """【可选扩展】返回全部 M 个相关值，供 GUI 显示 top-K / 和弦候选。"""
    # TODO(成员4 / saku，可选): 建议让 match_filter_decision 调用本函数，
    #   避免“判决”和“top-K 显示”各写一套点积。
    raise NotImplementedError("板块 4 match_filter_scores() 尚未实现（可选）")


# =============================================================================
# 板块 4/5 之间的胶水：标准音模板矩阵
# =============================================================================


def build_template_matrix(
    template_dir: str | Path,
    params: PipelineParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """把 26 个标准音 WAV 变成 (M, N) 的归一化模板矩阵。

    参数
    ----
    template_dir : str | Path
        目录下应有 B3.wav、C4.wav、C#4.wav … C6.wav，共 26 个文件；
        每个文件是一个单独录制的标准音，前后留 0.3 s 静音。
    params : PipelineParams
        分帧参数。

    返回
    ----
    np.ndarray
        形状 (26, 26)，第 j 行是第 j 个标准音的归一化频谱 S_j。

    实现步骤（可直接复用板块 1–3）
    ------------------------------
    for name in NOTE_NAMES:
        data = preprocess(f"{template_dir}/{name}.wav", params.frame_ms, params.hop_ms)
        segments = segment_notes(data)
        note = max(segments, key=len)                     # 单音文件取最长段
        row = calc_goertzel_spectrum(note, data["sr"], TARGET_FREQS)
        rows.append(row)
    return np.vstack(rows)

    验收
    ----
    · 返回 shape == (26, 26)；每行范数为 1；
    · 缺文件时给出明确错误（列出缺哪几个音）；
    · 模板与测试数据必须来自不同 take（计划 4.5 节）。
    """
    # TODO(成员4 / saku + 成员5 / FeO):
    #   实现上面的循环；文件命名、缺文件提示、最长段选择都要处理。
    raise NotImplementedError("模板矩阵 build_template_matrix() 尚未实现（成员 4/5）")


# =============================================================================
# 主流程：单音识别 / 乐句识别（成员 1 / Skywich 串联）
# =============================================================================


def recognize_single(
    note_signal: np.ndarray,
    sr: int,
    templates: np.ndarray,
    params: PipelineParams = DEFAULT_PARAMS,
) -> RecognitionResult:
    """单音识别：幅度归一化 → Goertzel → 匹配滤波 → 阈值判决。

    返回
    ----
    RecognitionResult
        idx/name/midi/freq_hz 为 None 表示低于 params.min_score，判为 rest。
        scores 保留全部 26 个相关值，GUI 可据此显示 top-3。
    """
    # TODO(成员1 / Skywich):
    #   x = note_signal / (max(abs(note_signal)) + eps)
    #   S = calc_goertzel_spectrum(x, sr, TARGET_FREQS)
    #   idx, score = match_filter_decision(S, templates)
    #   若 score < params.min_score -> RecognitionResult(None, ...)
    raise NotImplementedError("主流程 recognize_single() 尚未实现（成员 1 / Skywich）")


def recognize_phrase(
    audio_path: str,
    templates: np.ndarray,
    params: PipelineParams = DEFAULT_PARAMS,
) -> list[RecognitionResult]:
    """整段音频识别：预处理 → 端点检测 → 逐音识别。

    参数
    ----
    audio_path : str
        包含一个或多个单音的 WAV。
    templates : 2D np.ndarray
        build_template_matrix() 的输出，(26, 26)。
    params : PipelineParams

    返回
    ----
    list[RecognitionResult]
        按时间顺序排列；低于阈值的音以 name=None 表示 rest。
        如果实现了 segment_notes_detailed，请把 start_time/end_time 填进结果。

    实现步骤
    --------
    data = preprocess(audio_path, params.frame_ms, params.hop_ms)
    segments = segment_notes(data)
    results = [recognize_single(seg, data["sr"], templates, params) for seg in segments]
    return results

    验收
    ----
    · 单音文件返回 1 个结果，7 音旋律返回 7 个结果；
    · 输出顺序与音频时间顺序一致；
    · 不把静音段输出成音符。
    """
    # TODO(成员1 / Skywich): 串联板块 1、2、4，并保留时间戳。
    raise NotImplementedError("主流程 recognize_phrase() 尚未实现（成员 1 / Skywich）")


# =============================================================================
# 板块 5：GUI（成员 5 / FeO）
# =============================================================================


class RecognitionGUI:
    """最终成品的 GUI 骨架（建议 tkinter，标准库自带）。

    界面至少包含：
      · 音频文件选择窗口（文件对话框，支持 .wav）；
      · “开始识别”按钮；
      · 结果输出窗口：音名序列、MIDI、频率、相关值、时间戳；
      · 可选：波形图 / 离散谱线柱状图 / top-K 表格。

    使用方式：
        root = tkinter.Tk()
        app = RecognitionGUI(root)
        root.mainloop()
    """

    def __init__(self, master: Any, params: PipelineParams = DEFAULT_PARAMS) -> None:
        # TODO(成员5 / FeO): 保存 master/params，创建控件，绑定按钮回调。
        raise NotImplementedError("GUI 初始化尚未实现（成员 5 / FeO）")

    def choose_audio(self) -> str | None:
        """弹出文件选择框，返回选中的 WAV 路径；取消则返回 None。"""
        # TODO(成员5 / FeO): tkinter.filedialog.askopenfilename(...)
        raise NotImplementedError("GUI 文件选择尚未实现（成员 5 / FeO）")

    def run_recognition(self) -> None:
        """读取当前音频，调用 recognize_phrase()，把结果交给 show_results()。"""
        # TODO(成员5 / FeO): 需要 --templates 对应的模板矩阵；可先缓存到 self.templates。
        raise NotImplementedError("GUI 识别回调尚未实现（成员 5 / FeO）")

    def show_results(self, results: list[RecognitionResult]) -> None:
        """把识别结果写入文本框/表格；rest 显示为“—”。"""
        # TODO(成员5 / FeO)
        raise NotImplementedError("GUI 结果展示尚未实现（成员 5 / FeO）")


def launch_gui(audio_path: str | None = None) -> None:
    """启动 GUI；audio_path 不为 None 时预填文件路径。"""
    # TODO(成员5 / FeO):
    #   import tkinter as tk
    #   root = tk.Tk(); root.title("钢琴音识别")
    #   app = RecognitionGUI(root); root.mainloop()
    raise NotImplementedError("GUI 启动尚未实现（成员 5 / FeO）")


# =============================================================================
# CLI 与接口自检
# =============================================================================


_STATUS_ROWS = (
    ("1", "preprocess", "成员1 / Skywich", "dict"),
    ("2", "segment_notes", "成员2 / banS", "list[np.ndarray]"),
    ("3", "calc_goertzel_spectrum", "成员3 / 天启", "np.ndarray (26,)"),
    ("4", "match_filter_decision", "成员4 / saku", "tuple[int, float]"),
    ("5", "RecognitionGUI / launch_gui", "成员5 / FeO", "GUI"),
)


def _print_check() -> None:
    print("=" * 78)
    print(" 最终成品框架 frame.py —— 五个板块接口清单")
    print("=" * 78)
    print(f"音域：B3~C6，{M} 个标准音；模板矩阵 S 形状 ({M}, {N})；A4 = {A4_HZ:.1f} Hz")
    print()
    print(f"{'序号':<4}{'板块/函数':<32}{'负责人':<18}{'输出'}")
    print("-" * 78)
    for no, name, owner, out in _STATUS_ROWS:
        print(f"{no:<4}{name:<32}{owner:<18}{out}")
    print("-" * 78)
    print("当前状态：以上函数/类均为占位，调用时会 raise NotImplementedError。")
    print(
        "\n填充顺序建议：1 → 2 → 3 → 4 → 5。\n"
        "每个函数的 docstring 都写了输入、输出、实现要点和验收标准；\n"
        "填完后用下面命令联调：\n"
        "    python frame.py --templates data/templates --audio data/test.wav\n"
        "    python frame.py --gui\n"
        "\n注意：模板标准音与测试音频必须来自不同 take（计划 4.5 节）。"
    )
    print("=" * 78)


def _print_results(results: Sequence[RecognitionResult]) -> None:
    print(f"{'序号':>4}  {'音名':<4}  {'MIDI':>4}  {'频率/Hz':>8}  {'相关值':>7}  起止/s")
    print("-" * 60)
    for k, r in enumerate(results, start=1):
        name = r.name if r.name is not None else "rest"
        midi = str(r.midi) if r.midi is not None else "--"
        freq = f"{r.freq_hz:.2f}" if r.freq_hz is not None else "--"
        span = (
            f"{r.start_time:.3f}-{r.end_time:.3f}"
            if r.start_time is not None and r.end_time is not None
            else "--"
        )
        print(f"{k:>4}  {name:<4}  {midi:>4}  {freq:>8}  {r.score:>7.3f}  {span}")
    print("-" * 60)
    sequence = " ".join(r.name if r.name is not None else "rest" for r in results)
    print(f"识别序列：{sequence}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="最终成品框架（各板块待实现）；用 --check 查看接口清单",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", type=str, help="待识别的 WAV")
    parser.add_argument(
        "--templates",
        type=Path,
        help="26 个标准音 WAV 所在目录（B3.wav … C6.wav）",
    )
    parser.add_argument("--gui", action="store_true", help="启动 GUI")
    parser.add_argument("--check", action="store_true", help="打印五个板块接口清单后退出")
    args = parser.parse_args(argv)

    if args.check or (not args.audio and not args.gui):
        _print_check()
        return 0

    if args.gui:
        try:
            launch_gui(args.audio)
        except NotImplementedError as exc:
            print(f"[未实现] {exc}", file=sys.stderr)
            return 2
        return 0

    if args.templates is None:
        print(
            "最终成品需要 26 个标准音模板：请用 --templates 指定目录。\n"
            "目录内应包含 B3.wav、C4.wav、C#4.wav … C6.wav。",
            file=sys.stderr,
        )
        return 2

    try:
        templates = build_template_matrix(args.templates, DEFAULT_PARAMS)
        results = recognize_phrase(args.audio, templates, DEFAULT_PARAMS)
    except NotImplementedError as exc:
        print(f"[未实现] {exc}", file=sys.stderr)
        print(
            "请按 docstring 完成对应板块后重试；接口清单见 `python frame.py --check`。",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
