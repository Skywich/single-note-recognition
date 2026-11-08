# -*- coding: utf-8 -*-
"""
demo_preprocess.py —— 不依赖你自己的录音，先生成一段测试音频，跑通 preprocess()

运行（在项目根目录下）：
    py src\\demo_preprocess.py

它会做 4 件事：
    1. 生成 data/demo_tone.wav
       （0.3s 静音 + 0.5s 440Hz + 0.2s 静音 + 0.5s 660Hz + 0.3s 静音）
    2. 调用 preprocess()
    3. 打印返回值，做 4 个自检
    4. 把波形图和每帧能量图画到 output/preprocess_demo.png，并弹窗显示
"""

import os

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from preprocess_m import preprocess


def make_tone(freq, duration, sr, amplitude=0.6):
    """生成一段正弦波（单音）。"""
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def make_demo_wav(wav_path, sr=44100):
    """生成一段"静音-单音-静音-单音-静音"的测试 wav。"""
    parts = [
        np.zeros(int(0.3 * sr)),          # 0.3 s 静音
        make_tone(440.0, 0.5, sr),        # 0.5 s A4（440 Hz）
        np.zeros(int(0.2 * sr)),          # 0.2 s 静音
        make_tone(660.0, 0.5, sr),        # 0.5 s E5（659.25 Hz 附近）
        np.zeros(int(0.3 * sr)),          # 0.3 s 静音
    ]
    y = np.concatenate(parts)

    # wav 的 16-bit PCM 用 int16 存，范围 -32768 ~ 32767。
    # 先乘 32767 再裁剪，防止溢出，最后转成 int16。
    y_int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)

    wavfile.write(wav_path, sr, y_int16)
    return y_int16, sr


def main():
    # 以本文件所在目录的上一级作为项目根目录，这样无论从哪里运行都能找对路径。
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. 生成测试音频 ----------
    wav_path = os.path.join(data_dir, "demo_tone.wav")
    y_int16, sr = make_demo_wav(wav_path, sr=44100)
    print("已生成测试音频：", wav_path)
    print(f"采样率 = {sr}，采样点数 = {y_int16.size}，时长 = {y_int16.size / sr:.2f} s")

    # ---------- 2. 调用 preprocess ----------
    ctx = preprocess(wav_path, frame_ms=25.0, hop_ms=10.0)

    # ---------- 3. 打印返回值 ----------
    print("\n" + "=" * 60)
    print("preprocess() 返回的字典")
    print("=" * 60)
    for key, value in ctx.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:10s} shape={str(value.shape):12s} dtype={value.dtype}")
        else:
            print(f"  {key:10s} {value}")

    # ---------- 4. 四个自检 ----------
    print("\n" + "=" * 60)
    print("自检")
    print("=" * 60)

    expected_keys = {"y_raw", "frames", "sr", "frame_len", "hop_len"}
    check1 = set(ctx) == expected_keys
    print("1) 是否严格 5 个键：", "通过" if check1 else "失败")

    peak = float(np.max(np.abs(ctx["y_raw"])))
    check2 = peak == 1.0
    print(f"2) max(abs(y_raw)) = {peak:.15f}：", "通过" if check2 else "失败")

    check3 = np.array_equal(ctx["frames"][0], ctx["y_raw"][: ctx["frame_len"]])
    print("3) frames[0] 是否等于 y_raw 前 frame_len 个点：", "通过" if check3 else "失败")

    last_end = (ctx["frames"].shape[0] - 1) * ctx["hop_len"] + ctx["frame_len"]
    check4 = last_end <= len(ctx["y_raw"])
    print(f"4) 最后一帧终点 {last_end} <= 信号长度 {len(ctx['y_raw'])}：",
          "通过" if check4 else "失败")

    # ---------- 5. 画图 ----------
    # 让 matplotlib 能正常显示中文（Windows 自带微软雅黑；后面几项是其他系统的备选）。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # 上图：整段归一化波形，并用橙色标出其中一帧的位置。
    ax = axes[0]
    t = np.arange(len(ctx["y_raw"])) / ctx["sr"]
    ax.plot(t, ctx["y_raw"], linewidth=0.8)
    k = 20
    start_s = k * ctx["hop_len"] / ctx["sr"]
    end_s = (k * ctx["hop_len"] + ctx["frame_len"]) / ctx["sr"]
    ax.axvspan(start_s, end_s, color="orange", alpha=0.3,
               label=f"第 {k} 帧（{end_s - start_s:.3f} s）")
    ax.set_title("归一化后的整段信号 y_raw")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("幅度")
    ax.legend(loc="upper right")

    # 下图：每帧的短时能量，提前看看模块 2 会用的特征长什么样。
    ax = axes[1]
    energy = np.sum(ctx["frames"] ** 2, axis=1)
    frame_t = (np.arange(len(energy)) * ctx["hop_len"] + ctx["frame_len"] / 2) / ctx["sr"]
    ax.plot(frame_t, energy, linewidth=0.8)
    ax.set_title("每帧的短时能量（模块 2 端点检测会用的特征）")
    ax.set_xlabel("帧中心时间 / s")
    ax.set_ylabel("能量")

    plt.tight_layout()
    out_png = os.path.join(out_dir, "preprocess_demo.png")
    plt.savefig(out_png, dpi=150)
    print("\n图已保存：", out_png)

    # 如果不想弹窗，把下面这行注释掉即可。
    plt.show()


if __name__ == "__main__":
    main()
