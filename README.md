# 音频前期处理模块（M1）README

> **项目**：《信号与系统（1）》课程项目——基于频域匹配滤波的钢琴音阶识别
> **模块**：M1 音频前期处理（读 wav → 幅度归一化 → 分帧）
> **规范实现**：[src/preprocess.py](src/preprocess.py)（`src/preprocess_m.py` 是同接口的个人练习版，下游统一用 `preprocess.py`）
> **接口版本**：v1.0（冻结）　**更新日期**：2026-11-08
> **下游**：M2 时域分割、M3 Goertzel、M4 匹配滤波、M5 GUI
> **配套文档**：[设计计划](docs/音频前期处理脚本计划.md)

---

## 1. 这个模块做什么

输入一个 wav 文件路径，输出一个字典，包含下游需要的**全部上下文**：

- `y_raw`：归一化后的整段一维信号，**切音时切它**；
- `frames`：分帧后的二维矩阵，**算短时能量/过零率、画图时用它**；
- `sr`、`frame_len`、`hop_len`：采样率、帧长点数、帧移点数，**换算时用**。

一句话：**下游拿到这个字典后，不需要再读 wav、再归一化、再分帧。**

---

## 2. 快速开始

```python
from preprocess import preprocess

ctx = preprocess("data/demo_tone.wav")          # 默认 25ms / 10ms
ctx = preprocess("data/demo_tone.wav", frame_ms=25.0, hop_ms=10.0)
```

如果当前目录不是 `src`，先把 `src` 加进 `sys.path`：

```python
import sys
sys.path.insert(0, "src")
from preprocess import preprocess
```

或者直接在 `src` 目录下运行 Python。

命令行自检：

```powershell
python src\demo_preprocess.py                  # 先生成 data/demo_tone.wav
python src\preprocess.py data\demo_tone.wav    # 再单独跑预处理
```

---

## 3. 环境与依赖

| 依赖 | 最低版本 | 用途 |
|---|---|---|
| Python | 3.9+ | 类型注解、`pathlib` |
| NumPy | 1.20+ | `sliding_window_view`、数组运算 |
| SciPy | 1.7+ | `scipy.io.wavfile` 读写 wav |

```powershell
python -m pip install -r requirements.txt
```

**格式限制**：只支持 **wav**（PCM 或 float）。mp3 请在 GUI 层先转成 wav 再传进来。

---

## 4. API

### 4.1 函数签名

```python
def preprocess(audio_path: str, frame_ms: float = 25.0, hop_ms: float = 10.0) -> dict:
```

### 4.2 输入参数

| 参数 | 类型 | 默认 | 约束 | 说明 |
|---|---|---|---|---|
| `audio_path` | `str` | 必填 | 文件存在、可读、wav | wav 文件路径；也接受 `pathlib.Path` |
| `frame_ms` | `float` | `25.0` | `> 0` 且 `>= hop_ms` | 帧长（毫秒） |
| `hop_ms` | `float` | `10.0` | `> 0` 且 `<= frame_ms` | 帧移（毫秒） |

- `hop_ms > frame_ms` 会直接抛 `ValueError`，因为帧间会留空隙，破坏端点检测的连续性。
- **不做重采样**：文件是多少 Hz，`sr` 就是多少 Hz；M3 必须用 `ctx["sr"]` 算频率。

### 4.3 返回值（严格 5 个键）

| 键 | 类型 | 形状 | 取值范围 | 用途 | 谁用 |
|---|---|---|---|---|---|
| `y_raw` | `np.ndarray` (`float64`) | `(L,)` 一维 | 非静音时 `[-1, 1]`，峰值精确为 `±1` | 归一化后的整段信号，**切音的唯一数据源** | M2、M5 |
| `frames` | `np.ndarray` (`float64`) | `(n_frames, frame_len)` 二维 | 与 `y_raw` 同量级 | 短时能量、过零率、分帧可视化 | M2、M5 |
| `sr` | `int` | 标量 | 原文件采样率，如 `44100` | Goertzel 频率轴、时间换算 | M2/M3/M4/M5 |
| `frame_len` | `int` | 标量 | 帧长对应点数，如 `1103` | 帧 ↔ 采样点换算 | M2/M3 |
| `hop_len` | `int` | 标量 | 帧移对应点数，如 `441` | 帧 ↔ 采样点换算 | M2/M3 |

**返回示例**（44.1kHz、1.8s 的 demo）：

```python
{
    "y_raw":     np.ndarray,   # shape (79380,), dtype float64, max(abs) == 1.0
    "frames":    np.ndarray,   # shape (178, 1103), dtype float64, C 连续可写
    "sr":        44100,        # int
    "frame_len": 1103,         # int
    "hop_len":   441,          # int
}
```

**明确不返回**：`n_frames`（用 `frames.shape[0]`）、`duration`、归一化前的原始信号、`peak`、窗函数、任何调试字段。

---

## 5. 帧与采样点的对齐（M2 必读）

设 `L = len(y_raw)`，第 `k` 帧从 0 开始：

| 量 | 公式 |
|---|---|
| 起点采样（含） | `k * hop_len` |
| 终点采样（不含） | `k * hop_len + frame_len` |
| 中心时间（秒） | `(k * hop_len + frame_len / 2) / sr` |
| 帧数 | `n_frames = 1 + (L - frame_len) // hop_len`（`L >= frame_len` 时） |

**核心不变量**：对所有完整帧，

```python
frames[k] == y_raw[k * hop_len : k * hop_len + frame_len]
```

### 5.1 尾部是怎么处理的

只保留完整帧，不足一帧的尾部**丢弃**，与 MATLAB `enframe` 一致：

```
丢弃长度 = (L - frame_len) % hop_len
```

所以丢弃的是 **`< hop_len`** 的采样，不是一整个 `frame_len`。默认 44.1kHz/25ms/10ms 时，丢弃 **< 10ms**。

以 1 秒音频为例：

```
L = 44100, frame_len = 1103, hop_len = 441
n_frames = 1 + (44100 - 1103) // 441 = 98
最后一帧：42777 ~ 43880
丢弃尾部：44100 - 43880 = 220 点（5.0ms）
```

> 例外：`L < frame_len` 时不会丢成 0 帧，而是补零成 1 帧；此时 `y_raw` 长度仍是 `L`，只有 `frames` 补了零。补零会拉低这一帧的短时能量，M2 对“整段不到一帧”的音频要单独处理。

### 5.2 切一个音符（含最后一帧特判）

```python
n_frames = ctx["frames"].shape[0]
L = len(ctx["y_raw"])

start = k0 * ctx["hop_len"]
if k1 == n_frames - 1:
    # 最后一个音一直响到文件末尾：把被丢弃的尾部也包进来
    end = L
else:
    end = k1 * ctx["hop_len"] + ctx["frame_len"]

note = ctx["y_raw"][start:end]
```

**注意**：`end = min(k1 * hop_len + frame_len, L)` 只负责“防止越界”，**不会**把丢弃的尾部补回来。如果最后一个音需要覆盖到文件末尾，必须用上面的 `if k1 == n_frames - 1: end = L`。

---

## 6. 下游调用示例

### 6.1 M2：短时能量 + 过零率 + 切音

```python
import sys
sys.path.insert(0, "src")   # 从项目根目录运行时需要

import numpy as np
from preprocess import preprocess

ctx = preprocess("data/phrase.wav")

frames = ctx["frames"]          # (n_frames, frame_len)
sr = ctx["sr"]
frame_len = ctx["frame_len"]
hop_len = ctx["hop_len"]
L = len(ctx["y_raw"])

# 短时能量：每一帧的平方和
energy = np.sum(frames ** 2, axis=1)

# 过零率：相邻采样符号发生变化的比例
sign = np.signbit(frames)       # True 表示负数
zcr = np.mean(sign[:, 1:] != sign[:, :-1], axis=1)

# …… 用 energy / zcr 做阈值判决，得到起止帧 k0、k1 ……

start = k0 * hop_len
end = L if k1 == frames.shape[0] - 1 else k1 * hop_len + frame_len
note = ctx["y_raw"][start:end]  # 切音只切 y_raw

print(note.shape, sr)
```

> `energy` 和 `zcr` 只是示例；阈值、最短音符、合并间隔由 M2 自己定。

### 6.2 M3：Goertzel

```python
S = calc_goertzel_spectrum(note, ctx["sr"], target_freqs)
```

- `note` 是 `y_raw` 的一维切片；
- `frames` **没有加窗**；如果 M3 需要 Hamming 窗，请在 M3 内部自己加；
- 不要重采样，直接用 `ctx["sr"]`。

### 6.3 M4：匹配滤波

`S` 由 M3 输出并做 L2 归一化；M4 的模板矩阵形状为 `(M, N)`，与 M1 的返回值无关。

### 6.4 M5：GUI

```python
t = np.arange(len(ctx["y_raw"])) / ctx["sr"]
ax.plot(t, ctx["y_raw"])        # 画整段波形
# 画分帧网格/能量曲线时用 ctx["frames"]、ctx["hop_len"]、ctx["sr"]
```

不需要再读文件。

---

## 7. 行为保证（下游可以依赖）

1. **峰值归一化**：非静音时 `np.max(np.abs(y_raw)) == 1.0`（浮点精确）。
2. **帧对齐**：完整帧满足 `frames[k] == y_raw[k*hop : k*hop+frame_len]`。
3. **类型统一**：`y_raw`、`frames` 都是 `float64`；`frames` 是 C 连续、可写数组。
4. **严格 5 键**：`set(ctx.keys()) == {"y_raw", "frames", "sr", "frame_len", "hop_len"}`。
5. **单声道**：双声道/多声道按 `mean(axis=1)` 平均。
6. **不做额外处理**：不重采样、不加窗、不去直流、不做预加重、不做逐帧归一化。
7. **8-bit PCM**：若原始文件是 `uint8`，会先减 128 去掉中点偏置。

---

## 8. 边界情况与错误行为

| 情况 | 行为 |
|---|---|
| 文件不存在 | `FileNotFoundError`（不包装） |
| 不是 wav / 文件损坏 | `scipy` 的解析异常原样上抛 |
| 空音频（0 采样） | `ValueError("音频为空...")` |
| 含 `NaN` / `Inf` | `ValueError` |
| 全零 / 全静音 | 返回全零 `y_raw`，并发 `RuntimeWarning`，不抛异常 |
| 双声道 / 多声道 | 按采样点取平均，下混成单声道 |
| 8-bit PCM（`uint8`） | 减 128 去偏置后再归一化 |
| 24-bit / 32-bit / float wav | 转 `float64` 后按峰值归一化 |
| `frame_ms <= 0` / `hop_ms <= 0` | `ValueError` |
| `hop_ms > frame_ms` | `ValueError` |
| `L < frame_len` | `frames` 补零成 1 帧，`y_raw` 长度不变 |
| `L - frame_len` 不是 `hop_len` 的整数倍 | 丢弃尾部，丢弃量 `< hop_len` |

---

## 9. 下游不要做的事

- ❌ 不要再 `wavfile.read` 一遍——直接用传下来的 `ctx`。
- ❌ 不要对 `y_raw` 再做一次时域归一化——峰值已经是 1。
- ❌ 不要用 `frames` 拼接出音符——帧之间有重叠，会把同一段采样算多次；切音用 `y_raw`。
- ❌ 不要假设 `frames` 已经加窗——本模块不加窗。
- ❌ 不要把 `sr` 写死成 44100——用 `ctx["sr"]`。
- ❌ 不要修改或新增返回字典的键——接口已冻结；确实需要新上下文时，先和 M1 负责人沟通，另开函数或升版本。
- ❌ 不要把 `frames.shape[0]` 当成信号长度——信号长度是 `len(ctx["y_raw"])`。

---

## 10. 参数速查表

### 10.1 默认帧长/帧移（25ms / 10ms）

| `sr` | `frame_len` | `hop_len` |
|---:|---:|---:|
| 8000 | 200 | 80 |
| 16000 | 400 | 160 |
| 22050 | 551 | 221 |
| 32000 | 800 | 320 |
| 44100 | **1103** | **441** |
| 48000 | 1200 | 480 |
| 96000 | 2400 | 960 |

### 10.2 帧数公式

```python
n_frames = 1 + (L - frame_len) // hop_len      # L >= frame_len
n_frames = 1                                   # L <  frame_len（补零）
```

---

## 11. 文件与相关文档

| 文件 | 作用 |
|---|---|
| [src/preprocess.py](src/preprocess.py) | 本模块的**规范实现**，下游统一 import 这个 |
| [src/preprocess_m.py](src/preprocess_m.py) | 手工制作版；函数名、参数、5 个返回键一致，下游不需要用它 |
| [src/demo_preprocess.py](src/demo_preprocess.py) | 生成测试 wav、跑通、4 项自检、画图 |
| [requirements.txt](requirements.txt) | 依赖清单 |
| [docs/音频前期处理脚本计划.md](docs/音频前期处理脚本计划.md) | 接口设计与算法依据 |
| [docs/images/preprocess_demo.png](docs/images/preprocess_demo.png) | demo 预期输出图 |

---

## 12. 接口冻结与变更流程

- 当前接口版本：**v1.0**，5 个键、归一化方式、帧数公式均已冻结。
- 任何改动（键名、键数量、归一化方式、分帧规则）必须：
  1. 先更新 [docs/音频前期处理脚本计划.md](docs/音频前期处理脚本计划.md) 并升版本号；
  2. 再改 `src/preprocess.py`；
  3. 通知 M2/M3/M4/M5。
- 禁止“代码先改、文档后补”，否则下游会拿到不一致的契约。
