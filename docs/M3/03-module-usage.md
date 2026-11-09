# M3 模块使用说明（只讲我负责的 M3）

> **本文只讲 M3：`calc_goertzel_spectrum.py`。**
> M3 的职责一句话：把**一个单音的一维数组 + 采样率**，变成**26 维单位向量**。
>
> 不讲 M1 怎么读 wav、M2 怎么切音、M4 怎么判决；只在第 5 节说清楚“谁给 M3 什么、M3 给谁什么”。
>
> 配套：原理看 [01-tutorial.md](01-tutorial.md)，逐行看 [02-code-walkthrough.md](02-code-walkthrough.md)。

---

## 0. 先划清边界：M3 做什么、不做什么

| M3 做 | M3 不做 |
|---|---|
| 接收一个单音的一维数组 | 不读 wav 文件 |
| 在 26 个标准音基频上算 Goertzel | 不分帧、不切音、不做端点检测 |
| 把 26 个幅度做 L2 归一化 | 不拿模板做判决 |
| 把 26 个标准音堆成 26×26 模板 | 不做 GUI、不画图、不打印 |
| 输出 `S (26,)` 和 `T (26,26)` | 不决定“这是哪个音” |

**打个比方**：M3 像“把一段声音翻译成 26 个刻度上的分数”；选哪个音是 M4 的事，M3 只负责把分数算准。

---

## 1. 文件与依赖

### 1.1 只需要一个文件

```
项目根目录/
└── calc_goertzel_spectrum.py    # M3，你负责的模块
```

> 本仓库就是这种布局。如果组里把 M3 放进 `src/`，把后文所有 `sys.path` 的根目录换成 `src/` 即可。

### 1.2 依赖只有 NumPy

M3 核心**不需要** SciPy、Matplotlib、M1：

```bash
python -m pip install numpy
```

检查：

```bash
python -c "import numpy; print(numpy.__version__)"
```

### 1.3 确认模块能被找到

在项目根目录运行：

```bash
python -c "import sys; sys.path.insert(0,'src'); import calc_goertzel_spectrum as m; print(m.TARGET_FREQS.shape)"
```

期望输出：

```
(26,)
```

如果报 `ModuleNotFoundError`，说明 `sys.path` 或文件位置不对，见第 7 节。

---

## 2. 30 秒自检：先证明 M3 自己是对的

M3 文件末尾有一段 `if __name__ == "__main__":` 自检，直接运行：

```bash
python calc_goertzel_spectrum.py
```

期望输出：

```
shape=(26,)  norm=1.000000
  C4   0.8945
  C5   0.4471
  B3   0.0046
```

这说明三件事：

1. 输出长度是 **26**；
2. 范数是 **1.000000**（L2 归一化生效）；
3. 一个 C4+C5 的测试音，最高分是 C4（`C5 0.4471` 是 C4 的二次谐波，正常）。

> 这一步不需要 M1、M2、M4，也不需要任何 wav。**M3 是纯函数，先单独跑通它。**

---

## 3. 在代码里调用 M3

### 3.1 最小例子：直接给数组

```python
import sys
sys.path.insert(0, ".")          # 模块在项目根目录；在根目录运行

import numpy as np
from calc_goertzel_spectrum import TARGET_FREQS, calc_goertzel_spectrum

sr = 44100
t = np.arange(int(0.6 * sr)) / sr
x = np.sin(2 * np.pi * 261.63 * t)   # 一个 C4 正弦，一维数组

S = calc_goertzel_spectrum(x, sr, TARGET_FREQS)

print(S.shape)                              # (26,)
print(round(float(np.linalg.norm(S)), 6))   # 1.0
```

### 3.2 三个参数

```python
calc_goertzel_spectrum(note_signal, sr, target_freqs)
```

| 参数 | 传什么 | 要求 |
|---|---|---|
| `note_signal` | 一个单音的一维数组 | `ndim == 1`，非空，不能有 NaN/Inf |
| `sr` | 采样率，整数 | `> 0`，必须和信号真实采样率一致 |
| `target_freqs` | 26 个频点 | **必填**，通常传 `TARGET_FREQS`；要求 `0 < f < sr/2` |

**`target_freqs` 为什么必填**：计划书的签名就是三参数；显式传能避免“忘了用哪套频点”。只传两个参数会直接：

```
TypeError: calc_goertzel_spectrum() missing 1 required positional argument: 'target_freqs'
```

### 3.3 返回值怎么读

```python
import numpy as np
from calc_goertzel_spectrum import NOTE_NAMES

top3 = np.argsort(S)[::-1][:3]
for i in top3:
    print(NOTE_NAMES[i], round(float(S[i]), 4))
```

C4 的典型输出：

```
C4 0.8504
C5 0.4252
G5 0.2253
C6 0.2126
```

| 性质 | 值 |
|---|---|
| 形状 | `(26,)` |
| 类型 | `float64` |
| 范围 | 每个数 0–1（幅度非负） |
| 范数 | `‖S‖₂ = 1` |
| 全静音 | 返回全 0（不是 NaN） |

**`S[i]` 不是概率**，是“这个音和第 i 个标准音的像不像”。26 个数里正确音最大，但绝对分数受泛音影响，不能当概率解释。

### 3.4 想换一套频点

```python
# 只算 C4–B4 的 7 个自然音
idx7 = [1, 3, 5, 6, 8, 10, 12]
S7 = calc_goertzel_spectrum(x, sr, TARGET_FREQS[idx7])
print(S7.shape)   # (7,)
```

- 返回长度 = `len(target_freqs)`；
- 项目验收请用 26 个，别用 7 个交差；
- 自定义频点也必须满足 `0 < f < sr/2`。

---

## 4. M3 的另一半：建 26 个模板

M3 不只算待测音的 `S`，还要把 26 个标准音变成 `T`，交给 M4。

### 4.1 函数签名

```python
build_templates(note_signals, sr, target_freqs)
```

| 参数 | 传什么 |
|---|---|
| `note_signals` | 26 个一维数组，**顺序必须和 `NOTE_NAMES` 一致** |
| `sr` | 采样率 |
| `target_freqs` | 26 个频点，通常 `TARGET_FREQS` |

返回：`(26, 26)`，第 i 行是第 i 个标准音的 `S`，每行范数为 1。

### 4.2 例子

```python
import numpy as np
from calc_goertzel_spectrum import NOTE_NAMES, TARGET_FREQS, build_templates

# 26 个一维数组：顺序必须是 B3, C4, C#4, ..., C6
notes = [ ... ]                       # 每个元素是一维 np.ndarray

T = build_templates(notes, 44100, TARGET_FREQS)
print(T.shape)                         # (26, 26)
print(np.linalg.norm(T, axis=1))       # 26 个 1.0
```

> `notes` 从 M1 的 `ctx["y_raw"]` 来（M1 不在本文范围）；26 个数组必须**同一采样率**，否则不能统一传 44100，要逐个 `calc_goertzel_spectrum(..., ctx["sr"], ...)` 再 `np.vstack`。

### 4.3 三个必须遵守的约定

1. **顺序**：`notes[i]` 必须对应 `NOTE_NAMES[i]`；顺序错了，M4 的 `argmax` 会把音名翻译错；
2. **模板和测试分开**：建模板的音不能拿来当测试音（不同 take / 不同录音）；
3. **不能有静音行**：某个标准音是静音，那一行全 0，该音永远识别不出来；建完先看 `norm(T, axis=1)`。

### 4.4 保存给 M4

```python
np.savez(
    "data/templates.npz",
    T=T,
    names=np.array(NOTE_NAMES),
    freqs=TARGET_FREQS,
)
```

M4 这样加载：

```python
d = np.load("data/templates.npz", allow_pickle=False)
T = d["T"]          # (26, 26)
names = d["names"]  # 26 个音名
```

> 到这一步，M3 的交付物就有了：**`S` 的计算函数 + `T` 的生成函数 + `templates.npz`**。

---

## 5. 和上下游的接口（只说到边界，不展开他们的代码）

| 角色 | 给 M3 什么 | M3 给什么 |
|---|---|---|
| M1 | `sr`（`ctx["sr"]`） | — |
| M2 | `note_signal`（一个单音的一维数组） | — |
| M3 | — | `S (26,)` 给 M4；`T (26,26)` 给 M4 |
| M4 | — | 用 `T @ S` 判决，M3 不参与 |

**一句话**：M1 负责“读”，M2 负责“切”，M3 负责“算谱”，M4 负责“选音”。M3 不 import M1/M2/M4，保持纯函数。

> 整条链路的 demo（M1 → M2 → M3）不属于 M3 的交付范围；本仓库不包含该 demo。**验收 M3 时只需要第 2 节和第 6 节。**

---

## 6. M3 自己的验收指标

M3 不负责最终准确率（那是 M4 的 `argmax`），M3 只需要证明“谱算对了、模板建对了”。

### 6.1 单音自检

| 检查 | 期望 |
|---|---|
| `S.shape` | `(26,)` |
| `S.dtype` | `float64` |
| `np.linalg.norm(S)` | `1.0`（误差 < 1e-12） |
| `S >= 0` | 全部为真 |
| 静音输入 | 返回全 0，无 NaN |
| 非法输入 | 抛 `ValueError` |

### 6.2 模板自检

```python
T = build_templates(notes, 44100, TARGET_FREQS)
C = T @ T.T
off = C - np.eye(26)

print(T.shape)                 # (26, 26)
print(np.linalg.norm(T, axis=1).min(), np.linalg.norm(T, axis=1).max())  # 1.0 1.0
print(np.diag(C).min(), np.diag(C).max())   # 1.0 1.0
print(off.max())               # 约 0.4754
```

| 指标 | 期望 | 说明 |
|---|---|---|
| `T.shape` | `(26, 26)` | 26 个模板 |
| 每行范数 | 1.0 | L2 归一化 |
| `C` 对角线 | 1.0 | 自己和自己 |
| 非对角最大 | 约 0.4754 | C4 ↔ C5 八度对 |
| 非对角均值 | 约 0.0283 | 大部分接近 0 |

> 走 M1 读 16-bit wav 时，非对角最大约 **0.4755**，差 1e-4，是量化误差。

### 6.3 7 个自然音子矩阵（和论文表 2 对齐）

```python
idx7 = [1, 3, 5, 6, 8, 10, 12]   # C4, D4, E4, F4, G4, A4, B4
sub = C[np.ix_(idx7, idx7)]
print((sub - np.eye(7)).max())   # 约 0.1014
```

论文表 2 的非对角最大是 0.140；我们合成数据是 0.1014，量级一致。**这是 M3 在报告里最能拿得出手的证据。**

---

## 7. 常见问题（只列 M3 相关）

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: calc_goertzel_spectrum` | 项目根目录没进 `sys.path` | `sys.path.insert(0, ".")`，或在项目根目录运行 |
| `TypeError: missing ... 'target_freqs'` | 只传了两个参数 | 传 `TARGET_FREQS` |
| `ValueError: note_signal must be 1-D` | 传了 `frames` 或 `(L,1)` | 用 `y_raw` 的一维切片 |
| `ValueError: note_signal contains NaN/Inf` | 信号里有 NaN | 检查上游读文件/归一化 |
| `ValueError: target_freqs must satisfy 0 < f < sr/2` | 频点超过 Nyquist | 检查 `sr` 和 `TARGET_FREQS` |
| 返回全 0 | 输入是静音/全 0 | 确认这一段真的有声音 |
| 范数不是 1 | 没走 `calc_goertzel_spectrum`，自己手算漏了归一化 | 用函数返回的 `S` |
| 模板某行范数是 0 | 那个标准音是静音 | 重录/换文件 |
| C4 被认成 C5 | M3 输出的八度对串扰（0.4754） | 单音 `argmax` 仍对；多音用 M4 的 margin/top-K，不是 M3 的 bug |
| `T @ S` 维度报错 | `T` 不是 `(26,26)` 或 `S` 不是 `(26,)` | 打印两个 shape；检查 `target_freqs` 长度 |

---

## 8. 30 秒检查清单

```bash
# 1. 模块能 import，频点 26 个
python -c "import sys; sys.path.insert(0,'src'); import calc_goertzel_spectrum as m; print(m.TARGET_FREQS.shape)"

# 2. M3 自检
python calc_goertzel_spectrum.py

# 3. 模板形状
python -c "import numpy as np; d=np.load('data/templates.npz'); print(d['T'].shape)"
```

期望：

```
(26,)
shape=(26,)  norm=1.000000
(26, 26)
```

---

## 9. 术语小抄（M3 范围内）

| 词 | 通俗解释 |
|---|---|
| 谱向量 `S` | 26 个刻度上的“像不像”分数，长度 26 |
| 模板 `T` | 26 个标准音的 `S` 堆起来，`(26,26)` |
| Goertzel | 只算一个频点的快速算法 |
| L2 归一化 | 把向量长度变成 1，消掉音量影响 |
| 频点 `target_freqs` | 你要算的 26 个频率 |
| 八度对 | C4/C5 这种高八度关系，会互相串 |
| Nyquist | `sr/2`，超过它算出来是假峰 |
| 静音行 | 模板里全 0 的一行，那个音永远识别不出来 |

---

## 10. 一句话总结

**M3 的交付物只有三样：`calc_goertzel_spectrum()` 能算出 `(26,)` 单位向量、`build_templates()` 能堆出 `(26,26)` 模板、自检通过（范数 1、对角线 1、非对角最大约 0.4754）。** 判决、GUI、M1 的读文件都不在你负责的范围内，接口对上即可。
