# M3 代码逐块讲解（新手版）

> **配套**：[01-tutorial.md](01-tutorial.md)（原理、实验、报告）、[03-module-usage.md](03-module-usage.md)（怎么调用）
> **本文只做一件事**：按代码顺序，一块一块解释 `calc_goertzel_spectrum.py` 里每一段在干什么。
> **代码版本**：v2.0（26 音 B3–C6，`target_freqs` 必填）
> **读法**：先看第 0 节地图，然后从“块 1”顺序往下读。每块都有：代码 → 一句话 → 逐行 → 为什么 → 写错会怎样。

---

## 0. 先看地图

### 0.1 模块 `calc_goertzel_spectrum.py` 的 24 个块

| 块 | 内容 | 一句话作用 |
|---:|---|---|
| 1 | 文件头 + docstring | 说明这个文件是干什么的、接口版本 |
| 2 | `import numpy` | 引入数组和数学函数 |
| 3 | `NOTE_NAMES` | 26 个音名，**顺序就是索引** |
| 4 | MIDI 常量 | B3=59、C6=84、A4=440 |
| 5 | `midi_to_freq` | MIDI 号 → 频率 |
| 6 | `note_frequencies` | 生成 26 个基频 |
| 7 | `TARGET_FREQS` | 模块级常量，全文件共用 |
| 8 | `goertzel_power` 定义 | 算**一个**频点的功率 |
| 9 | 类型转换 + 空检查 | 把输入变成 float64 一维数组 |
| 10 | `w0` / `coeff` | 角频率和递推系数 |
| 11 | `s1` / `s2` 初值 | 零状态 |
| 12 | 递推循环 | Goertzel 的核心 |
| 13 | 幅度平方 + 返回 | 从 `s1/s2` 算出 `|X|²` |
| 14 | `calc_goertzel_spectrum` 定义 | 单音 → 26 维谱向量 |
| 15 | 输入 `x` 校验 | 必须是一维、非空、无 NaN |
| 16 | `sr` 校验 | 采样率必须为正 |
| 17 | `target_freqs` 转换 | 频点列表 → float64 数组 |
| 18 | 频点校验 | 必须 1 维、非空、`0 < f < sr/2` |
| 19 | 加窗 | 乘 Hamming 窗，压旁瓣 |
| 20 | 逐频点计算 | 对 26 个频点各调一次 `goertzel_power` |
| 21 | 开方 + 范数 | 幅度谱 + L2 范数 |
| 22 | 静音保护 + 返回 | 全 0 输入返回全 0；否则除以范数 |
| 23 | `build_templates` | 26 个单音 → 26×26 模板矩阵 |
| 24 | `__main__` 自检 | 直接运行文件时做一个小测试 |

### 0.2 先记住这几个变量的形状

| 变量 | 形状 | 含义 |
|---|---|---|
| `x` | `(L,)` | 一个单音的采样点，L 约 26460（0.6 s） |
| `xw` | `(L,)` | 加窗后的信号 |
| `power` | `(26,)` | 26 个频点的功率 `|X|²` |
| `mag` | `(26,)` | 26 个频点的幅度 `|X|` |
| `norm` | 标量 | `mag` 的 L2 范数 |
| `S` | `(26,)` | 归一化后的谱向量，`‖S‖=1` |
| `T` | `(26, 26)` | 26 个模板，每行一个 |
| `C` | `(26, 26)` | 模板相关矩阵 `T @ T.T` |

---

## 1. 模块：`calc_goertzel_spectrum.py`

### 块 1　文件头与模块说明

```python
# -*- coding: utf-8 -*-
"""M3：Goertzel 频谱 + L2 能量归一化（论文对齐版，26 音）。

输入一个单音的一维信号，输出 B3–C6 共 26 个标准音基频上的归一化幅度谱。
模板矩阵 T 的形状为 (26, 26)，第 i 行对应 NOTE_NAMES[i]。
接口版本：v2.0（2026-11-09，按论文表 1 与 M=26 冻结）。
"""
```

**一句话**：告诉读代码的人（以及 M4 的 saku）“这个文件是干什么的、输入输出是什么、版本是多少”。

**逐行**：

| 行 | 意思 |
|---|---|
| `# -*- coding: utf-8 -*-` | 声明源文件用 UTF-8 编码。Python 3 默认就是 UTF-8，这行主要是历史习惯；有中文注释时留着更稳。 |
| `"""..."""` | 模块的 **docstring**（文档字符串），不是普通注释。`import` 之后可以用 `模块.__doc__` 读到它。 |
| `输入一个单音的一维信号...` | 写清楚输入是“一维数组”，不是文件路径。 |
| `输出 B3–C6 共 26 个...` | 写清楚输出长度和音域。 |
| `模板矩阵 T 的形状...` | 写清楚 M4 最关心的两个约定：形状、行顺序。 |
| `接口版本...` | 冻结版本，方便组内对接口。 |

**为什么这样写**：M4 的 saku 打开文件第一眼要知道“给我什么、我拿到什么”。docstring 写在最上面，比在函数里写十行注释都管用。

**删了会怎样**：程序照样跑，但别人（包括两周后的你）要重新猜接口。课程项目验收时，接口说明不清是扣分点。

**新手常见错误**：

- 用 `#` 写一大段，而不是 `"""..."""`；`#` 不会进 `__doc__`。
- docstring 里写“26 个音”但代码实际是 24 个；文档和代码必须同步。

---

### 块 2　引入 NumPy

```python
import numpy as np
```

**一句话**：把 NumPy 引进来，后面用 `np.数组`、`np.cos`、`np.sqrt`。

**逐行**：

- `import numpy`：导入 NumPy 包；
- `as np`：起个短别名。后面写 `np.cos` 比 `numpy.cos` 省事，也是全组统一的写法。

**为什么这样写**：NumPy 是这门课做数组运算的基础；`np` 是社区惯例，别人一看就懂。

**删了会怎样**：第一处 `np.asarray` 就会 `NameError: name 'np' is not defined`。

**新手常见错误**：

- `import numpy` 但后面写 `np.`，忘了 `as np`；
- `from numpy import *`：能用，但会把一堆名字倒进当前命名空间，容易和别的变量撞名，不推荐。

---

### 块 3　26 个音名

```python
NOTE_NAMES = [
    "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4",
    "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5",
    "G5", "G#5", "A5", "A#5", "B5", "C6",
]
```

**一句话**：把 26 个音名按固定顺序存成一个列表；**列表下标就是模型输出的索引**。

**逐行**：

- `NOTE_NAMES = [...]`：定义一个模块级列表；
- 每行 4 个字符串，一共 26 个；
- 第 0 个是 `"B3"`，第 25 个是 `"C6"`；
- 升号 `#` 表示升高半音，例如 `C#4` 是 C4 和 D4 之间的黑键。

**为什么这样写**：M3 返回的 `S` 是一个 26 维向量，M4 用 `argmax` 得到 0–25 的索引；只有靠这个列表，才能把索引翻译回音名。**顺序错了，所有预测都会“张冠李戴”**。

**删了/写错会怎样**：

- 少一个音（比如漏了 B5、C6）→ 列表 24 个，但 `TARGET_FREQS` 26 个，后面 `S` 和 `T` 长度对不上，`T @ S` 直接报错；
- 顺序换成降号（`Db4` 代替 `C#4`）→ 音名和频率表对不上，准确率会莫名其妙地掉。

**新手常见错误**：

- 按“C4 开始”数，漏掉 B3；
- 把 `A#5` 写成 `A5#`（升号写在音名后面，不是八度后面）；
- 用降号混排，导致同一个键有两个名字、索引却只有一个。

---

### 块 4　MIDI 与 A4 常量

```python
MIDI_B3 = 59
MIDI_C6 = 84
A4_HZ = 440.0
```

**一句话**：把“边界 MIDI 号”和“标准音 A4 频率”起个名字，避免在代码里写魔法数字。

**逐行**：

- `MIDI_B3 = 59`：B3 的 MIDI 号是 59；
- `MIDI_C6 = 84`：C6 的 MIDI 号是 84；
- `A4_HZ = 440.0`：A4 = 440 Hz，是十二平均律的基准。

**为什么这样写**：后面 `np.arange(MIDI_B3, MIDI_C6 + 1)` 一眼能看出“从 B3 到 C6”。如果直接写 `np.arange(59, 85)`，别人得猜 59/85 是什么。

**改一个会怎样**：把 `MIDI_C6` 改成 83（B5），音域就少了 C6，`TARGET_FREQS` 变 25 个，模板矩阵也跟着变 25×25。**音域是全局约定，不能只改一处**。

**新手常见错误**：

- 把 MIDI 号记成“C4=60、C5=72”，却把 B3 写成 58；
- 把 `A4_HZ` 写成 441 或 442（有些乐团用 442，但本项目按 440）。

---

### 块 5　MIDI 号转频率

```python
def midi_to_freq(midi, a4=A4_HZ):
    """MIDI 音高 → 频率（Hz），A4 = 440 Hz 对应 MIDI 69。"""
    return a4 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)
```

**一句话**：用十二平均律公式，把 MIDI 号（可以是一个数，也可以是一串数）换成频率。

**逐行**：

- `def midi_to_freq(midi, a4=A4_HZ):`：定义函数，`midi` 是输入，`a4` 有默认值 440；
- `np.asarray(midi, dtype=np.float64)`：把输入统一成 float64 数组。传 `60` 得到 0 维数组，传 `[59, 60]` 得到一维数组；
- `- 69.0`：以 A4（MIDI 69）为基准；
- `/ 12.0`：每 12 个半音一个八度；
- `2.0 ** (...)`：频率按 2 的幂增长；
- `a4 * ...`：最后乘基准频率。

**为什么这样写**：公式只写一遍，标量和数组都能用。后面 `note_frequencies()` 直接传 `np.arange(...)`，不用写循环。

**删了/写错会怎样**：

- 写成 `a4 * 2 ** (midi / 12)`：少了 `-69`，C4 会变成 440×2^(60/12) ≈ 1396 Hz，全错；
- 写成 `(midi - 69) // 12`：整数除法，半音会被吞掉；
- `2 **` 而不是 `2.0 **`：Python 3 里结果仍是 float，问题不大，但统一写 `2.0` 更稳。

**新手常见错误**：

- 把 A4 的 MIDI 号记成 60（那是 C4）；
- 忘了 `np.asarray`，传列表时 `list - 69` 直接报错。

---

### 块 6　生成 26 个基频

```python
def note_frequencies():
    """B3–C6 共 26 个半音的基频，单位 Hz。"""
    return midi_to_freq(np.arange(MIDI_B3, MIDI_C6 + 1))
```

**一句话**：生成 `[B3, C4, …, C6]` 对应的 26 个频率。

**逐行**：

- `np.arange(MIDI_B3, MIDI_C6 + 1)`：`arange` 是**左闭右开**，所以右边要 `+1` 才能包含 84；
- 结果是从 59 到 84 的 26 个整数；
- `midi_to_freq(...)`：块 5 的函数，把这些整数一次换成频率；返回 `(26,)` 的 float64 数组。

**为什么这样写**：用公式生成，不用手抄 26 个频率。手抄容易错，而且改音域时只改两个常量就行。

**删了/写错会怎样**：

- `np.arange(MIDI_B3, MIDI_C6)`：少了 C6，只有 25 个；
- `np.arange(59, 85, 2)`：步长 2，只有 13 个，而且全是隔一个音。

**新手常见错误**：

- 把 `arange` 当成“包含右端”；
- 把 `+1` 加到 `MIDI_C6` 外面写成 `(MIDI_C6 + 1)` 但忘了括号优先级（这里不涉及）。

---

### 块 7　模块级频率常量

```python
TARGET_FREQS = note_frequencies()
```

**一句话**：模块被 import 时算一次，得到 26 个频率，后面所有函数共用。

**逐行**：

- `TARGET_FREQS`：全大写，表示“常量，不要随便改”；
- `note_frequencies()`：块 6 的函数，返回 `(26,)` 数组；
- 这句在 `import` 时执行一次，不是每次调用都算。

**为什么这样写**：26 个频率是全局约定；算一次、大家用同一个数组，能保证模板和测试用的是同一套频点。

**删了/写错会怎样**：`calc_goertzel_spectrum` 的调用方要自己传 `TARGET_FREQS`，没有这个常量就得每次手写 26 个数。

**新手常见错误**：

- 在函数里写 `TARGET_FREQS = ...` 覆盖全局变量；
- 对 `TARGET_FREQS` 做原地修改（`TARGET_FREQS[0] = ...`），导致所有调用都变。

---

### 块 8　`goertzel_power` 的定义

```python
def goertzel_power(x, sr, freq):
    """返回 |X(e^{jω0})|²，x 应已加窗；ω0 = 2π·freq/sr。"""
```

**一句话**：算**一个**频点的功率 `|X|²`，不是整个频谱。

**逐行**：

- `x`：一维信号，**调用前应该已经加窗**；
- `sr`：采样率；
- `freq`：要算的那个频率（Hz）；
- 返回：`|X(e^{jω0})|²`，一个标量；
- docstring 里的 `ω0 = 2π·freq/sr` 是角频率。

**为什么返回功率而不是幅度**：块 20 要对 26 个频点算，块 21 再统一开方一次；每个频点都开方会多算 26 次 `sqrt`。

**删了会怎样**：没有这个函数，`calc_goertzel_spectrum` 里就没法逐频点计算。

**新手常见错误**：

- 以为它返回复数 `X`；实际返回的是实数功率；
- 传进来的 `x` 没加窗，却以为函数会自动加。

---

### 块 9　输入转换与空检查

```python
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("x is empty")
```

**一句话**：把输入统一成 float64，并拦住空数组。

**逐行**：

- `np.asarray(x, dtype=np.float64)`：如果传的是 list、int16 数组、float32 数组，都转成 float64；
- `n = x.size`：采样点个数；
- `if n == 0: raise ValueError(...)`：空数组直接报错，不带病往下算。

**为什么这样写**：后面 `coeff * s1` 是浮点运算；统一类型能避免“整数/float32 混进来”的类型问题。空数组如果放过去，循环一次都不跑，最后返回 0，看起来像“静音”，很难查。

**删了会怎样**：

- 不转 float64：传 `int16` 时结果类型可能不稳定；
- 不查空：空输入静默返回 0，M4 的 `argmax` 可能得到奇怪结果。

**新手常见错误**：

- 用 `x.astype(np.float64)` 而不是 `np.asarray`；如果 `x` 是 list，`list.astype` 不存在；
- 用 `if not x:` 判断空数组，NumPy 会报“truth value of an array is ambiguous”。

---

### 块 10　角频率与递推系数

```python
    w0 = 2.0 * np.pi * float(freq) / float(sr)
    coeff = 2.0 * np.cos(w0)
```

**一句话**：把频率换成角频率 `ω0`，再算递推系数 `2cosω0`。

**逐行**：

- `2.0 * np.pi * float(freq)`：`2πf`；
- `/ float(sr)`：除以采样率，得到“每采样点的弧度”；
- `coeff = 2.0 * np.cos(w0)`：Goertzel 递推里的 `2cosω0`。

**为什么这样写**：Goertzel 的差分方程是 `s[n] = x[n] + 2cosω0·s[n-1] - s[n-2]`，系数只跟 `ω0` 有关，所以只算一次，放在循环外面。

**删了/写错会怎样**：

- `w0 = 2πfreq`（忘了 `/sr`）：频率单位错，算出来的谱完全不对；
- `coeff = np.cos(w0)`（少了 `2`）：递推公式错，结果不是 DFT；
- 把 `coeff` 放进循环：结果对，但慢很多。

**新手常见错误**：

- 把 `freq` 当成“归一化频率”（0–0.5）又除了一次 `sr`；
- 用 `np.cos(2 * np.pi * freq / sr)` 但忘了乘 2。

---

### 块 11　两个状态变量

```python
    s1 = 0.0
    s2 = 0.0
```

**一句话**：初始化 `s[n-1]` 和 `s[n-2]`，对应 `s[-1] = s[-2] = 0`。

**逐行**：

- `s1`：上一时刻的状态 `s[n-1]`；
- `s2`：上上时刻的状态 `s[n-2]`；
- 初值 0：零状态，表示“输入之前没有能量”。

**为什么这样写**：递推需要前两个状态；第一次循环时它们还不存在，所以补 0。

**删了会怎样**：`s1`/`s2` 未定义，循环第一句就 `UnboundLocalError`。

**新手常见错误**：

- 把 `s1`、`s2` 初始化成空数组；它们应该是标量；
- 在循环里重新初始化，导致每个采样点都从头算。

---

### 块 12　Goertzel 递推循环

```python
    for v in x:
        s0 = v + coeff * s1 - s2
        s2 = s1
        s1 = s0
```

**一句话**：对每个采样点，按差分方程更新一次状态。

**逐行**：

- `for v in x:`：依次取每个采样点；`v` 是当前 `x[n]`；
- `s0 = v + coeff * s1 - s2`：算当前状态 `s[n]`；
- `s2 = s1`：把旧的 `s[n-1]` 挪到 `s[n-2]`；
- `s1 = s0`：把新算的 `s[n]` 存为 `s[n-1]`，给下一轮用。

**用 3 个采样点走一遍**（假设 `coeff = 1`，`x = [1, 1, 1]`）：

| 轮次 | `v` | 旧 `s1` | 旧 `s2` | `s0` | 新 `s1` | 新 `s2` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| 2 | 1 | 1 | 0 | 2 | 2 | 1 |
| 3 | 1 | 2 | 1 | 2 | 2 | 2 |

循环结束后 `s1 = 2`、`s2 = 2`，正好是 `s[N-1]` 和 `s[N-2]`。

**为什么这样写**：这是 Goertzel 的核心。`s2 = s1` 必须在 `s1 = s0` **之前**，否则 `s2` 会拿到新值。

**删了/写错会怎样**：

- 顺序写成 `s1 = s0; s2 = s1`：`s2` 会等于新的 `s0`，递推错；
- 写成 `s0 = v + coeff * s0 - s1`：状态混用，结果不对；
- 把 `for v in x` 写成 `for i in range(n)` 但忘了 `x[i]`：`v` 是下标，类型错。

**新手常见错误**：

- 以为 `s1`、`s2` 是数组；它们是标量；
- 在循环里打印调试时改了 `s1/s2`，导致结果变。

---

### 块 13　幅度平方与返回

```python
    p = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return float(max(p, 0.0))
```

**一句话**：从最后两个状态算出 `|X|²`，并防止浮点误差产生负数。

**逐行**：

- `s1 * s1`：`s[N-1]²`；
- `s2 * s2`：`s[N-2]²`；
- `coeff * s1 * s2`：因为 `coeff = 2cosω0`，所以这一项就是 `2cosω0·s1·s2`；
- `max(p, 0.0)`：浮点误差可能让理论上的 0 变成 `-1e-12`，开方会 `nan`；
- `float(...)`：返回 Python 标量，不返回 `np.float64`，打印更干净。

**用块 12 的例子**：`s1=2, s2=2, coeff=1` → `4 + 4 - 4 = 4`，所以 `|X|² = 4`。

**为什么不开方**：块 21 会统一 `sqrt`；这里只返回功率。

**删了/写错会怎样**：

- 写成 `s1*s1 + s2*s2 - 2*s1*s2`：把 `coeff` 写死成 2，只有 `ω0=0` 时才对；
- 不加 `max`：某些频点可能 `sqrt(负数)` → `nan`；
- 忘了 `float`：返回值是 `np.float64`，不影响计算，但打印时会出现 `np.float64(...)`。

**新手常见错误**：

- 把 `coeff * s1 * s2` 写成 `coeff * s1 * s2 * 2`（重复乘 2）；
- 以为 `p` 是幅度，后面又开方一次（块 21 已经开方）。

---

### 块 14　`calc_goertzel_spectrum` 的定义

```python
def calc_goertzel_spectrum(note_signal, sr, target_freqs):
    """单音（一维）→ 26 维 L2 归一化幅度谱（一维，范数 1）。"""
```

**一句话**：M3 的主函数，输入一个单音和采样率，输出 26 维单位向量。

**逐行**：

- `note_signal`：一维数组，M2 切出来的一个音；
- `sr`：采样率，必须用 `ctx["sr"]`，不能写死；
- `target_freqs`：26 个频点，**必填**，通常传 `TARGET_FREQS`；
- 返回：`(26,)` 的 float64 数组，范数为 1。

**为什么 `target_freqs` 必填**：计划书的签名就是三参数；显式传能让调用方清楚“用的哪套频点”，也方便你做实验（换 24 音、7 音）。

**删了会怎样**：M4 没有谱向量可用，整个识别链断掉。

**新手常见错误**：

- 调用时只传两个参数：现在会直接 `TypeError`，这是故意的；
- 把 `target_freqs` 当成可选，自己加默认值 `None`，又回到“接口不严格”的问题。

---

### 块 15　输入 `x` 的校验

```python
    #---校验---
    x = np.asarray(note_signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"note_signal must be 1-D, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("note_signal is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("note_signal contains NaN/Inf")
```

**一句话**：保证 `x` 是“一维、非空、没有 NaN/Inf”的 float64 数组。

**逐行**：

- `np.asarray(..., dtype=np.float64)`：统一类型；
- `x.ndim != 1`：只接受一维；传 `(L, 1)` 或 `frames`（二维）会报错；
- `x.size == 0`：空数组报错；
- `np.all(np.isfinite(x))`：NaN/Inf 会传染整条谱，必须拦住。

**为什么这样写**：M2 传进来的应该是 `y_raw` 的一维切片；但联调时经常有人把 `frames`（二维）或整段音频传进来。早报错，比在 `argmax` 那里报一个看不懂的错好。

**删了会怎样**：

- 二维输入：`for v in x` 里 `v` 是数组，`s0` 变成数组，最后 `p` 形状爆炸；
- NaN：`argmax` 可能返回 0，看起来“全识别成 B3”；
- 空数组：返回全 0，M4 以为静音。

**新手常见错误**：

- 用 `if not x.any()` 判空/静音；`any()` 对全 0 返回 False，会误判；
- 只检查 `len(x)`，不检查 `ndim`，二维数组也能过。

---

### 块 16　`sr` 校验

```python
    if float(sr) <= 0:
        raise ValueError("sr must be positive")
```

**一句话**：采样率必须是正数。

**逐行**：

- `float(sr)`：允许传 `int` 或 `np.int64`；
- `<= 0`：0 或负数都不合法；
- 报错信息直接说 `sr must be positive`。

**为什么这样写**：块 10 要除以 `sr`；`sr=0` 会得到 `inf`，`cos(inf)` 是 `nan`，整条谱报废。

**删了会怎样**：`sr=0` 时 `w0=inf`，后面全是 `nan`，很难定位。

**新手常见错误**：

- 用 `if sr == 0`，漏掉负数；
- 把 `sr` 写成字符串 `"44100"`，`float(sr)` 能转，但最好一开始就传数字。

---

### 块 17　`target_freqs` 转换

```python
    freqs = np.asarray(target_freqs, dtype=np.float64)
```

**一句话**：把频点列表统一成 float64 数组。

**逐行**：

- `target_freqs`：可以是 list、tuple、NumPy 数组；
- `np.asarray(..., dtype=np.float64)`：统一类型；
- 结果 `freqs` 后面用于校验和循环。

**为什么这样写**：`np.any(freqs <= 0)`、`freqs.ndim` 这些操作要求它是数组；list 没有 `.ndim`。

**删了会怎样**：如果传 list，块 18 的 `freqs.ndim` 直接 `AttributeError`。

**新手常见错误**：

- 传成二维 `[[261.63], [277.18]]`；块 18 会报“must be 1-D”；
- 传成空列表 `[]`；块 18 会报“non-empty”。

---

### 块 18　频点校验

```python
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("target_freqs must be a non-empty 1-D array")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("target_freqs contains NaN/Inf")
    if np.any(freqs <= 0) or np.any(freqs >= float(sr) / 2.0):
        raise ValueError("target_freqs must satisfy 0 < f < sr/2")
```

**一句话**：频点必须是一维、非空、有限，并且落在 `(0, sr/2)` 内。

**逐行**：

- `ndim != 1 or size == 0`：形状检查；
- `isfinite`：不能有 NaN/Inf；
- `freqs <= 0`：不能是 0 或负数；
- `freqs >= sr/2`：不能超过 Nyquist 频率。

**为什么这样写**：超过 `sr/2` 的频率没有物理意义，算出来是混叠的假峰；`freq=0` 是直流，不在我们的音高网格里。

**删了会怎样**：

- 传 22050（`sr/2`）：正好在 Nyquist 边界，Goertzel 的极点落在 `-1`，结果不可靠；
- 传 30000：超过 Nyquist，得到的是混叠频率的谱，识别必错。

**新手常见错误**：

- 用 `>` 而不是 `>=`，把 Nyquist 放进去；
- 把 `sr/2` 写成 `sr//2`，整数除法在奇数采样率时会差 0.5。

---

### 块 19　加窗

```python
    #---加窗---
    xw = x * np.hamming(x.size)
```

**一句话**：给信号乘一个 Hamming 窗，减少频谱泄漏。

**逐行**：

- `np.hamming(x.size)`：生成长度等于 `x` 的 Hamming 窗，值在 0–1 之间；
- `x * ...`：逐点相乘，形状不变；
- 结果 `xw` 是加窗后的信号，后面送进 `goertzel_power`。

**为什么这样写**：我们只截了 0.6 秒，相当于乘了矩形窗；矩形窗旁瓣高，强频率会“漏”到别的频点。Hamming 把旁瓣压到 −43 dB。

**删了会怎样**：不加窗也能跑，但非对角串扰会大一点（教程实测：非对角均值 0.0343 → 0.0283）。

**新手常见错误**：

- 对每个频点分别加窗（错；应该对整段 `x` 加一次）；
- M1/M2 已经加过窗，这里又加一次（重复加窗）；
- 加窗后做“窗增益补偿”（不需要；L2 归一化会消掉常数增益）。

---

### 块 20　逐频点计算功率

```python
    #---逐频点 Goertzel---
    power = np.array([goertzel_power(xw, float(sr), f) for f in freqs])
```

**一句话**：对 26 个频点各调一次 `goertzel_power`，把结果装成 `(26,)` 数组。

**逐行**：

- `[... for f in freqs]`：列表推导，遍历每个频点；
- `goertzel_power(xw, float(sr), f)`：块 8–13 的函数，返回该频点的 `|X|²`；
- `np.array(...)`：把 Python 列表转成 NumPy 数组，形状 `(26,)`。

**为什么这样写**：26 个频点相互独立，循环最直观；每个频点只占 O(1) 额外内存。

**删了会怎样**：没有 `power`，块 21 没法开方。

**新手常见错误**：

- 把 `xw` 写成 `x`（忘了加窗）；
- 把 `float(sr)` 写成 `sr`（一般也行，但统一 float 更稳）；
- 把 `freqs` 写成 `TARGET_FREQS`（能跑，但失去“显式传参”的灵活性）。

---

### 块 21　开方与 L2 范数

```python
    #---L2 归一化---
    mag = np.sqrt(power)
    norm = float(np.linalg.norm(mag))
```

**一句话**：功率开方得到幅度谱，再算它的 L2 范数。

**逐行**：

- `np.sqrt(power)`：`|X|² → |X|`，形状仍 `(26,)`；
- `np.linalg.norm(mag)`：`sqrt(sum(mag²))`，得到一个标量；
- `float(...)`：转成 Python 标量。

**为什么这样写**：论文的归一化是在“离散谱线”上做 L2；先开方再归一化，和“功率归一化”不是一回事（差一个平方）。

**删了会怎样**：不归一化，`T @ S` 的分数会随音量变，阈值失效（教程实测最大点积约 2349.5）。

**新手常见错误**：

- 先归一化功率再开方：`sqrt(power / norm_power)` 和 `sqrt(power) / norm_mag` 在数学上其实等价（因为 `norm_power = norm_mag²`），但代码里别混；
- 用 `np.sum(mag)` 代替 `np.linalg.norm`：L1 和 L2 不一样，会破坏 Cauchy–Schwarz 上界。

---

### 块 22　静音保护与返回

```python
    if norm == 0.0:
        return np.zeros_like(mag)
    return mag / norm
```

**一句话**：全静音返回全 0；否则返回单位向量。

**逐行**：

- `norm == 0.0`：所有频点都是 0，说明输入是静音/全 0；
- `np.zeros_like(mag)`：返回同样形状的全 0 数组；
- `mag / norm`：逐点除以范数，得到 `‖S‖ = 1`。

**为什么这样写**：M2 可能切出一段近乎静音的音符；返回全 0 让 M4 用 `best=0` 拦掉，而不是让程序崩（除以 0 会得到 NaN）。

**删了会怎样**：静音时 `mag/norm` 是 `0/0 = nan`，M4 的 `argmax` 行为不可预测。

**新手常见错误**：

- 用 `if norm < 1e-12` 也行，但 `== 0.0` 对本例足够；
- 返回 `None`，M4 还要多写一个判断；返回全 0 更统一。

---

### 块 23　`build_templates`

```python
#---模板矩阵---
def build_templates(note_signals, sr, target_freqs):
    """26 个标准音 → (26, 26) 模板矩阵，第 i 行对应 NOTE_NAMES[i]。"""
    rows = [calc_goertzel_spectrum(s, sr, target_freqs) for s in note_signals]
    return np.vstack(rows)
```

**一句话**：把 26 个标准音的谱向量按行堆成 `(26, 26)` 矩阵，交给 M4。

**逐行**：

- `note_signals`：26 个一维数组，顺序必须和 `NOTE_NAMES` 一致；
- `rows = [...]`：对每个标准音调一次 `calc_goertzel_spectrum`；
- `np.vstack(rows)`：把 26 个 `(26,)` 行堆成 `(26, 26)`。

**为什么这样写**：M4 的 `scores = T @ S` 要求 `T` 是二维、每行一个模板；`vstack` 比循环赋值简单。

**删了会怎样**：M4 要自己循环 26 次调用，代码重复，容易把顺序搞错。

**新手常见错误**：

- 把 `note_signals` 按字母序传，而不是 `NOTE_NAMES` 序；
- 某个标准音是静音，那一行全 0，该音永远识别不出来；建模板前要检查行范数；
- 传空列表：`np.vstack([])` 会报错。

---

### 块 24　`__main__` 自检

```python
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
```

**一句话**：直接运行这个文件时，造一个 C4+C5 的测试音，打印谱向量和 top-3。

**逐行**：

- `if __name__ == "__main__":`：只有 `python calc_goertzel_spectrum.py` 时才执行；被 import 时不执行；
- `fs = 44100`：测试采样率；
- `t = np.arange(int(0.5*fs)) / fs`：0.5 秒的时间轴；
- `x = 0.8*sin(2π·261.63·t) + 0.4*sin(2π·523.25·t)`：C4 基频 + C5 二次谐波；
- `S = calc_goertzel_spectrum(x, fs, TARGET_FREQS)`：走完整流程；
- `np.argsort(S)[::-1][:3]`：从小到大排序后倒序，取前 3 个下标；
- 打印形状、范数、top-3 音名和分数。

**为什么这样写**：不依赖外部 wav，别人拿到文件就能跑；输出能快速判断“C4 是否排在第一位”。

**删了会怎样**：模块仍能被 import，但少了“拿到就能跑”的自检，联调时少了一个快速验证手段。

**新手常见错误**：

- 把测试代码写在模块顶层（没有 `if __name__`），一 import 就执行；
- 用 `np.argsort(S)[:3]`（取到的是最小的 3 个）；
- 忘了 `int(...)`，`np.arange` 需要整数长度。

---

## 2. 把代码和数学对上

| 代码 | 数学 | 说明 |
|---|---|---|
| `w0 = 2πf/sr` | `ω0` | 角频率 |
| `coeff = 2cos(w0)` | `2cosω0` | 递推系数 |
| `s0 = v + coeff*s1 - s2` | `s[n] = x[n] + 2cosω0·s[n-1] - s[n-2]` | 差分方程 |
| `s1` | `s[N-1]` | 最后状态 |
| `s2` | `s[N-2]` | 倒数第二状态 |
| `p = s1²+s2²-coeff·s1·s2` | `|X(ω0)|²` | 功率 |
| `mag = sqrt(power)` | `|X(ω0)|` | 幅度 |
| `mag / norm` | `S̄ = S/‖S‖₂` | 单位向量 |
| `T @ S` | `⟨S̄ᵢ, S̄ⱼ⟩` | 相关/点积 |

---

## 3. 新手调试顺序

1. **先跑模块自检**：`python calc_goertzel_spectrum.py`，应看到 `shape=(26,) norm=1.000000`；
2. **再跑 5.3 的三个测试**：整数 bin、直接 DFT、随机信号；
3. **再拿一个自己的数组调用**：`calc_goertzel_spectrum(x, sr, TARGET_FREQS)`，确认 `(26,)`、范数 1；
4. **再建模板**：26 个信号 `build_templates`，确认 `(26,26)`、每行范数 1；
5. **最后接 M4**：`T @ S` 看 `argmax`（属于 M4 的范围，可选）。

---

## 4. 常见报错 → 对应块

| 报错 | 最可能出问题的块 | 怎么查 |
|---|---|---|
| `NameError: np is not defined` | 块 2 | 是否写了 `import numpy as np` |
| `TypeError: missing 1 required positional argument: 'target_freqs'` | 块 14 | 调用时是否传了 `TARGET_FREQS` |
| `ValueError: note_signal must be 1-D` | 块 15 | 是否把 `frames` 或 `(L,1)` 传进来 |
| `ValueError: target_freqs must satisfy 0 < f < sr/2` | 块 18 | 频点是否超过 Nyquist |
| `nan` / 全 0 | 块 22 | 输入是否静音、`norm` 是否为 0 |
| M4 的准确率很低但 M3 不报错 | 块 3 / 块 23 | `NOTE_NAMES` 顺序、模板顺序是否一致 |
| `T @ S` 维度错 | 块 23 | `T` 是否 `(26,26)`，`S` 是否 `(26,)` |

---

## 最后一句

**不要一上来背代码，先记住每块的“输入 → 输出”**：块 12 把 `x` 变成 `s1/s2`，块 13 把 `s1/s2` 变成 `|X|²`，块 20 把 26 个 `|X|²` 装成 `power`，块 21–22 把 `power` 变成单位向量 `S`。把这条链在纸上画一遍，代码就真正是你的了。
