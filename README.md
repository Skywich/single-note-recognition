# 单音识别项目 · M3 Goertzel 频谱与 L2 归一化

> 《信号与系统（1）》课程项目 · 基于频域匹配滤波的钢琴音阶识别
> 本仓库是 5 人小组仓库；本页对应 **天启** 负责的 **M3 音频频域处理**。
> 当前分支：`lq`　代码版本：`v2.0`（26 个音，B3–C6，`N = M = 26`）

## 这个模块做什么

**输入**：一个单音的一维数组 + 采样率 + 26 个目标频点  
**输出**：26 维 L2 归一化谱向量 `S`，以及 26×26 模板矩阵 `T`

```
单音 x (L,) ──► M3 Goertzel ──► S (26,) ──► M4 点积判决
26 个标准音 ──► M3 ──► T (26,26) ──► templates.npz
```

M3 只负责“算谱”和“建模板”，不读 wav、不切音、不做判决。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 1. 模块自检
python calc_goertzel_spectrum.py

# 2. 单元测试
python -m unittest discover -s tests -v
```

期望看到：

```
shape=(26,)  norm=1.000000
  C4   0.8945
  C5   0.4471
  B3   0.0046

Ran 5 tests in 0.0s
OK
```

## M3 接口

```python
from calc_goertzel_spectrum import TARGET_FREQS, calc_goertzel_spectrum

S = calc_goertzel_spectrum(note_signal, sr, TARGET_FREQS)
# S.shape == (26,), np.linalg.norm(S) == 1.0
```

| 函数 | 作用 | 返回 |
|---|---|---|
| `calc_goertzel_spectrum(note_signal, sr, target_freqs)` | 单音 → 谱向量 | `(26,) float64`，范数 1 |
| `build_templates(note_signals, sr, target_freqs)` | 26 个标准音 → 模板矩阵 | `(26, 26)`，每行范数 1 |
| `goertzel_power(x, sr, freq)` | 单频点功率（内部用） | `float` |

`target_freqs` 是**必填参数**，通常传 `TARGET_FREQS`。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/M3/README.md](docs/M3/README.md) | M3 文档索引与阅读顺序 |
| [docs/M3/01-tutorial.md](docs/M3/01-tutorial.md) | 原理、论文表 1/2/3/4、实验、报告与答辩 |
| [docs/M3/02-code-walkthrough.md](docs/M3/02-code-walkthrough.md) | 按代码顺序逐块讲解（模块 24 块） |
| [docs/M3/03-module-usage.md](docs/M3/03-module-usage.md) | 只讲 M3 的使用：调用、建模板、自检、报错 |

项目其他文档：

- [项目复现计划](docs/信号与系统课程项目复现计划v1.0.3.md)
- [分工确认](docs/分工确认.md)
- [工作方法](docs/work_method.md)

## 目录结构

```
.
├── README.md                     # 本页
├── calc_goertzel_spectrum.py     # M3 模块（本仓库放在根目录）
├── preprocess_m.py               # M1（本地文件，组内提供，本分支不提交）
├── requirements.txt              # 依赖：只有 NumPy
├── tests/
│   └── test_m3.py                # M3 单元自检
└── docs/
    ├── M3/                       # M3 文档
    └── ...                       # 小组其他文档
```

## 上传到 Git

确认自己有当前分支的写权限后：

```bash
git status
git add README.md docs/M3 tests requirements.txt .gitignore
git commit -m "docs(M3): 整理 M3 文档、使用说明与自检"
git push        # 当前 lq 跟踪 origin/calc_goertzel_spectrum
```

> 如果使用 SSH 被拒绝，先看 `ssh -T git@github.com` 当前是哪个账号；本仓库的 remote 是 `Skywich/single-note-recognition`，需要有该仓库的写权限。

## 版本与范围

- **v2.0（2026-11-09）**：按论文表 1 改为 26 个音（B3–C6），正文 M=26；
- 代码接口：`calc_goertzel_spectrum(note_signal, sr, target_freqs)`；
- 论文没写窗函数；本实现用 Hamming，属于实现选择，报告里注明；
- 合成数据指标：`T (26,26)`、对角线 1.0、非对角最大约 0.4754、7×7 子矩阵约 0.1014。
