# M3 文档索引

> **模块**：M3 音频频域处理（Goertzel + L2 能量归一化）
> **负责人**：天启
> **代码**：仓库根目录的 `calc_goertzel_spectrum.py`
> **版本**：v2.0，26 个音（B3–C6），`N = M = 26`

## 先读哪一份

| 你的目的 | 文档 |
|---|---|
| 从头理解原理、做实验、写报告 | [01-tutorial.md](01-tutorial.md) |
| 按代码顺序一块一块看 | [02-code-walkthrough.md](02-code-walkthrough.md) |
| 只想调用 M3、建模板、查报错 | [03-module-usage.md](03-module-usage.md) |

## 三份文档的分工

### 01-tutorial.md —— 总教程

- 论文表 1/2/3/4 的对应关系；
- Goertzel 推导、L2 归一化、匹配滤波；
- 26 音网格、代码实现、自检实验；
- 报告 3.1 节写法、答辩 Q&A。

### 02-code-walkthrough.md —— 逐块讲解

- 模块 24 个块（只讲 M3，不含最终 demo 源码）；
- 每块：代码 → 一句话 → 逐行 → 为什么 → 写错会怎样；
- 适合第一次读代码、准备讲给别人听。

### 03-module-usage.md —— 使用说明（只讲 M3）

- M3 的边界：输入数组，输出 `S (26,)`；
- 怎么调用 `calc_goertzel_spectrum()`；
- 怎么用 `build_templates()` 建 `T (26,26)`；
- M3 自己的验收指标与常见报错；
- **不含** M1 读 wav、M2 切音、M4 判决的实现。

## 一句话

**M3 的交付物：`calc_goertzel_spectrum()` 能算 `(26,)` 单位向量，`build_templates()` 能堆 `(26,26)` 模板，自检通过。**

## 相关文件

| 文件 | 作用 |
|---|---|
| `../../calc_goertzel_spectrum.py` | M3 模块 |
| `../../tests/test_m3.py` | M3 单元自检 |
| `../../requirements.txt` | 依赖（只有 NumPy） |
