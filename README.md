# 信号与系统课程项目：频域匹配滤波钢琴音识别

## 环境（Windows）

本项目只使用 Windows 原生 Python 环境，不依赖 WSL/Linux：

- Windows 10/11 x64
- Python 3.13（项目内 `.venv` 由 `D:\Miniforge3\python.exe` 创建）
- numpy（见 `requirements.txt`）

项目内已经带好 Windows 虚拟环境：

```bat
.venv\Scripts\python.exe --version
```

如果 `.venv` 丢失，可在 Windows CMD / PowerShell 中重建：

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行 Demo

```bat
.venv\Scripts\python.exe demo.py
.venv\Scripts\python.exe demo.py --save-audio demo_audio
.venv\Scripts\python.exe demo.py --wav my_phrase.wav
.venv\Scripts\python.exe demo.py --wav my.wav --templates data\templates
```

也可以激活环境后运行：

```bat
.venv\Scripts\activate.bat
python demo.py
```

## 最终成品框架

```bat
.venv\Scripts\python.exe frame.py --check
.venv\Scripts\python.exe frame.py --templates data\templates --audio data\test.wav
.venv\Scripts\python.exe frame.py --gui
```

`frame.py` 中五个板块均为占位，填完对应函数后即可联调。

## 文件

- `demo.py`：完整可运行的单音 + 乐句识别 Demo
- `frame.py`：五人小组最终成品框架（各板块 `NotImplementedError` 占位）
- `requirements.txt`：Windows 环境依赖
- `.venv/`：Windows 虚拟环境，不要提交到 git
