# Stable packing of irregular stones in 2D masonry wall construction
![fft](./figs/build_process.gif)

![rubble_wall](./figs/rubble_wall.gif)


# News
We have another implementation of the algorithm online! The [new implementation](https://github.com/qianqing-wanggg/StablePacking2D/tree/master) is easy to read with a few lines of code!

# Installation

- create virtual environment in the current directory with `python -m venv env` and `source env/bin/activate`.
- run `pip install -e .` in this directory for an editable install

# Structure

The code is structured into two parts:

1. The first part in _Stonepacker2D_ contains the functionalities for manipulating stones.
2. The second part in _Tests_ contains dataset and the main function to run the examples of stone packing task.

## 頂層項目

| 項目 | 說明 |
| --- | --- |
| `Stonepacker2D/` | 堆石演算法函式庫；見該處 README |
| `Tests/` | 論文範例案例與資料；見該處 README |
| `research/` | 本 fork 的研究層：追蹤濾器順序建牆、離線分析、ReAct 控制器；見該處 README |
| `figs/` | 本 readme 用的動畫 |
| `pyproject.toml`、`setup.py` | 套件與相依設定、ruff／black／pytest 設定 |
| `LICENSE` | MIT 授權 |

# Examples

To run an example, go to the directory of the example and _python run.py_. Results (images of the constructed wall, sequence etc...) will be stored in the result subfolder.

| Name               | Corresponding figures in paper         | Description                                                  |
| ------------------ | -------------------------------------- | ------------------------------------------------------------ |
| Almeida            | Figure 18, Figure 21, Figure B.30-B.32 | real stone shapes from irregular to regular                  |
| Bricks             | Figure B.34                            | rectangular shapes of fixed sizes                            |
| FFT-stones         | Figures 26-27                          | Stones shapes generated from a Fourier-based shape generator |
| Rubble stone walls | Figures B.33                           | 92 real stone shapes                                         |
| Thangavelu         | Figures 22-25                          | polygonal stones of different noise levels                   |
