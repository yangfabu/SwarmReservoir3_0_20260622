# SwarmReservoir 项目架构文档

> 版本: v2.0 | 最后更新: 2026-06-21

---

## 一、项目概述

本项目为**群体储层计算（Swarm Reservoir Computing）**研究框架。实验流程分为四个阶段：

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ 阶段1         │      │ 阶段2         │      │ 阶段3         │      │ 阶段4         │
│ 输入信号生成   │ ───→ │ 硬件实验采集   │ ───→ │ 图像特征提取   │ ───→ │ 储层基准测试   │
│              │      │              │      │              │      │              │
│ 产物:         │      │ 产物:         │      │ 产物:         │      │ 产物:         │
│ current_     │      │ images/      │      │ features.csv │      │ mc_results   │
│ sequence.csv │      │ summary.csv  │      │ particles.csv│      │ figures/     │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
```

**核心原则**：每个阶段只通过文件系统交换数据，阶段之间不直接 import。

---

## 二、目录结构

```
SwarmReservoir/
│
├── config/                          # [集中配置] 所有可调参数
│   ├── global.yaml                  #   全局：路径、硬件端口、随机种子
│   ├── stage1_generate.yaml         #   阶段1：电流范围、步数、精度、信号模式
│   ├── stage2_experiment.yaml       #   阶段2：相机参数、实验时序
│   ├── stage3_extract.yaml          #   阶段3：蒙版、圆检测、熵计算参数
│   └── stage4_benchmark.yaml        #   阶段4：MC延迟范围、PCA参数、启用的benchmark列表
│
├── src/                             # [核心源码] 按阶段分包，每包内部再按功能分层
│   │
│   ├── common/                      # 跨阶段共享工具
│   │   ├── config.py                #   配置加载器（单例，加载全部yaml）
│   │   ├── io_utils.py              #   CSV读写、图像批量IO、文件查找
│   │   ├── logging_utils.py         #   统一日志格式（毫秒级时间戳）
│   │   └── constants.py             #   物理常数（粒子直径、培养皿尺寸等）
│   │
│   ├── stage1_generate/             # 阶段1：输入信号生成
│   │   ├── pipeline.py              #   阶段1主管线（加载config → 生成 → 导出CSV）
│   │   ├── signal_generator.py      #   信号生成调度器（根据config选择策略）
│   │   ├── strategies/              #   [可扩展] 信号生成策略
│   │   │   ├── base.py              #     抽象基类 BaseSignalStrategy
│   │   │   ├── uniform_random.py    #     均匀随机策略（当前使用）
│   │   │   ├── sinusoidal.py        #     正弦波策略（预留）
│   │   │   └── binary_sequence.py   #     二进制序列策略（预留）
│   │   └── quantization.py          #   硬件精度量化（截断/四舍五入）
│   │
│   ├── stage2_experiment/           # 阶段2：硬件实验与数据采集
│   │   ├── pipeline.py              #   阶段2主管线
│   │   ├── experiment.py            #   实验核心逻辑（电流序列 → 图像采集循环）
│   │   ├── hardware/                #   硬件驱动层
│   │   │   ├── camera.py            #     海康工业相机操作封装
│   │   │   └── power_supply.py      #     程控电源串口通信
│   │   └── ui.py                    #   PyQt5相机控制界面（仅在需要手动控制时使用）
│   │
│   ├── stage3_extract/              # 阶段3：图像特征提取
│   │   ├── pipeline.py              #   阶段3主管线（编排预处理→检测→特征计算）
│   │   ├── preprocessing/           #   图像预处理
│   │   │   ├── mask.py              #     圆形蒙版应用
│   │   │   └── find_center.py       #     培养皿圆心自动/手动定位
│   │   ├── detection/               #   粒子检测
│   │   │   └── circle_detector.py   #     Hough圆检测
│   │   ├── features/                #   [可扩展] 特征计算
│   │   │   ├── base.py              #     抽象基类 BaseFeature
│   │   │   ├── entropy.py           #     邻居距离熵（Voronoi / KDTree）
│   │   │   ├── system_radius.py     #     归一化系统半径
│   │   │   └── neighbor_spacing.py  #     归一化邻居间距
│   │   └── verification/            #   参数调优验证工具
│   │       ├── single_image_viewer.py  #   单张图逐步调参
│   │       └── grid_search.py       #   参数网格搜索对比
│   │
│   ├── stage4_benchmark/            # 阶段4：储层计算基准测试
│   │   ├── pipeline.py              #   阶段4主管线（预处理→执行所有启用的benchmark）
│   │   ├── preprocessing.py         #   特征平铺(flatten)、标准化、PCA降维
│   │   ├── benchmarks/              #   [可扩展] 各种benchmark
│   │   │   ├── base.py              #     抽象基类 BaseBenchmark
│   │   │   └── memory_capacity.py   #     记忆容量 MC（线性回归 R² 曲线）
│   │   └── evaluation.py            #   结果汇总、对比绘图、报告生成
│   │
│   └── __init__.py
│
├── scripts/                         # [运行入口] 薄层脚本，只做参数拼装和调用pipeline
│   ├── run_stage1.py                #   运行: python scripts/run_stage1.py
│   ├── run_stage2.py                #   运行: python scripts/run_stage2.py
│   ├── run_stage3.py                #   运行: python scripts/run_stage3.py
│   ├── run_stage4.py                #   运行: python scripts/run_stage4.py
│   ├── run_all.py                   #   一键运行全管线（需确认硬件就绪）
│   ├── verify_stage3.py             #   阶段3参数调优：python scripts/verify_stage3.py --image xxx.jpg
│   └── test_stage3_step.py          #   阶段3单步测试：只跑10张图验证逻辑
│
├── data/                            # [数据产物] 所有中间和最终数据（建议 gitignore）
│   ├── stage1_output/               #   阶段1产物：current_sequence.csv, mc_metadata_log.csv
│   ├── stage2_output/               #   阶段2产物：原始图像 + experiment_summary.csv + experiment.log
│   └── stage3_output/               #   阶段3产物：particles.csv, features.csv
│
├── output/                          # [最终输出] 图表、模型、报告（建议 gitignore）
│   ├── figures/                     #   阶段4生成的图表
│   ├── models/                      #   训练的模型文件（如有）
│   └── reports/                     #   批量评估报告
│
├── notebooks/                       # [探索分析] Jupyter notebooks
│   ├── 01_signal_analysis.ipynb     #   分析生成信号序列的统计特性
│   ├── 02_experiment_review.ipynb   #   实验日志回顾、逐周期图像抽查
│   ├── 03_feature_exploration.ipynb #   特征分布探索、异常帧检测
│   └── 04_mc_analysis.ipynb         #   MC曲线深度分析、多实验对比
│
├── tests/                           # [单元测试]
│   ├── test_stage1/
│   ├── test_stage3/
│   └── test_stage4/
│
├── requirements.txt                 # Python依赖
├── pyproject.toml                   # 项目元信息
├── ARCHITECTURE.md                  # 本文档
└── README.md                        # 项目简介与快速开始
```

---

## 三、数据契约（Data Contract）

阶段之间**仅通过文件系统交换数据**。这是最重要的设计约束。

### 阶段1 → 阶段2

```
产物: data/stage1_output/current_sequence.csv
格式:
  step_index, target_u, applied_current_A
  0,          0.7321,   1.73
  1,          0.1548,   1.15
  ...

产物: data/stage1_output/mc_metadata_log.csv
格式: 同上，含完整元数据
```

### 阶段2 → 阶段3

```
产物: data/stage2_output/{experiment_name}/
  ├── 001_000000_858.jpg          # 第1周期第0帧，858ms时间戳
  ├── 001_000001_875.jpg          # 第1周期第1帧
  ├── ...
  ├── experiment_summary.csv      # 每周期: cycle_index, current_value, frames_saved
  └── experiment.log              # 毫秒级时间戳日志

文件命名规则: {cycle_index:03d}_{frame_count:06d}_{timestamp_ms:03d}.jpg
```

### 阶段3 → 阶段4

```
产物: data/stage3_output/particles.csv
格式:
  original_filename,    file_prefix, frame, id, center_x, center_y, radius
  001_000000_858.jpg,   1,           1,     1,  450.2,    320.8,    8.5
  001_000000_858.jpg,   1,           1,     2,  510.3,    315.2,    7.9
  ...

产物: data/stage3_output/features.csv
格式:
  frame, entropy, norm_system_radius, norm_neighbor_dist
  1,     2.34,    15.6,              2.1
  2,     2.41,    15.2,              2.3
  ...
```

### 阶段4 输出

```
产物: output/figures/mc_curve.png
产物: output/figures/mc_comparison.png     (多次实验对比)
产物: output/reports/benchmark_summary.csv (各项benchmark结果汇总)
```

---

## 四、配置文件说明

### config/global.yaml

```yaml
# 所有阶段共享的全局配置
project:
  name: "SwarmReservoir"
  random_seed: 42

paths:
  data_root: "./data"
  output_root: "./output"

hardware:
  camera:
    type: "Hikvision"           # 相机类型
    connection: "GigE"          # USB / GigE
  power_supply:
    port: "COM8"
    baudrate: 9600
    model: "ITECH"              # SCPI兼容设备

logging:
  level: "DEBUG"
  format: "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s"
```

### config/stage1_generate.yaml

```yaml
# 阶段1：信号生成参数
signal:
  strategy: "uniform_random"    # 可选: uniform_random | sinusoidal | binary_sequence | narma
  n_steps: 100
  decimal_places: 2             # 硬件电流精度（小数位数）

  # uniform_random 策略参数
  uniform_random:
    low_limit: 1.0              # 电流下限 A
    high_limit: 2.0             # 电流上限 A

  # sinusoidal 策略参数（预留）
  sinusoidal:
    amplitude: 0.5
    frequency: 0.1
    offset: 1.5

output:
  current_list_file: "data/stage1_output/current_sequence.csv"
  metadata_file: "data/stage1_output/mc_metadata_log.csv"
```

### config/stage2_experiment.yaml

```yaml
# 阶段2：实验采集参数
experiment:
  name: "MCTest_20260621"       # 实验名称，用于创建子目录
  voltage: 15.0                 # 电源电压 V
  cycle_count: null             # null = 使用电流序列的全部步数
  frame_interval_ms: 16.67      # 帧间隔（60 FPS = 16.67ms）
  
  # 每周期时序（单位：秒）
  timing:
    phase_a: 0.0                # 发送电流命令
    phase_b: 0.5                # 保持电流 + 启动图像保存
    phase_c: 0.7                # 发送 CURR 0.0
    phase_d: 0.9                # 停止图像保存
    phase_e: 1.0                # 进入下一周期

camera:
  framerate: 90.0
  exposure_time: 10000.0        # 微秒
  gain: 0.0
  trigger_mode: "continuous"    # continuous / software_trigger

output:
  image_dir: "data/stage2_output"
```

### config/stage3_extract.yaml

```yaml
# 阶段3：特征提取参数
mask:
  center_x: 902                 # 培养皿圆心 X（通过 find_center 确定）
  center_y: 1157                # 培养皿圆心 Y
  radius: 450                   # 培养皿半径

circle_detection:
  binary_threshold: 130
  canny_weak: 135
  canny_strong: 170
  dp: 1.5
  min_dist: 20
  param1: 80
  param2: 16
  min_radius: 6
  max_radius: 12

features:
  enabled:                      # 启用哪些特征计算
    - entropy
    - system_radius
    - neighbor_spacing
  
  entropy:
    method: "voronoi"           # voronoi / kdtree
    particle_diameter: 20.0     # 粒子直径（像素），用于距离归一化
    bins: [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    min_particles: 5            # 少于这个粒子数则跳过该帧

  system_radius:
    min_particles: 3
  
  neighbor_spacing:
    method: "voronoi"

output:
  particles_csv: "data/stage3_output/particles.csv"
  features_csv: "data/stage3_output/features.csv"
  images_with_labels: "data/stage3_output/labeled_images/"   # 可选，调试用
```

### config/stage4_benchmark.yaml

```yaml
# 阶段4：基准测试参数
preprocessing:
  features:
    - entropy                   # 使用哪些特征列
    - system_radius
    # - neighbor_spacing        # 可多选
  frames_per_group: 22          # 每组多少帧（等于每周期帧数）
  normalize: true
  pca:
    enabled: true
    variance_threshold: 0.95

benchmarks:
  enabled:                      # 启用哪些 benchmark
    - memory_capacity
    # - nonlinearity            # 预留
    # - separation_curve        # 预留
  
  memory_capacity:
    d_max: 40                   # 最大延迟
    train_ratio: 0.8            # 训练集比例（0=全量拟合）
    regression:
      method: "ridge"           # ridge / linear / lasso
      alpha: 0.1                # Ridge 正则化系数

output:
  figures_dir: "output/figures"
  reports_dir: "output/reports"
```

---

## 五、参数调优与验证流程

这是开发过程中最高频的操作。分为三条路径：

### 路径A：参数调优（交互式）

```
用途：拿到新一批数据后，手动确定最优参数
工具：scripts/verify_stage3.py 或 notebooks/03_feature_exploration.ipynb

流程：
  1. 选取 3-5 张代表性图片
  2. 运行验证脚本，逐步调整参数：
     ├── 蒙版参数   → 确认培养皿区域正确
     ├── 二值化阈值 → 确认粒子与背景分离清晰
     ├── 边缘参数   → 确认粒子边缘被完整捕获
     └── 圆检测参数 → 确认所有粒子被检测，无误检
  3. 确认参数后，手动写入 config/stage3_extract.yaml
```

### 路径B：单步测试（快速验证）

```
用途：验证管线代码逻辑是否正确（不是验证参数）
工具：scripts/test_stage3_step.py

流程：
  1. 从数据集中随机抽取 10 张图片
  2. 用 config 中的参数运行完整管线
  3. 输出中间产物到临时目录
  4. 自动检查：
     ├── 粒子数是否在合理范围（如 15-25 个）
     ├── 是否有异常帧（粒子数为 0 或过多）
     └── 特征值是否有 NaN
  5. 通过 → 继续；失败 → 回头调参
```

### 路径C：批量运行（正式执行）

```
用途：确认无误后全量运行
工具：scripts/run_stage3.py

流程：
  1. 读取 config/stage3_extract.yaml
  2. 读取全部图片
  3. 批量处理，输出 particles.csv + features.csv
  4. 打印汇总统计（总帧数、平均粒子数、异常帧列表）
```

### 完整工作流

```
             拿新数据
                │
                ▼
    ┌──────────────────────┐
    │ 路径A: 参数调优       │  ← 在几张代表性图片上调参
    │ verify_stage3.py     │     确定参数后写入 config
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ 路径B: 单步测试       │  ← 跑 10 张验证管线逻辑
    │ test_stage3_step.py  │     通过 ─→ 继续
    └──────────┬───────────┘     失败 ─→ 回到路径A
               │
               ▼
    ┌──────────────────────┐
    │ 路径C: 批量运行       │  ← 全量处理
    │ run_stage3.py        │     输出最终产物
    └──────────────────────┘
```

---

## 六、如何扩展功能

### A. 添加新的信号生成策略（阶段1）

```python
# 1. 创建文件 src/stage1_generate/strategies/sinusoidal.py
from .base import BaseSignalStrategy

class SinusoidalStrategy(BaseSignalStrategy):
    def name(self) -> str:
        return "sinusoidal"
    
    def generate(self, n_steps: int, config: dict) -> np.ndarray:
        cfg = config["sinusoidal"]
        t = np.arange(n_steps)
        return cfg["offset"] + cfg["amplitude"] * np.sin(cfg["frequency"] * t)

# 2. 在 config/stage1_generate.yaml 中添加参数
#    signal:
#      strategy: "sinusoidal"
#      sinusoidal:
#        amplitude: 0.5
#        frequency: 0.1
#        offset: 1.5

# 3. 运行（自动通过 strategy 名称匹配到对应类）
```

### B. 添加新的特征提取（阶段3）

```python
# 1. 创建文件 src/stage3_extract/features/orientation_order.py
from .base import BaseFeature

class OrientationOrderFeature(BaseFeature):
    def name(self) -> str:
        return "orientation_order"
    
    def compute(self, particles_df: pd.DataFrame, config: dict) -> float:
        # 你的方向序参数计算逻辑
        ...
        return order_parameter

# 2. 在 config/stage3_extract.yaml 中启用
#    features:
#      enabled:
#        - entropy
#        - orientation_order   # 新增

# 3. 运行（管线自动调用所有 enabled 的特征）
```

### C. 添加新的 Benchmark（阶段4）

```python
# 1. 创建文件 src/stage4_benchmark/benchmarks/nonlinearity.py
from .base import BaseBenchmark

class NonlinearityBenchmark(BaseBenchmark):
    def name(self) -> str:
        return "nonlinearity"
    
    def run(self, features: np.ndarray, targets: np.ndarray, config: dict) -> dict:
        # 计算非线性度
        ...
        return {"nonlinearity_score": score, "linear_fit_r2": linear_r2}
    
    def plot(self, results: dict, output_dir: Path) -> None:
        # 生成图表
        ...

# 2. 在 config/stage4_benchmark.yaml 中启用
#    benchmarks:
#      enabled:
#        - memory_capacity
#        - nonlinearity   # 新增

# 3. 运行
```

### D. 添加新的实验设备（阶段2）

```python
# src/stage2_experiment/hardware/power_supply.py
# 如果换用不同品牌的电源，只需继承基类：

class BasePowerSupply(ABC):
    @abstractmethod
    def connect(self, port: str, baudrate: int) -> bool: ...
    @abstractmethod
    def send_command(self, cmd: str) -> bool: ...
    @abstractmethod
    def set_current(self, value: float) -> bool: ...
    @abstractmethod
    def close(self) -> None: ...

class ITECHPowerSupply(BasePowerSupply):
    ...  # 当前实现

class RigolPowerSupply(BasePowerSupply):
    ...  # 新增品牌
```

### 扩展规则速查表

| 要做什么 | 在哪里加文件 | 继承哪个基类 | 修改哪个 config |
|----------|------------|-------------|----------------|
| 新信号策略 | `stage1_generate/strategies/` | `BaseSignalStrategy` | `stage1_generate.yaml` |
| 新相机品牌 | `stage2_experiment/hardware/camera.py` | 修改 `CameraDriver` 基类 | `global.yaml` |
| 新电源品牌 | `stage2_experiment/hardware/power_supply.py` | `BasePowerSupply` | `global.yaml` |
| 新图像预处理 | `stage3_extract/preprocessing/` | 独立函数即可 | `stage3_extract.yaml` |
| 新粒子检测算法 | `stage3_extract/detection/` | `BaseDetector` | `stage3_extract.yaml` |
| 新特征 | `stage3_extract/features/` | `BaseFeature` | `stage3_extract.yaml` |
| 新 benchmark | `stage4_benchmark/benchmarks/` | `BaseBenchmark` | `stage4_benchmark.yaml` |
| 新可视化 | `notebooks/` 或 `stage4_benchmark/evaluation.py` | — | — |

---

## 七、Jupyter Notebooks 的定位

`notebooks/` 目录用于**探索性分析和临时分析**，与 `src/` 的关系如下：

| | `src/` | `notebooks/` |
|---|---|---|
| **性质** | 正式代码，可版本管理 | 临时探索，个人工作记录 |
| **内容** | 核心逻辑、函数、类 | 调参过程、可视化探索、论文图草稿 |
| **复用** | 被 scripts 和 pipeline 调用 | 不导出函数，只消费 src |
| **版本** | 提交到 git | 视情况提交或不提交 |

**使用原则**：
- 在 notebook 中确定算法和参数后 → 将逻辑移入 `src/`，参数写入 `config/`
- notebook 不包含核心业务逻辑，只包含调用和分析
- 推荐 notebook 命名加数字前缀以保持顺序：`01_`, `02_`...

---

## 八、运行方式

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 编辑配置（至少要修改路径和硬件端口）
#    编辑 config/global.yaml 和各个 stage 的 config

# 3. 分步运行
python scripts/run_stage1.py          # 生成电流序列
python scripts/run_stage2.py          # 硬件实验（需连接相机和电源）
python scripts/verify_stage3.py       # 选几张图确定参数
python scripts/test_stage3_step.py    # 单步测试 10 张
python scripts/run_stage3.py          # 全量特征提取
python scripts/run_stage4.py          # 基准测试

# 4. 或一键运行（会提示确认）
python scripts/run_all.py
```

### 常用工作流

```bash
# 场景1: 新实验数据到了，只想跑阶段3和4
python scripts/run_stage3.py --input data/stage2_output/20260621_experiment/
python scripts/run_stage4.py --input data/stage3_output/20260621_features.csv

# 场景2: 只重新跑 benchmark（特征提取不变）
python scripts/run_stage4.py --input data/stage3_output/features.csv --skip-preprocess

# 场景3: 比较两次实验的 MC 曲线
# 在 notebook 中: notebooks/04_mc_analysis.ipynb
```

---

## 九、从旧代码迁移计划

| 当前文件 | 新位置 | 迁移要点 |
|----------|--------|----------|
| `Benchmark/Data_generate.py` | `src/stage1_generate/strategies/uniform_random.py` | 拆分为策略类 + 量化模块 |
| `Current_Input/SRCExperiment.py` | `src/stage2_experiment/experiment.py` | 重构为状态机，分离IO和逻辑 |
| `Current_Input/Serial.py` | `src/stage2_experiment/hardware/power_supply.py` | 添加基类 + 更健壮的错误处理 |
| `Current_Input/CamOperation_class.py` | `src/stage2_experiment/hardware/camera.py` | 消除全局变量，替换暴力杀线程 |
| `Current_Input/camera_initial.py` | `src/stage2_experiment/ui.py` | 分离UI和业务逻辑 |
| `Parameter_extract/Code/Circle_Recognize.py` | `src/stage3_extract/detection/circle_detector.py` | 修复逻辑错误，添加类型标注 |
| `Parameter_extract/Code/Mask.py` | `src/stage3_extract/preprocessing/mask.py` | 保持纯函数风格 |
| `Parameter_extract/Code/HNDist_Voronoi.py` | `src/stage3_extract/features/entropy.py` | 合并3份重复实现为一份 |
| `Parameter_extract/Code/HNDist_KDTree.py` | `并入 entropy.py` | 作为 entropy 的一个 method 选项 |
| `Parameter_extract/Code/Contraction_interdistance.py` | `src/stage3_extract/features/system_radius.py` + `neighbor_spacing.py` | 拆为两个独立特征 |
| `Benchmark/MC.py` | `src/stage4_benchmark/benchmarks/memory_capacity.py` | 区分训练/测试集，添加交叉验证 |
| `Benchmark/MC_PAC.py` | `并入 memory_capacity.py` | PCA 作为预处理选项而非独立脚本 |
| `Benchmark/flatten_data.py` | `src/stage4_benchmark/preprocessing.py` | 合并 flatten + normalize 为统一预处理 |
| `Benchmark/normalize_data.py` | `同上` | 同上 |
| `Parameter_extract/utils/*` | `src/common/io_utils.py` 或 `scripts/` | 按功能归类 |

---

## 十、注意事项

1. **数据目录不要提交到 git**：`data/` 和 `output/` 应加入 `.gitignore`
2. **config 中不要包含绝对路径**：使用相对路径 + `paths.data_root` 前缀
3. **每个阶段的产物有明确命名**：包含实验日期或名称，避免覆盖
4. **log 文件保留**：`experiment.log` 是排查硬件问题的关键
5. **粒子直径常数**：`D` 是像素值，不同实验设置可能不同，应在 config 中配置
6. **随机种子固定**：`config/global.yaml` 中设置 `random_seed` 保证可复现

---

> **文档维护**：当目录结构或数据契约发生变化时，请同步更新本文档。
