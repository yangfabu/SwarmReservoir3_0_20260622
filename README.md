# SwarmReservoir v3.0

群体储层计算（Swarm Reservoir Computing）研究框架。

## 管线概览

```text
阶段1              阶段2                 阶段3                 阶段4
generate ---------> experiment ---------> extract ----------> benchmark
电流序列生成        硬件实验采集          图像特征提取           基准测试评估
                   (相机 + 电源)         (粒子检测 + 特征)
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 编辑配置文件（详见下方各阶段说明）
#    config/global.yaml            — 路径、硬件端口、随机种子
#    config/stage1_generate.yaml   — 信号生成参数
#    config/stage2_experiment.yaml — 实验采集参数 + 输入电流文件
#    config/stage3_extract.yaml    — 特征提取参数 + 输入图像目录
#    config/stage4_benchmark.yaml  — 基准测试参数

# 3. 分步运行
python scripts/run_stage1.py          # 生成电流序列
python scripts/run_stage2.py          # 硬件实验（需连接相机和电源）
python scripts/run_stage2.py --headless  # 无头模式（自动打开相机）

# 阶段3：三步调参流程（推荐）
python scripts/verify_stage3.py --image <单张图路径>   # 路径A: 可视化调参
python scripts/test_stage3_step.py --samples 20       # 路径B: 抽样验证
python scripts/run_stage3.py                          # 路径C: 全量提取

python scripts/run_stage4.py          # 基准测试

# 4. 或一键运行
python scripts/run_all.py
```

---

## 配置文件说明

### `config/global.yaml` — 全局配置（所有阶段共享）

| 节 | 关键参数 | 说明 |
| --- | --- | --- |
| `project` | `random_seed: 42` | 全局随机种子 |
| `paths` | `data_root`, `output_root` | 数据与输出根目录 |
| `hardware.camera` | `type`, `connection` | 相机型号与连接方式 |
| `hardware.power_supply` | `port`, `baudrate` | 电源串口与波特率 |
| `logging` | `level`, `format` | 日志级别与格式 |

### `config/stage1_generate.yaml` — 阶段1：电流序列生成

| 节 | 关键参数 | 说明 |
| --- | --- | --- |
| `signal.strategy` | `"discrete_set"` | 生成策略：`uniform_random` / `sinusoidal` / `binary_sequence` / `discrete_set` / `narma` |
| `signal.n_steps` | `100` | 生成的电流值总数 |
| `signal.decimal_places` | `1` | 电流精度（保留小数位数） |
| `signal.discrete_set.allowed_values` | `[1.0, 1.5, 2.0, ...]` | 离散候选电流值集合 |
| `output` | `current_list_file`, `metadata_file` | 产物路径 |

### `config/stage2_experiment.yaml` — 阶段2：硬件实验采集

```yaml
input:
  current_list_file: "data/stage1_output/current_sequence.csv"  # <-- 选择哪个电流文件
```

| 节 | 关键参数 | 说明 |
| --- | --- | --- |
| `input.current_list_file` | 路径字符串 | **电流序列输入文件**，通常指向阶段1产物，也可手动指定 |
| `experiment.name` | `"MCTest_20260622"` | 实验名称，图像保存到 `data/stage2_output/<name>/` |
| `experiment.voltage` | `15.0` | 电源电压（V），恒定 |
| `experiment.cycle_count` | `null` | `null` = 跑完所有电流值；填数字 = 只跑前 N 步 |
| `experiment.frame_interval_ms` | `16.67` | 图像保存帧间隔（60 FPS = 16.67ms） |
| `experiment.timing` | `phase_a ~ phase_e` | 每周期 1 秒内的时序划分（见下方时序图） |
| `camera` | `framerate`, `exposure_time`, `gain` | 相机参数 |
| `output.image_dir` | `"data/stage2_output"` | 图像输出根目录 |

**每周期时序（1 秒）：**

```text
t=0.0  phase_a -> 发送 "CURR <值>" 命令
t=0.5  phase_b -> 启动图像保存线程（开始高速拍照）
t=0.7  phase_c -> 发送 "CURR 0.0"（断电）
t=0.9  phase_d -> 停止图像保存
t=1.0  phase_e -> 进入下一个电流值的周期
```

有效拍照窗口为 0.5s ~ 0.9s（共 0.4 秒）。

### `config/stage3_extract.yaml` — 阶段3：图像特征提取

```yaml
input:
  image_dir: "data/stage2_output/MCTest_20260622"  # <-- 选择哪个实验的图像目录
  max_images: null                                   # <-- 限制处理张数（null = 全部）
```

| 节 | 关键参数 | 说明 |
| --- | --- | --- |
| `input.image_dir` | 路径字符串 | **待处理图像目录**，指向阶段2的某个实验输出 |
| `input.max_images` | `null` 或数字 | 最多处理张数，`null` = 全部 |
| `mask.center_x / y` | 像素坐标 | 培养皿圆心，通过 `find_center` 工具确定 |
| `mask.radius` | 像素 | 培养皿半径 |
| `circle_detection.binary_threshold` | `130` | 二值化阈值（0-255），粒子与背景分离 |
| `circle_detection.canny_weak / strong` | `135 / 170` | Canny 边缘检测双阈值 |
| `circle_detection.dp` | `1.5` | Hough 累加器分辨率 |
| `circle_detection.min_dist` | `20` | 两圆心最小间距 |
| `circle_detection.param1 / param2` | `80 / 16` | Hough 检测灵敏度参数 |
| `circle_detection.min_radius / max_radius` | `6 / 12` | 粒子半径范围（像素） |
| `features.enabled` | `[entropy, system_radius, ...]` | 启用哪些特征 |
| `features.<name>.method` | `"voronoi"` / `"kdtree"` | 特征计算方法 |
| `features.<name>.min_particles` | `5` | 少于此粒子数则跳过该帧 |
| `output` | `particles_csv`, `features_csv` | 产物路径 |

---

## 阶段3 参数调优工作流

阶段3 的参数（蒙版位置、二值化阈值、Canny/Hough 参数）需要根据你的图像来调整。
项目提供了**三条路径**，按顺序使用：

### 路径 A：单张图可视化调参

```bash
python scripts/verify_stage3.py --image "data/stage2_output/MCTest_20260622/001_000000_858.jpg"
```

弹出一个 matplotlib 窗口，展示 **6 个子图**，帮你直观判断每个参数的效果：

| 子图 | 内容 | 判断标准 |
| --- | --- | --- |
| Original | 原始图像 | 图像质量是否正常 |
| Masked | 圆形蒙版后 | 蒙版是否精准覆盖培养皿区域（调 `mask.center_x/y`, `radius`） |
| Binary | 二值化 | 粒子是否清晰分离、无粘连（调 `binary_threshold`） |
| Canny | 边缘检测 | 粒子边缘是否完整、背景无噪点（调 `canny_weak/strong`） |
| Detected | 检测结果叠加 | 所有粒子是否被检出、无错误检测（调 Hough 参数） |
| Info | 当前参数值 | 确认好的参数组合，复制到 YAML |

**调参顺序建议**：蒙版 -> 二值化 -> Canny -> Hough（`min_radius/max_radius` 按粒子实际大小设定）

调好后，将参数写入 `config/stage3_extract.yaml`。

### 路径 B：抽样验证稳健性

```bash
python scripts/test_stage3_step.py --samples 20
python scripts/test_stage3_step.py --samples 20 --input data/stage2_output/MyExp/
```

随机抽取 20 张图跑完整管线，输出每张图的粒子数和特征值。关键检查项：

- **异常帧**（粒子数=0）：应该为 0
- **粒子数波动**：不同帧的粒子数应稳定（如 45+/-5）
- **NaN 值**：所有特征值应正常计算

如果测试不通过，回到路径 A 继续调参。

### 路径 C：全量批量处理

```bash
python scripts/run_stage3.py                          # 使用配置文件中的 input.image_dir
python scripts/run_stage3.py --max 100                # 只处理前 100 张
python scripts/run_stage3.py --input data/stage2_output/MyExp/  # 手动指定目录
```

全量跑完后生成：

```text
data/stage3_output/
|-- particles.csv       # 每帧每个粒子的位置和半径
|-- features.csv        # 每帧的特征值（entropy, system_radius, ...）
```

---

## 目录结构

详见 [ARCHITECTURE.md](ARCHITECTURE.md)
