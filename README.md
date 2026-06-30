<p align="center">
  <h1 align="center">test_auto_route</h1>
  <p align="center"><strong>模型自动评测 + 智能路由程序</strong></p>
  <p align="center">
    <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python version"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg" alt="FastAPI"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  </p>
</p>

---

## 项目简介

**test_auto_route** 是一个模型能力评测 + 智能路由程序，解决两个核心问题：

- **自动评测**：客观、全面地评测大语言模型在多类题库上的能力
- **智能路由**：面对多个可用模型，根据用户输入自动选择最优模型

两个功能通过 `model_benchmarks.json` 串联：评测系统产出模型能力数据，路由系统消费这些数据进行智能选型。

**核心特性：**
- **KNN + 关键词先验路由** — domain 分类准确率高，纠正嵌入混淆
- **8 类纯文本题库 + 5 类视觉识图题库 + 1 类文生图题库** — 从选择题、数学推理到图表理解、视觉问答、文生图
- **多模态任务按能力类型路由** — 区分视觉识图与视觉生图，仅对具备对应能力类型的模型路由
- **6 种路由策略** — Average、Majority Voting、KNN、KNN+Prior、Ensemble 等
- **效率优先选型** — 80% 低难度任务自动选用高效模型，仅极难题才用专家模型
- **远程端点自动发现** — 给出 URL + Key 即可自动检测正在运行的模型，无需手动填模型名
- **多用户并发安全** — 路由推理和非流式调用均为异步，支持多用户并发访问，回答不会混淆
- **可视化调参台** — Web 界面拖动滑块实时调节路由权重，无需重启
- **OpenAI 兼容接口** — 标准 `/v1/chat/completions`，任意客户端即插即用
- **`@模型名` 指定模型** — 消息开头使用 `@model-name` 可跳过路由直接指定模型
- **强制路由勾选框** — Web 界面勾选模型即可强制路由，优先级仅次于 `@模型名`
- **GPU 加速** — 自动检测 GPU，嵌入推理 ~10ms

---

## 目录

- [快速开始](#快速开始)
- [模块化测试（按需选测）](#模块化测试按需选测)
- [Web UI 使用指南（面向最终用户）](#web-ui-使用指南面向最终用户)
- [工作流程](#工作流程)
- [项目结构](#项目结构)
- [路由算法](#路由算法)
- [模型选型公式](#模型选型公式)
- [可视化调参台](#可视化调参台)
- [支持的题库](#支持的题库)
- [多模态路由原理](#多模态路由原理)
- [API 接口](#api-接口)
- [配置指南](#配置指南)
- [常见问题](#常见问题)
- [License](#license)

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+（推荐 CUDA 11.8+ 用于 GPU 加速）
- 8GB+ 显存（可选，用于 GPU 加速嵌入推理）

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/test_auto_route.git

# 进入项目根目录（后续所有命令均在此目录下执行）
cd test_auto_route

# 创建虚拟环境
python -m venv venv

# 激活
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

**第 1 步：配置评测 API**

编辑 `autotest/config.py`：

```python
TEST_API_KEY = "sk-your-api-key-here"         # 待测模型 API 密钥
TEST_BASE_URL = "https://api.example.com/v1/chat/completions"
TEST_MODEL_NAME = "your-model-name"
```

**第 2 步：配置路由模型服务器**

编辑 `router/config.py`，推荐使用**远程端点自动发现**（只需 URL + Key，程序自动检测正在运行的模型）：

```python
REMOTE_SERVER_CONFIG = {
    # 方式一（推荐）：远程端点自动发现 —— 给出 URL 和 Key，程序调用 {url}/models 自动检测
    "remote_endpoints": [
        {"url": "https://your-server:8443/v1", "api_key": "sk-xxxx"},
    ],

    # 方式二（高级，向后兼容）：手动模型路由映射 —— 逐个填写模型名 → API URL
    # "model_routes": {
    #     "model-a": "https://your-server:8443/v1/chat/completions",
    # },
}
```

> 也可不修改配置文件，直接在 Web 界面（`/dashboard`）的「路由服务配置」卡片中切换到正常模式，填入 URL + Key 后点击「🔍 保存并检测模型」即可。

**第 3 步：运行自动测试**

在项目根目录执行：

```bash
python run.py test
```

测试完成后，结果保存在 `results/` 目录，能力数据汇总至 `model_benchmarks.json`。

> **按需选测模块**：可用 `--modules` 指定只测某些模块，结果**增量合并**到 `model_benchmarks.json`（不会覆盖其他模块的已有结果）。详见 [模块化测试](#模块化测试按需选测)。

**第 4 步：启动路由服务**

在项目根目录执行：

```bash
python run.py route
```

服务默认运行在 `http://localhost:8000`，启动后会**自动在浏览器中打开首页**（`http://localhost:8000/home`），可在首页选择进入「模型评测」或「路由调参」。如未自动打开，请手动访问该地址。

---

## 模块化测试（按需选测）

测试系统支持**按模块选测**：测完一个模块后只写入该模块的结果，后续再测其他模块时**增量合并**到 `model_benchmarks.json`，互不覆盖。路由时若分到对应任务，仅会在**正在运行且具备该模块测试结果**的模型中选型。

### 可选模块

| 模块名 | 说明 | 写入位置 |
|--------|------|---------|
| `text` | 纯文本（8 类题库） | `benchmarks` 子结构 |
| `vision_recognition` | 视觉识图（理解已有图片） | `multimodal.vision_recognition` |
| `image_generation` | 视觉生图（文生图） | `multimodal.image_generation` |
| `efficiency` | 效率（TTFT/吞吐/并发） | `efficiency` 子结构 |

### 一条链路示例

```bash
# 1) 先测通用语言模块（纯文本），结果写入 benchmarks 子结构
python run.py test --model qwen36-35b-a3b --modules text

# 2) 再测视觉识图模块，结果增量合并到 multimodal.vision_recognition
python run.py test --model qwen36-35b-a3b --modules vision_recognition

# 3) 再测视觉生图模块，结果增量合并到 multimodal.image_generation
python run.py test --model qwen36-35b-a3b --modules image_generation

# 4) 启动路由：路由时自动只选「正在运行 + 具备对应能力」的模型
python run.py route
```

测试结果统一更新到项目根目录的 **`model_benchmarks.json`**，路由程序启动时读取该文件作为选型依据。

### 常用参数

| 参数 | 说明 |
|------|------|
| `--model NAME` | 指定待测模型名（覆盖 `autotest/config.py` 的 `TEST_MODEL_NAME`） |
| `--modules a,b,c` | 选择测试模块，逗号分隔（见上表） |
| `--num-samples N` | 每题集最多测试 N 题（采样加速） |
| `--api-key KEY` | 指定 API 密钥（覆盖 `autotest/config.py`） |
| `--base-url URL` | 指定 API 地址（覆盖 `autotest/config.py`） |
| `--skip-probe` | 跳过能力探测（若明确知道模型支持可加速） |

### 能力探测（避免误判）

测试多模态模块前，程序会**自动探测**模型是否真正支持该能力（发送最小测试请求）：

- **支持** → 标记 `capability_status: "supported"` 并继续测试
- **明确不支持**（404/405 或 400 含"not support"等）→ 标记 `capability_status: "unsupported"` 并跳过该模块
- **无法确定**（超时/5xx）→ 不标记为不支持，避免把实际支持的模型误判为不支持

`capability_status` 写入 `model_benchmarks.json`，路由时**明确标记为 `unsupported` 的模型会被跳过**，不会被选去执行它不具备的任务。

### 启用多模态测试

测试视觉识图 / 文生图模块前，需在 `autotest/config.py` 中开启对应开关：

1. 识图测试：设置 `ENABLE_MULTIMODAL_TEST = True`
2. 文生图测试：设置 `ENABLE_T2I_TEST = True`
3. 确保待测模型支持对应接口（OpenAI Vision 兼容接口 / `/v1/images/generations` 接口）
4. 运行 `python run.py test --modules vision_recognition,image_generation`，测试前会自动探测能力，结果按能力类型写入 `model_benchmarks.json` 的 `multimodal` 子结构和 `capability_status`

> 也可在 Web UI 模型评测界面勾选「视觉识图」「视觉生图」模块直接测试，效果与命令行一致。

---

## Web UI 使用指南（面向最终用户）

系统提供完整的 Web 界面，**全程无需命令行**即可完成模型评测与路由调参。以下按实际操作顺序说明。

### 1. 启动服务并打开首页

在项目根目录执行：

```bash
python run.py route
```

服务启动后会**自动在浏览器中打开首页**（`http://localhost:8000/home`）。如未自动打开，手动在浏览器地址栏输入该地址即可。

### 2. 首页（/home）

首页是整个 Web 界面的入口，提供两个大按钮：

![首页](./docs/images/home_page.png)

| 按钮 | 进入的页面 | 用途 |
|------|-----------|------|
| **模型评测** | `/test` | 对模型进行能力测试（选择题、数学、视觉识图、文生图等） |
| **路由调参** | `/dashboard` | 可视化调节路由权重参数，拖动滑块即时生效 |

点击对应按钮即可进入相应功能页面。页面顶部也有导航链接，可随时在三个页面间切换。

### 3. 模型评测界面（/test）

在此页面可在浏览器中完成全部测试配置，无需编辑任何代码或配置文件。

![测试页](./docs/images/test_page.png)

**操作步骤：**

1. **填写 API 信息**
   - 待测模型名（如 `qwen36-35b-a3b`）
   - API 密钥（如 `sk-xxxx`）
   - API 地址（如 `https://api.example.com/v1/chat/completions`）
   - 以上三项会自动从 `autotest/config.py` 预填默认值，无需手动输入即可直接使用

2. **选择测试模块**（勾选复选框）
   - **通用语言**：纯文本能力（选择题、数学、长文本理解等 8 类题库）
   - **视觉识图**：理解已有图片的能力（图表理解、文字识别、视觉数学等 5 类题库）
   - **视觉生图**：文生图能力（根据文字描述生成图片）
   - **效率**：响应速度测试（首 Token 延迟、吞吐、并发上限）

3. **设置测试参数**
   - **全局采样题数**：对所有题库统一设置采样数（留空则按各题库默认值；填数字则对所有题库统一生效，除非下方分题库单独设置）
   - **分题库采样设置（可选）**：展开「⚙ 分题库采样设置」面板后，可为每个题库单独设置采样题数，留空则使用该题库默认值或全局值。优先级：分题库设置 > 全局采样题数 > 默认值
   - **能力探测**：勾选后，测多模态模块前会先发一个最小请求探测模型是否支持

4. **开始测试**
   - 点击「开始测试」按钮，下方实时显示运行日志
   - 测试过程中可随时点击「停止」中断

5. **查看结果**
   - 测试完成后，结果自动增量写入 `model_benchmarks.json`
   - 多次测试不同模块不会互相覆盖，结果会合并

> **提示**：如需只测某个模块而不影响其他模块已有结果，勾选对应模块即可——结果会增量合并，不会覆盖。

### 6. 导出测试文档（XLSX）

测试完成后，点击「📤 导出测试文档」按钮，可将测试结果导出为 Excel 文件：

- **下载 ZIP**：浏览器弹出下载对话框，由用户选择保存位置（推荐普通用户使用）
- **导出到指定目录**：输入服务器路径，将 xlsx 文件复制到该目录（适合服务器端操作）

**边界情况处理**：

| 情况 | 系统提示 |
|------|---------|
| 尚未进行测试 | 提示"尚未进行测试，请先运行测试后再导出文档" |
| 结果目录不存在 | 提示"未找到模型 X 的测试结果目录，请先完成测试" |
| 保存目录已有同名文件 | 询问"是否覆盖？"，确认后覆盖 |
| 文件被 Excel 打开无法写入 | 提示"文件被占用，请关闭后重试" |
| 保存目录不存在 | 自动创建 |

> 更详细的 UI 操作说明见 [docs/Web_UI操作手册.md](./docs/Web_UI操作手册.md)。

### 4. 路由调参台（/dashboard）

在此页面可通过拖动滑块实时调节路由权重，无需重启服务。详见下文 [可视化调参台](#可视化调参台) 章节。

### 5. 仅测试路由效果（不消耗 API 额度）

如想验证路由选型是否准确，但不实际调用模型（不消耗 API 费用），可使用命令行的路由分析接口：

```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt":"3x+5=20,求x"}'
```

返回结果会显示路由器选择的模型和原因，但不会真正调用远程模型。

---

## 工作流程

![工作流程](./workflow.svg)

---

## 项目结构

```
test_auto_route/
├── run.py                          # 【入口】统一入口脚本
├── requirements.txt                # Python 依赖
├── .gitignore
├── README.md                       # 项目说明（中文）
├── README_EN.md                    # 项目说明（英文）
├── 项目说明文档.md                  # 详细技术文档
├── model_benchmarks.json           # 模型评测数据（路由决策依据）
│
├── autotest/                       # ===== 模型自动评测 =====
│   ├── config.py                   # 评测配置（API密钥、路径）
│   ├── main.py                     # 评测主控
│   ├── model_api.py                # 模型API调用 + 评分逻辑
│   ├── parser.py                   # 题集解析器（8种格式）
│   ├── utils.py                    # 结果导出（XLSX）
│   ├── benchmark_efficiency.py     # 效率测试（TTFT/吞吐/并发）
│   └── benchmarks_json.py          # JSON 汇总更新
│
├── router/                         # ===== 智能路由 =====
│   ├── config.py                   # 路由配置
│   ├── app.py                      # FastAPI 服务入口
│   ├── whoengine.py                # 【核心】WhoEngine 路由器
│   ├── routing_params.py           # 【新增】集中式可调参数管理
│   ├── router_engine.py            # 路由引擎
│   ├── model_client.py             # 远程模型客户端
│   ├── task_classifier.py          # 任务分类器（备用）
│   ├── scoring.py                  # 模型评分选择
│   ├── model_profiles.py           # 模型能力档案
│   └── static/
│       ├── home.html               # 首页（选择测试/路由）
│       ├── test.html               # 模型评测界面
│       └── dashboard.html          # 可视化调参台前端
│
├── docs/                           # ===== 文档 =====
│   └── 项目说明文档.docx            # 详细技术文档（DOCX）
│
├── benchmarks/                     # ===== 题库 =====
│   ├── mmlu_gsm8k_hellaswag/       # 基础题库
│   ├── bbh_longbench/              # 进阶题库
│   ├── training_extra/             # 路由器训练扩充样本
│   ├── workplace/                  # 职场主观题
│   └── multimodal/                 # 视觉多模态题库（5类，120题）
│       ├── chartqa-图表理解(20).txt
│       ├── textvqa-文字识别(20).txt
│       ├── mathvista-视觉数学(20).txt
│       ├── vqa-视觉问答(30).txt
│       ├── mmmu-多模态理解(30).txt
│       ├── generate_images.py     # 图片生成脚本
│       └── images/                # 题目配图（PNG）
│
├── results/                        # 测试结果（自动生成，已 gitignore）
└── models/                         # 嵌入模型缓存（自动生成，已 gitignore）
```

---

## 路由算法

### 算法演进

| 算法 | 准确率 | 特点 |
|------|--------|------|
| 岭回归（基线） | 71.0% | 线性决策边界 |
| Token 级投票 | 67.7% | 逐 Token 分类后投票 |
| KNN (k=20) | 83.9% | 非参数方法，拟合非线性边界 |
| **KNN + 关键词先验 (推荐)** | **高** | **KNN + 关键词偏置，纠正嵌入混淆** |

### 核心原理

**KNN + 关键词先验混合路由** 是此项目的核心创新：

1. **KNN 近邻软投票**：将 query 编码为多池化句向量（mean+max+cls），计算与所有训练样本的余弦相似度，取 top-k 近邻做 softmax 软投票
2. **关键词先验偏置**：为每个 domain 维护强信号关键词表，检测到关键词时生成先验概率分布
3. **混合决策**：最终概率 = α × KNN概率 + (1-α) × 关键词先验（α=0.5 效果最佳）

> 为什么需要关键词先验？错误分析发现，嵌入模型会将 "chemical formula" 误判为数学题（因训练数据中含大量英文数学题）。关键词先验提供强信号，在不破坏纯 ML 能力的前提下修正这类系统性混淆。

详细实验数据见 [项目说明文档](./项目说明文档.md)。

---

## 模型选型公式

路由选型综合考虑**能力**和**效率**，公式如下：

```
最终得分 = α × benchmark[domain] + (1-α) × efficiency
```

其中 α 由 domain 基础值 + 难度自适应调整得出：

| 难度 | 调整 | 含义 |
|------|------|------|
| ≥ 8（极难） | α += 0.20 | 几乎只看能力 |
| ≥ 6（较难） | α += 0.10 | 偏重能力 |
| ≤ 5（中等） | α -= 0.15 | 偏重效率 |
| ≤ 3（简单） | α -= 0.30 | 几乎只看效率 |

**效率分数**由三个子指标加权得出：

```
efficiency = 0.4 × norm(TPS) + 0.3 × norm(延迟) + 0.3 × norm(并发)
```

**默认预设：平衡（balanced）** — 在能力与效率间取得平衡，适合大多数场景。可通过可视化调参台一键切换为「效率优先」或「能力优先」预设。修改参数会即时生效并记录到 `router/params_changes.log` 日志文件（含时间戳和参数值快照），重启后恢复默认预设。

---

## 可视化调参台

系统内置一个 Web 可视化调参台，支持实时调节所有路由权重参数，无需重启服务。

![可视化调参台界面 1](./docs/images/dashboard_1.png)

![可视化调参台界面 2](./docs/images/dashboard_2.png)

![可视化调参台界面 3](./docs/images/dashboard_3.png)

### 访问方式

服务启动后会**自动在浏览器中打开**调参台。如未自动打开，请手动访问：

```
http://localhost:8000/dashboard
```

### 功能

- **路由服务配置（核心）**：在页面顶部「路由服务配置」卡片中可即时切换测试/正常模式、配置模型 API 地址
  - **测试模式开关**：勾选=测试模式（不连接远程服务器，仅验证路由逻辑）；取消勾选=正常模式（实际调用远程模型 API）
  - **请求超时 / SSL 校验**：可在线调整请求超时秒数和 SSL 证书校验开关（自签名证书需关闭）
  - **模型路由表**：可动态增删模型→API 地址映射（如 `qwen36-35b-a3b → https://xxx/v1/chat/completions`），所有模型可共用一个 URL（服务器通过 model 字段区分）。点击「💾 保存并应用路由配置」后即时生效
  - 提示：此处修改为内存级，重启后恢复 `router/config.py` 原值。如需永久生效请同步修改配置文件
- **6 大参数分组**：选型权重、难度评估、难度自适应、效率归一化、KNN 路由、关键词先验
- **滑块调节**：每个参数显示当前值、上下限，拖动即时显示数值
- **3 套预设方案**：效率优先 / 平衡（默认）/ 能力优先，一键切换
- **保存即生效**：修改后点击「保存并应用」，路由决策立即使用新参数
- **变更日志**：每次保存自动追加到 `router/params_changes.log`，记录时间戳和参数值快照
- **文档查看**：界面内嵌项目文档查看器，无需跳转
- **DOCX 下载**：界面内可直接下载详细技术文档

### 可调参数清单（共 34 项）

| 分组 | 参数 | 范围 | 默认值 |
|------|------|------|--------|
| 选型权重 | `domain_alpha.*`（8 个 domain） | 0.10 ~ 0.95 | 0.35 ~ 0.85（按 domain） |
| 难度评估 | `difficulty.base.*`（8 个 domain） | 1 ~ 10 | 3 ~ 8（按 domain 难度） |
| 难度评估 | `confidence_base` / `confidence_factor` | 0~1 / 1~10 | 0.5 / 4 |
| 难度自适应 | `boost_high` / `boost_medium` | 0 ~ 0.30 / 0 ~ 0.20 | 0.15 / 0.08 |
| 难度自适应 | `reduce_mid_low` / `reduce_low` | 0 ~ 0.25 / 0 ~ 0.40 | 0.10 / 0.20 |
| 难度自适应 | `alpha_min` / `alpha_max` | 0.05 ~ 0.50 / 0.50 ~ 1.00 | 0.20 / 0.95 |
| 效率归一化 | `weight_tps` / `weight_latency` / `weight_concurrency` | 0 ~ 1 | 0.40 / 0.30 / 0.30 |
| 效率归一化 | `tps_max` / `latency_max` / `concurrency_max` | 饱和点 | 60 / 3.0 / 20 |
| 效率归一化 | `default_score` | 0 ~ 1 | 0.5 |
| KNN 路由 | `k` / `sim_temp` | 1~50 / 1~30 | 20 / 10.0 |
| 关键词先验 | `alpha` | 0 ~ 1 | 0.7 |

> 完整参数说明详见 [项目说明文档](./项目说明文档.md) 第 9 章「参数详解」。

---

## 支持的题库

### 纯文本题库（8 类）

| 题库 | 类型 | 题数 | 评分方式 |
|------|------|------|---------|
| MMLU | 选择题 | 30 | 答案比对 |
| GSM8K | 数学填空 | 10 | 答案比对 |
| HellaSwag | 选择题 | 20 | 答案比对 |
| BBH 语义理解 | 选择题 | 10 | 答案比对 |
| BBH 数学推理 | 数学计算 | 10 | 答案比对 |
| LongBench | 长文本理解 | 10 | LLM 打分 (0-10) |
| 职场-项目经理 | 主观题 | 20 | LLM 打分 (0-10) |
| 职场-秘书 | 主观题 | 20 | LLM 打分 (0-10) |

### 视觉多模态题库（5 类，共 120 题）

| 题库 | 类型 | 题数 | 评分方式 | 来源 |
|------|------|------|---------|------|
| ChartQA | 图表理解 | 20 | 答案比对 | 业内标准图表问答基准 |
| TextVQA | 文字识别 | 20 | 答案比对 | 图中文字识别基准 |
| MathVista | 视觉数学 | 20 | 答案比对 | 视觉数学推理基准 |
| VQA | 视觉问答 | 30 | 答案比对 | 通用视觉问答基准 |
| MMMU | 多模态理解 | 30 | 答案比对 | 多学科多模态理解基准 |

> 多模态题集使用 OpenAI Vision 兼容接口（`image_url` + base64），仅在 `ENABLE_MULTIMODAL_TEST=True` 时运行。题目配图由 `benchmarks/multimodal/generate_images.py` 自动生成。

### 效率测试（efficiency 模块）

效率测试通过流式请求测量模型的响应速度，指标定义如下：

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **TTFT** | 首 Token 延迟（毫秒） | 从请求发出到收到第一个 token 的时间 |
| **单请求吞吐（TPS）** | 单请求场景下平均 token 输出速率（token/s） | `tokens / 生成耗时`，对多轮请求取算术平均 |
| **每通道吞吐** | 并发场景下每个通道的平均 token 产出（token/s） | `总 tokens / (并发数 × 总耗时)`，反映并发时每通道的实际效率 |
| **并发上限** | 吞吐未跌穿阈值时的最大并发数 | 并发每请求吞吐 ≥ 单请求吞吐 × 65% 时的最大并发 |

**吞吐计算说明**：
- **单请求吞吐** = `tokens / (首 token 到请求结束的时间)`，多轮取算术平均。
- **每通道吞吐** = `所有请求的 token 总数 / (并发数 × 并发总耗时)`。这是并发场景下每个通道真实产出的 token 速率，不受并行加速放大的影响。
  - 区别：旧版的「每请求视角吞吐」(mean(tokens/gen_time)) 在并行处理下会被放大，不能反映每通道实际效率。
  - 新算法明确按总产出与总耗时计算，物理含义清晰。

**并发扫描策略**：
- 前段密集扫描：`[1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]`
- 32 之后每次 `+10` 递增：`42, 52, 62, 72...`
- 终止条件：吞吐跌穿 65% 阈值 / 出现失败请求 / 达到安全上限 200

可在 `autotest/config.py` 调整 `EFFICIENCY_INITIAL_CONCURRENCY`、`EFFICIENCY_CONCURRENCY_STEP`、`EFFICIENCY_MAX_CONCURRENCY`。

---

## 多模态路由原理

系统支持视觉多模态任务的自动识别和路由，**按能力类型区分**视觉识图与视觉生图，且仅在**正在运行 + 具备对应能力**的模型中选型。

### 能力类型分组

`model_benchmarks.json` 的 `multimodal` 子结构按能力类型分组，并通过 `capability_status` 明确标记每个能力的状态：

| 能力类型 | 说明 | 包含 domain |
|---------|------|------------|
| `vision_recognition` | 视觉识图（理解已有图片） | chart_qa / text_vqa / math_vista / vqa / mmmu |
| `image_generation` | 视觉生图（生成新图片） | t2i |

`capability_status` 取值：

| 状态 | 含义 | 路由行为 |
|------|------|---------|
| `supported` | 已测试，模型具备该能力 | 可选 |
| `unsupported` | 已探测，模型/接口明确不支持 | **跳过**，不会被选去执行该任务 |
| `not_tested` / 缺失 | 未测试 | 回退到检查 `multimodal` 子结构是否有得分 |

### 工作原理

1. **任务识别**：路由器按优先级判定任务类型
   - **文生图任务**（优先）：文本命中生图关键词（画一张/生成图片/文生图 等）
   - **识图多模态任务**：含 image_url 或命中视觉关键词（图片/图表/截图/OCR 等）
   - **纯文本任务**：以上均不满足
2. **分支路由**（均仅在 `router/config.py` 的 `model_routes` 中已注册的运行中模型中选择）：
   - **纯文本**：走原有 KNN 路由，仅参考 `benchmarks` 子结构
   - **识图多模态**：仅对 `capability_status.vision_recognition != "unsupported"` 且有得分的模型选型
   - **文生图**：仅对 `capability_status.image_generation != "unsupported"` 且有得分的模型选型
3. **消息透传**：多模态消息（含 `image_url`）原样透传给远程模型

### 任务判定规则

- **文生图任务**：文本命中生图关键词（画一张/生成图片/文生图/draw a 等），优先级最高
- **识图多模态任务**：满足任一即为识图多模态
  - messages 中存在 `content` 为 list 且含 `type=image_url` 项
  - 文本中包含视觉关键词（图片/图表/截图/OCR/视觉/图像 等）
  - 文本中包含 base64 图片数据 URL（`data:image/`）
- **纯文本任务**：以上均不满足

> 多模态测试的启用方法见上文 [模块化测试 - 启用多模态测试](#启用多模态测试)，多模态请求示例见下文 [API 接口](#api-接口)。

---

## API 接口

路由服务提供以下 HTTP 接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/home` | GET | **首页**（选择测试/路由） |
| `/test` | GET | **模型评测界面**（Web UI） |
| `/v1/chat/completions` | POST | 聊天补全（核心接口，OpenAI 兼容） |
| `/v1/route` | POST | 仅查询路由结果（不调用远程模型） |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/models/@mention` | GET | 前端 @mention 模型列表 |
| `/dashboard` | GET | **可视化调参台**（Web 界面） |
| `/api/test/config` | GET | 获取测试默认配置（供 UI 预填） |
| `/api/test/run` | POST | 启动测试子进程（支持 `num_samples_map` 分题库采样） |
| `/api/test/status` | GET | 获取测试状态和增量日志 |
| `/api/test/stop` | POST | 停止测试子进程 |
| `/api/test/export` | POST | 导出测试结果为 XLSX（支持浏览器下载或复制到指定路径） |
| `/api/test/benchmarks` | GET | 获取各题库元信息（供分题库采样 UI 使用） |
| `/api/route/config` | GET / POST | 获取 / 更新路由服务配置（test_mode、remote_endpoints、model_routes、verify_ssl、request_timeout），即时生效 |
| `/api/route/discover` | POST | 触发远程模型自动发现（调用各端点 `{url}/models` 检测正在运行的模型） |
| `/api/route/status` | GET | 获取路由服务当前状态（运行模式、已注册模型、已发现模型数等） |
| `/api/route/models` | GET | 获取模型列表及能力类型（测试模式从测试结果读，非测试模式从远程 /v1/models 检测） |
| `/api/route/forced-model` | GET / POST | 获取 / 设置强制路由模型（勾选后所有请求路由到该模型，优先级仅次于 @模型名） |
| `/api/params` | GET / POST | 获取 / 更新路由参数 |
| `/api/params/meta` | GET | 获取参数元数据（含范围、说明） |
| `/api/params/preset` | GET / POST | 获取 / 应用预设方案 |
| `/api/params/reset` | POST | 重置为默认预设 |
| `/api/docs/markdown` | GET | 获取项目说明文档（Markdown） |
| `/api/docs/readme` | GET | 获取 README（Markdown） |
| `/api/docs/docx/download` | GET | 下载项目说明文档（DOCX） |
| `/health` | GET | 健康检查 |

### 示例

```bash
# 路由分析（不调用远程模型）
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt":"3x+5=20,求x"}'

# 聊天补全（自动路由 + 远程调用）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"3x+5=20,求x"}]}'
```

**指定模型（跳过路由）：** 在消息开头用 `@模型名`
```
@model-name 写一个快排算法
```

**多模态请求示例（识图）：** 在 messages 的 content 中传入 image_url 即可，路由器会自动识别为多模态任务：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片中的内容"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
      ]
    }]
  }'
```

> 多模态任务的路由判定规则见上文 [多模态路由原理](#多模态路由原理)。

---

## 配置指南

### 关键配置文件

| 文件 | 用途 |
|------|------|
| `autotest/config.py` | API 密钥、模型名、题库路径 |
| `router/config.py` | 路由策略、远程模型服务器地址、WhoEngine 配置 |
| `router/routing_params.py` | **集中式可调参数管理**（34 个参数 + 3 套预设 + 变更日志） |
| `router/static/dashboard.html` | **可视化调参台前端** |

### WhoEngine 配置示例

```python
WHOENGINE_CONFIG = {
    "embedder": "BAAI/bge-large-zh-v1.5",  # 嵌入模型（1024 维，与 whoengine.pt 训练时一致）
    "routing_strategy": "knn_prior",       # 推荐策略
    "knn_k": 20,                           # KNN 近邻数
    "knn_prior_alpha": 0.7,                # 先验混合系数（偏 KNN）
    "knn_sim_temp": 10.0,                  # 相似度温度
}
```

> ⚠️ `embedder` 必须与 `whoengine.pt` 检查点训练时使用的模型一致（默认 `bge-large-zh-v1.5`，1024 维 → 3072 维多池化特征）。使用 `bge-small-zh-v1.5`（512 维）会导致维度不匹配错误。

### 路由模型服务器配置

路由服务支持两种方式连接远程模型：

| 方式 | 说明 | 推荐度 |
|------|------|--------|
| `remote_endpoints` | 给出 URL + API Key，程序调用 `GET {url}/models` 自动检测正在运行的模型 | ⭐ 推荐 |
| `model_routes` | 手动逐个填写模型名 → API URL 映射 | 高级/向后兼容 |

两种方式可同时使用，`get_model_url()` 查找顺序为：先 `model_routes` → 再 `remote_endpoints` 自动发现缓存。

配置可在 `router/config.py` 中修改，也可通过 Web 界面（`/dashboard` → 正常模式 → 远程端点）在线配置，即时生效。

---

## 常见问题

**Q: 路由准确率低怎么办？**
A: 确认 `routing_strategy` 设为 `knn_prior`，删除 `whoengine.pt` 后重启服务重新训练。

**Q: 如何启用多模态视觉测试？**
A: 识图测试在 `autotest/config.py` 中设置 `ENABLE_MULTIMODAL_TEST = True`；文生图测试设置 `ENABLE_T2I_TEST = True`。然后运行 `python run.py test --modules vision_recognition,image_generation`。测试前会自动探测模型是否支持对应接口（Vision / images/generations），探测结果写入 `capability_status`，测试结果按能力类型增量合并到 `multimodal.vision_recognition` 和 `multimodal.image_generation`。

**Q: 多模态任务如何路由？**
A: 路由器按优先级判定任务类型：文生图（生图关键词）→ 识图多模态（图片/视觉关键词）→ 纯文本。三种任务均**仅在已注册或自动发现的运行中模型**中选择（`model_routes` 手动注册 + `remote_endpoints` 自动发现）。文生图任务仅对 `capability_status.image_generation != "unsupported"` 且有得分的模型选型；识图任务仅对 `capability_status.vision_recognition != "unsupported"` 且有得分的模型选型；纯文本走原有 KNN 路由，不受多模态结果影响。

**Q: 如何只测某个模块而不覆盖其他模块结果？**
A: 用 `--modules` 参数指定，例如 `python run.py test --model my-model --modules text`。结果会**增量合并**到 `model_benchmarks.json`，不会覆盖其他模块的已有结果。可选模块：`text` / `vision_recognition` / `image_generation` / `efficiency`。

**Q: 如何新增一个 Domain？**
1. 准备训练题目文件放到 `benchmarks/` 下
2. 在 `router/config.py` 的 `benchmark_files` 中添加路径
3. 在 `whoengine.py` 的 `DOMAIN_KEYWORDS_PRIOR` 中添加新 domain 的关键词
4. 删除 `whoengine.pt` 重启服务

**Q: 支持哪些嵌入模型？**
A: 支持所有 sentence-transformers 模型，推荐 `BAAI/bge-large-zh-v1.5`。首次运行自动缓存到 `models/sentence_transformers/`，后续无需联网。

**Q: 是否支持 GPU 加速？**
A: 是的，WhoEngine 自动检测 GPU，嵌入模型和 KNN 计算全部在 GPU 上执行，推理延迟 ~10ms。如需安装 CUDA 版 PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## License

[MIT](LICENSE)
