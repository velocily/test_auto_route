<p align="center">
  <h1 align="center">test_auto_route</h1>
  <p align="center"><strong>模型自动评测 + 智能路由 — 一体化平台</strong></p>
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

**test_auto_route** 是一个模型能力评测 + 智能路由的一体化平台，解决两个核心问题：

- **自动评测**：客观、全面地评测大语言模型在多类题库上的能力
- **智能路由**：面对多个可用模型，根据用户输入自动选择最优模型

两个功能通过 `model_benchmarks.json` 串联：评测系统产出模型能力数据，路由系统消费这些数据进行智能选型。

**核心特性：**
- **KNN + 关键词先验路由** — domain 分类准确率 **96.8%**，原题准确率 **100%**
- **8 类题库** — 从选择题、数学推理到长上下文、职场主观题
- **6 种路由策略** — Average、Majority Voting、KNN、KNN+Prior、Ensemble 等
- **OpenAI 兼容接口** — 标准 `/v1/chat/completions`，任意客户端即插即用
- **`@模型名` 指定模型** — 消息开头使用 `@model-name` 可跳过路由直接指定模型
- **GPU 加速** — 自动检测 GPU，嵌入推理 ~10ms

---

## 目录

- [快速开始](#快速开始)
- [工作流程](#工作流程)
- [项目结构](#项目结构)
- [路由算法](#路由算法)
- [支持的题库](#支持的题库)
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

编辑 `router/config.py`：

```python
REMOTE_SERVER_CONFIG = {
    "model_routes": {
        "model-a": "https://your-server:8443/v1/chat/completions",
        "model-b": "https://your-server:8443/v1/chat/completions",
    },
}
```

**第 3 步：运行自动测试**

```bash
python run.py test
```

测试完成后，结果保存在 `results/` 目录，能力数据汇总至 `model_benchmarks.json`。

**第 4 步：启动路由服务**

```bash
python run.py route
```

服务默认运行在 `http://0.0.0.0:8000`。

---

## 工作流程

```
┌──────────────────┐     ┌──────────────────┐
│  autotest 模块    │     │  router 模块      │
│  模型自动评测      │ ──> │  智能路由服务       │
│                  │ 产出 │                  │
│  • 题库解析       │     │  • WhoEngine 分类  │
│  • API 调用       │     │  • 模型选型        │
│  • 自动评分       │     │  • 远程调用        │
│  • 效率测试       │     │  • OpenAI 接口     │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         ▼                        ▼
   model_benchmarks.json ──── 路由决策依据
   (能力数据桥梁)
```

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
│   ├── router_engine.py            # 路由引擎
│   ├── model_client.py             # 远程模型客户端
│   ├── task_classifier.py          # 任务分类器（备用）
│   ├── scoring.py                  # 模型评分选择
│   └── model_profiles.py           # 模型能力档案
│
├── benchmarks/                     # ===== 题库 =====
│   ├── mmlu_gsm8k_hellaswag/       # 基础题库
│   ├── bbh_longbench/              # 进阶题库
│   ├── training_extra/             # 路由器训练扩充样本
│   └── workplace/                  # 职场主观题
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
| **KNN + 关键词先验 (推荐)** | **96.8%** | **KNN + 关键词偏置，纠正嵌入混淆** |

### 核心原理

**KNN + 关键词先验混合路由** 是此项目的核心创新：

1. **KNN 近邻软投票**：将 query 编码为多池化句向量（mean+max+cls），计算与所有训练样本的余弦相似度，取 top-k 近邻做 softmax 软投票
2. **关键词先验偏置**：为每个 domain 维护强信号关键词表，检测到关键词时生成先验概率分布
3. **混合决策**：最终概率 = α × KNN概率 + (1-α) × 关键词先验（α=0.5 效果最佳）

> 为什么需要关键词先验？错误分析发现，嵌入模型会将 "chemical formula" 误判为数学题（因训练数据中含大量英文数学题）。关键词先验提供强信号，在不破坏纯 ML 能力的前提下修正这类系统性混淆。

详细实验数据见 [项目说明文档](./项目说明文档.md)。

---

## 支持的题库

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

---

## API 接口

路由服务提供以下 HTTP 接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全（核心接口，OpenAI 兼容） |
| `/v1/route` | POST | 仅查询路由结果（不调用远程模型） |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/models/@mention` | GET | 前端 @mention 模型列表 |
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

---

## 配置指南

### 关键配置文件

| 文件 | 用途 |
|------|------|
| `autotest/config.py` | API 密钥、模型名、题库路径 |
| `router/config.py` | 路由策略、远程模型服务器地址、WhoEngine 配置 |

### WhoEngine 配置示例

```python
WHOENGINE_CONFIG = {
    "embedder": "BAAI/bge-large-zh-v1.5",  # 嵌入模型
    "routing_strategy": "knn_prior",       # 推荐策略
    "knn_k": 20,                           # KNN 近邻数
    "knn_prior_alpha": 0.5,                # 先验混合系数
    "knn_sim_temp": 10.0,                  # 相似度温度
}
```

---

## 常见问题

**Q: 路由准确率低怎么办？**
A: 确认 `routing_strategy` 设为 `knn_prior`，删除 `whoengine.pt` 后重启服务重新训练。

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
