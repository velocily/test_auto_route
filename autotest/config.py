"""
============================================================================
模型自动测试 - 配置文件
============================================================================
在此修改 API 密钥、URL、模型名、题集路径、输出路径等。
所有路径均使用相对于项目根目录的相对路径，由 main.py 自动解析。
"""

import os

# =========================
# 项目根目录（自动检测）
# =========================
# 项目根目录 = 本文件所在目录的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 待测模型 API 配置
# =========================
# 在此填入你的模型 API 信息（OpenAI 兼容接口）
TEST_API_KEY = "sk-your-api-key-here"
TEST_BASE_URL = "https://api.example.com/v1/chat/completions"
TEST_MODEL_NAME = "your-model-name"

# =========================
# 打分模型 API 配置（用于主观题自动打分，暂时可和待测模型相同）
# =========================
JUDGE_API_KEY = "sk-your-api-key-here"
JUDGE_BASE_URL = "https://api.example.com/v1/chat/completions"
JUDGE_MODEL_NAME = "your-model-name"

# =========================
# 题集文件路径（相对于项目根目录）
# =========================
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "mmlu_gsm8k_hellaswag")

BENCHMARK_FILES = {
    "mmlu":      os.path.join("benchmarks", "mmlu_gsm8k_hellaswag", "mmlu-知识广度(30).txt"),       # 客观选择题
    "gsm8k":     os.path.join("benchmarks", "mmlu_gsm8k_hellaswag", "gsm8k-数学推理(10).txt"),      # 数学填空题
    "hellaswag": os.path.join("benchmarks", "mmlu_gsm8k_hellaswag", "hellaswag-常识推理(20).txt"),   # 客观选择题
    "workplace_pm":         os.path.join("benchmarks", "workplace", "职场角色测试问题表.xlsx"),       # 主观题-项目经理
    "workplace_secretary":  os.path.join("benchmarks", "workplace", "职场角色测试问题表.xlsx"),       # 主观题-秘书
    "bbh_semantic":         os.path.join("benchmarks", "bbh_longbench", "bbh-语义理解(10).txt"),     # 客观选择题
    "bbh_math":             os.path.join("benchmarks", "bbh_longbench", "bbh-数学推理(10).txt"),     # 数学计算题
    "longbench":            os.path.join("benchmarks", "bbh_longbench", "longbench-长上下文(10).txt"), # 长上下文主观题
}

# =========================
# 输出
# =========================
# 结果输出目录（每题集一个文件，文件名自动包含模型名和题库名）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# 题集显示名称（用于输出文件名）
BENCHMARK_NAMES = {
    "mmlu":                "mmlu-知识广度(30)",
    "gsm8k":               "gsm8k-数学推理(10)",
    "hellaswag":           "hellaswag-常识推理(20)",
    "workplace_pm":        "职场角色测试-项目经理(20)",
    "workplace_secretary": "职场角色测试-秘书(20)",
    "bbh_semantic":        "bbh-语义理解(10)",
    "bbh_math":            "bbh-数学推理(10)",
    "longbench":           "longbench-长上下文(10)",
}

# =========================
# 效率测试配置
# =========================
# 是否启用效率测试
ENABLE_EFFICIENCY_TEST = True

# 单用户测试轮数
EFFICIENCY_SINGLE_ROUNDS = 3

# 并发数初始扫描列表（前段密集，后段自动 +10 持续递增直到不满足阈值）
EFFICIENCY_INITIAL_CONCURRENCY = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]

# 超过初始列表后，每次增加的并发数
EFFICIENCY_CONCURRENCY_STEP = 10

# 并发数上限（None = 无上限，一直测到吞吐跌穿阈值或全部失败为止）
EFFICIENCY_MAX_CONCURRENCY = None

# 吞吐阈值（相对于单用户吞吐的比例，低于此值认为达到上限）
EFFICIENCY_THROUGHPUT_THRESHOLD = 0.65

# 效率测试单个请求超时（秒）
EFFICIENCY_TIMEOUT = 120

# =========================
# 汇总 JSON 路径
# =========================
BENCHMARKS_JSON_PATH = os.path.join(PROJECT_ROOT, "model_benchmarks.json")

# =========================
# 评测控制
# =========================
# 每题最大测试数（设为 None 表示全部测试）
NUM_SAMPLES = None

# 请求超时（秒）
REQUEST_TIMEOUT = 600

# 请求间隔（秒），避免 API 限流
REQUEST_INTERVAL = 1.0
