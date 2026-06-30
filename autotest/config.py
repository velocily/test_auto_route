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
TEST_API_KEY = "sk-dummy"
TEST_BASE_URL = "https://u1015585-ca0q-76db8053.westb.seetacloud.com:8443/v1/chat/completions"
TEST_MODEL_NAME = "qwen36-35b-a3b"

# =========================
# 打分模型 API 配置（用于主观题自动打分，暂时可和待测模型相同）
# =========================
JUDGE_API_KEY = "sk-dummy"
JUDGE_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
JUDGE_MODEL_NAME = "deepseek-v4-flash"

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
# 多模态题集文件路径（相对于项目根目录）
# =========================
# 仅当待测模型支持视觉输入时启用（ENABLE_MULTIMODAL_TEST = True）
# 题目来源：业内标准视觉多模态基准（ChartQA / TextVQA / MathVista / VQA / MMMU）
MULTIMODAL_BENCHMARK_FILES = {
    "chartqa":    os.path.join("benchmarks", "multimodal", "chartqa-图表理解(20).txt"),     # 图表理解
    "textvqa":    os.path.join("benchmarks", "multimodal", "textvqa-文字识别(20).txt"),     # 文字识别
    "mathvista":  os.path.join("benchmarks", "multimodal", "mathvista-视觉数学(20).txt"),   # 视觉数学
    "vqa":        os.path.join("benchmarks", "multimodal", "vqa-视觉问答(30).txt"),         # 视觉问答
    "mmmu":       os.path.join("benchmarks", "multimodal", "mmmu-多模态理解(30).txt"),       # 多模态理解
}

# =========================
# 文生图（text-to-image）题集文件路径
# =========================
# 仅当待测模型支持图像生成时启用（ENABLE_T2I_TEST = True）
# 使用 OpenAI 兼容 /v1/images/generations 接口
T2I_BENCHMARK_FILES = {
    "t2i":        os.path.join("benchmarks", "multimodal", "t2i-文生图(50).txt"),           # 文生图
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

# 多模态题集显示名称
MULTIMODAL_BENCHMARK_NAMES = {
    "chartqa":    "chartqa-图表理解(20)",
    "textvqa":    "textvqa-文字识别(20)",
    "mathvista":  "mathvista-视觉数学(20)",
    "vqa":        "vqa-视觉问答(30)",
    "mmmu":       "mmmu-多模态理解(30)",
}

# 文生图题集显示名称
T2I_BENCHMARK_NAMES = {
    "t2i":        "t2i-文生图(15)",
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
# 安全上限 200（代码内硬限制，防止无限递增）
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
REQUEST_TIMEOUT = 120

# 请求间隔（秒），避免 API 限流
REQUEST_INTERVAL = 1.0

# =========================
# 多模态评测控制
# =========================
# 是否启用多模态视觉题集测试（仅当待测模型支持视觉输入时设为 True）
# 启用后会在纯文本题集之外，额外运行 5 类多模态题集（共 120 题）
# 多模态题集使用 OpenAI Vision 兼容接口（image_url + base64）
ENABLE_MULTIMODAL_TEST = True

# =========================
# 文生图评测控制
# =========================
# 是否启用文生图（text-to-image）题集测试（仅当待测模型支持图像生成时设为 True）
# 启用后会额外运行 1 类文生图题集（共 50 题）
# 使用 OpenAI 兼容 /v1/images/generations 接口
# 评分由打分模型（需支持多模态视觉输入）对生成图片做 0-10 分评估
ENABLE_T2I_TEST = False

# 文生图默认尺寸
T2I_IMAGE_SIZE = "1024x1024"
