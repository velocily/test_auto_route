# model_profiles.py - 自动生成，请勿手动编辑
# 如需修改，请编辑 model_benchmarks.json 后重新运行 generate_model_profiles.py
#
# 计算公式：
# - intelligence = 0.35*mmlu + 0.40*bbh + 0.25*gsm8k
# - reasoning = 0.6*bbh + 0.4*gsm8k
# - communication = 0.6*secretary + 0.4*hellaswag
# - execution = 0.7*project_manager + 0.3*reasoning
# - long_context = longbench

# 注意：以下为示例数据，实际使用时请运行 generate_model_profiles.py
#       根据你的 model_benchmarks.json 自动生成

MODELS = {
    "model-a": {
        "capability": {
            "intelligence": 0.7584,
            "reasoning": 0.75,
            "communication": 0.907,
            "execution": 0.8725,
            "long_context": 0.833
        },
        "efficiency": {
            "tps": 200,
            "latency": 0.4,
            "concurrency": 15
        }
    },
    "model-b": {
        "capability": {
            "intelligence": 0.9,
            "reasoning": 0.72,
            "communication": 0.925,
            "execution": 0.8985,
            "long_context": 0.8
        },
        "efficiency": {
            "tps": 50,
            "latency": 0.8,
            "concurrency": 10
        }
    },
    "model-c": {
        "capability": {
            "intelligence": 0.9,
            "reasoning": 0.75,
            "communication": 0.907,
            "execution": 0.8725,
            "long_context": 0.833
        },
        "efficiency": {
            "tps": 80,
            "latency": 0.4,
            "concurrency": 5
        }
    }
}
