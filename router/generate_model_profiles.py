# generate_model_profiles.py
"""
运行此脚本从 model_benchmarks.json 生成 model_profiles.py

使用方法:
    python generate_model_profiles.py

计算公式:
    intelligence  = 0.35 * mmlu + 0.40 * bbh + 0.25 * gsm8k
    reasoning     = 0.60 * bbh + 0.40 * gsm8k
    communication = 0.60 * secretary + 0.40 * hellaswag
    execution     = 0.70 * project_manager + 0.30 * reasoning
    long_context  = longbench
"""

from model_calculator import (
    validate_benchmarks,
    generate_model_profiles,
    save_model_profiles,
    print_comparison,
)
from config import BENCHMARK_FILE, MODEL_PROFILES_FILE

if __name__ == "__main__":
    # 验证数据完整性
    if validate_benchmarks(BENCHMARK_FILE):
        # 生成配置
        models_config = generate_model_profiles(BENCHMARK_FILE)

        # 保存到文件
        save_model_profiles(models_config, MODEL_PROFILES_FILE)

        # 打印对比表
        print_comparison(BENCHMARK_FILE)

        print("\n完成！model_profiles.py 已更新。")
    else:
        print("\n请先补充完整的测试数据后再运行。")