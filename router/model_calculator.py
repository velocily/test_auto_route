# model_calculator.py
"""
从 model_benchmarks.json 读取各模型能力数据，生成 model_profiles.py。
"""
import json
import os
from typing import Dict, Any


def calculate_capability_from_benchmarks(benchmarks: Dict[str, float]) -> Dict[str, float]:
    """
    根据benchmark分数计算模型能力值

    计算公式（所有参与计算的分数都必须来自测试数据）：
    - intelligence = 0.35*mmlu + 0.40*bbh + 0.25*gsm8k
    - reasoning = 0.6*bbh + 0.4*gsm8k
    - communication = 0.6*secretary + 0.4*hellaswag
    - execution = 0.7*project_manager + 0.3*reasoning
    - long_context = longbench

    注：bbh = (bbh_semantic + bbh_math) / 2，若两者都有则取平均。
    """

    mmlu = benchmarks.get("mmlu", 0.5)
    gsm8k = benchmarks.get("gsm8k", 0.5)
    hellaswag = benchmarks.get("hellaswag", 0.5)
    longbench = benchmarks.get("longbench", 0.5)
    secretary_score = benchmarks.get("secretary", 0.5)
    pm_score = benchmarks.get("project_manager", 0.5)

    # bbh 取 semantic 和 math 的平均
    bbh_sem = benchmarks.get("bbh_semantic", None)
    bbh_math = benchmarks.get("bbh_math", None)
    if bbh_sem is not None and bbh_math is not None:
        bbh = (bbh_sem + bbh_math) / 2
    elif bbh_sem is not None:
        bbh = bbh_sem
    elif bbh_math is not None:
        bbh = bbh_math
    else:
        bbh = benchmarks.get("bbh", 0.5)

    # 计算能力值
    intelligence = 0.35 * mmlu + 0.40 * bbh + 0.25 * gsm8k
    reasoning = 0.6 * bbh + 0.4 * gsm8k
    communication = 0.6 * secretary_score + 0.4 * hellaswag
    execution = 0.7 * pm_score + 0.3 * reasoning
    long_context = longbench

    return {
        "intelligence": round(intelligence, 4),
        "reasoning": round(reasoning, 4),
        "communication": round(communication, 4),
        "execution": round(execution, 4),
        "long_context": round(long_context, 4)
    }


def load_benchmarks(file_path: str) -> Dict[str, Any]:
    """加载benchmark数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_model_profiles(benchmark_file: str) -> Dict[str, Any]:
    """
    从benchmark文件生成model_profiles配置
    支持两种格式：
      1. 新格式: { model: { benchmarks: {...}, efficiency: {...} } }
      2. 旧格式: { model: { benchmarks: {...}, efficiency: {...} } }
    """
    benchmarks_data = load_benchmarks(benchmark_file)

    models_config = {}

    for model_name, model_data in benchmarks_data.items():
        benchmarks = model_data.get("benchmarks", {})
        efficiency = model_data.get("efficiency", {})

        if not benchmarks:
            continue

        capabilities = calculate_capability_from_benchmarks(benchmarks)

        models_config[model_name] = {
            "capability": capabilities,
            "efficiency": {
                "tps": efficiency.get("tps", 50),
                "latency": efficiency.get("latency", 1.0),
                "concurrency": efficiency.get("concurrency", 10)
            }
        }

    return models_config


def save_model_profiles(models_config: Dict[str, Any], output_file: str):
    """
    保存模型配置到Python文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# model_profiles.py - 自动生成，请勿手动编辑\n")
        f.write("# 如需修改，请编辑 model_benchmarks.json 后重新运行 generate_model_profiles.py\n")
        f.write("#\n")
        f.write("# 计算公式：\n")
        f.write("# - intelligence = 0.35*mmlu + 0.40*bbh + 0.25*gsm8k\n")
        f.write("# - reasoning = 0.6*bbh + 0.4*gsm8k\n")
        f.write("# - communication = 0.6*secretary + 0.4*hellaswag\n")
        f.write("# - execution = 0.7*project_manager + 0.3*reasoning\n")
        f.write("# - long_context = longbench\n")
        f.write("# 注：bbh = (bbh_semantic + bbh_math) / 2\n\n")

        f.write("MODELS = {\n")

        for i, (model_name, config) in enumerate(models_config.items()):
            f.write(f'    "{model_name}": {{\n')
            f.write('        "capability": {\n')
            caps = config["capability"]
            f.write(f'            "intelligence": {caps["intelligence"]},\n')
            f.write(f'            "reasoning": {caps["reasoning"]},\n')
            f.write(f'            "communication": {caps["communication"]},\n')
            f.write(f'            "execution": {caps["execution"]},\n')
            f.write(f'            "long_context": {caps["long_context"]}\n')
            f.write('        },\n')
            f.write('        "efficiency": {\n')
            eff = config["efficiency"]
            f.write(f'            "tps": {eff["tps"]},\n')
            f.write(f'            "latency": {eff["latency"]},\n')
            f.write(f'            "concurrency": {eff["concurrency"]}\n')
            f.write('        }\n')
            f.write('    }')
            if i < len(models_config) - 1:
                f.write(',')
            f.write('\n')

        f.write("}\n")

    print(f"模型配置已保存到 {output_file}")


def print_comparison(benchmark_file: str):
    """打印模型能力对比"""
    models_config = generate_model_profiles(benchmark_file)

    print("\n" + "=" * 80)
    print("模型能力对比表")
    print("=" * 80)

    print(f"{'模型':<15} {'智力':<8} {'推理':<8} {'沟通':<8} {'执行':<8} {'长上下文':<10}")
    print("-" * 80)

    for model_name, config in models_config.items():
        cap = config["capability"]
        print(f"{model_name:<15} {cap['intelligence']:<8} {cap['reasoning']:<8} "
              f"{cap['communication']:<8} {cap['execution']:<8} {cap['long_context']:<10}")

    print("=" * 80)

    print("\n计算公式详情:")
    print("-" * 80)
    print("intelligence  = 0.35 * mmlu + 0.40 * bbh + 0.25 * gsm8k")
    print("reasoning     = 0.60 * bbh + 0.40 * gsm8k")
    print("communication = 0.60 * secretary + 0.40 * hellaswag")
    print("execution     = 0.70 * project_manager + 0.30 * reasoning")
    print("long_context  = longbench")
    print("bbh           = (bbh_semantic + bbh_math) / 2")
    print("=" * 80)


def validate_benchmarks(benchmark_file: str):
    """验证benchmark文件是否包含所有必要的字段"""
    required_domains = [
        "mmlu", "gsm8k", "hellaswag",
        "longbench", "secretary", "project_manager",
    ]
    # bbh 可以用 bbh_semantic + bbh_math 替代，不强制要求

    benchmarks_data = load_benchmarks(benchmark_file)

    print("\n验证 Benchmark 数据...")
    print("-" * 80)

    all_valid = True
    for model_name, model_data in benchmarks_data.items():
        benchmarks = model_data.get("benchmarks", {})
        missing = [req for req in required_domains if req not in benchmarks]
        # bbh 特殊处理
        if "bbh" not in benchmarks and "bbh_semantic" not in benchmarks and "bbh_math" not in benchmarks:
            missing.append("bbh (或 bbh_semantic + bbh_math)")

        if missing:
            print(f"[FAIL] {model_name} 缺少以下字段: {missing}")
            all_valid = False
        else:
            print(f"[OK] {model_name} 包含所有必要字段")

    if all_valid:
        print("\n所有模型数据完整，可以生成配置")
    else:
        print("\n请补充缺失的测试数据后再运行")

    print("-" * 80)
    return all_valid


if __name__ == "__main__":
    from config import BENCHMARK_FILE, MODEL_PROFILES_FILE

    if validate_benchmarks(BENCHMARK_FILE):
        models_config = generate_model_profiles(BENCHMARK_FILE)
        save_model_profiles(models_config, MODEL_PROFILES_FILE)
        print_comparison(BENCHMARK_FILE)
        print("\n完成！model_profiles.py 已更新。")
    else:
        print("\n请先补充完整的测试数据后再运行。")