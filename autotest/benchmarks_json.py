"""
============================================================================
model_benchmarks.json 更新模块
============================================================================
每次测试完成后自动更新汇总 JSON，按模型名索引。
同时生成 router 兼容的 benchmarks / efficiency 子结构。
"""
import json
import os
from datetime import datetime


# 题集显示名 → router domain 短名 映射
_BENCHMARK_TO_DOMAIN = {
    "mmlu-知识广度(30)":           "mmlu",
    "gsm8k-数学推理(10)":          "gsm8k",
    "hellaswag-常识推理(20)":      "hellaswag",
    "bbh-语义理解(10)":            "bbh_semantic",
    "bbh-数学推理(10)":            "bbh_math",
    "longbench-长上下文(10)":      "longbench",
    "职场角色测试-项目经理(20)":    "project_manager",
    "职场角色测试-秘书(20)":       "secretary",
}


def _collect_efficiency(eff_result):
    """从效率测试结果提取关键指标"""
    if not eff_result:
        return None

    single = eff_result.get("single_user", {})
    stable = eff_result.get("stable_result", {})

    return {
        "type": "efficiency",
        "single_ttft_ms": round(single.get("avg_ttft_ms", 0), 1),
        "single_throughput_tok_s": round(single.get("avg_throughput", 0), 2),
        "stable_concurrency": eff_result.get("stable_concurrency"),
        "stable_ttft_ms": round(stable.get("avg_ttft_ms", 0), 1) if stable else None,
        "stable_throughput_tok_s": round(stable.get("avg_throughput", 0), 2) if stable else None,
    }


def _build_router_benchmarks(all_results):
    """从 all_results 构建 router 兼容的 benchmarks 子对象"""
    router_benchmarks = {}
    for data in all_results:
        display = data["benchmark_display"]
        domain = _BENCHMARK_TO_DOMAIN.get(display)
        if not domain:
            continue
        questions = data.get("questions", [])
        results = data.get("results", [])
        total = len(results)
        if total == 0:
            continue
        is_sub = questions[0]["type"] in ("subjective", "long_context") if questions else False
        if is_sub:
            tscore = sum(r["score"] for r in results)
            router_benchmarks[domain] = round(tscore / total / 10, 4)  # 归一化到 0~1
        else:
            correct = sum(1 for r in results if r["score"] == 1)
            router_benchmarks[domain] = round(correct / total, 4)
    return router_benchmarks


def _build_router_efficiency(eff_result):
    """从效率测试结果构建 router 兼容的 efficiency 子对象"""
    if not eff_result:
        return None
    single = eff_result.get("single_user", {})
    return {
        "tps": round(single.get("avg_throughput", 0), 1),
        "latency": round(single.get("avg_ttft_ms", 0) / 1000, 2),  # ms → s
        "concurrency": eff_result.get("stable_concurrency") or 0,
    }


def build_benchmark_summary(all_results, eff_result=None):
    """从 all_results 和 eff_result 构建汇总字典。"""
    summary = {}
    summary["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for data in all_results:
        key = data["benchmark_display"]
        questions = data.get("questions", [])
        results = data.get("results", [])
        total = len(results)

        if not questions:
            continue

        is_sub = questions[0]["type"] in ("subjective", "long_context")

        if is_sub:
            tscore = sum(r["score"] for r in results)
            avg = tscore / total if total > 0 else 0
            summary[key] = {
                "type": "subjective",
                "total": total,
                "total_score": tscore,
                "avg_score": round(avg, 2),
            }
        else:
            correct = sum(1 for r in results if r["score"] == 1)
            acc = correct / total if total > 0 else 0
            summary[key] = {
                "type": "objective",
                "total": total,
                "correct": correct,
                "accuracy": round(acc, 4),
            }

    if eff_result:
        eff_summary = _collect_efficiency(eff_result)
        if eff_summary:
            summary["模型效率测试"] = eff_summary

    # ========== 附加 router 兼容子结构 ==========
    summary["benchmarks"] = _build_router_benchmarks(all_results)
    eff_router = _build_router_efficiency(eff_result)
    if eff_router:
        summary["efficiency"] = eff_router

    return summary


def update_benchmarks_json(filepath, model_name, summary):
    """更新 JSON 文件。"""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {}
    else:
        data = {}

    data[model_name] = summary

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  -> 已更新: {filepath}")
    return filepath
