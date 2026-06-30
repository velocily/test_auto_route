"""
============================================================================
model_benchmarks.json 更新模块
============================================================================
每次测试完成后自动更新汇总 JSON，按模型名索引。
同时生成 router 兼容的 benchmarks / efficiency / multimodal 子结构。

多模态能力分类（multimodal 子结构按能力类型分组）：
  - vision_recognition : 视觉识图能力（理解已有图片），含 chart_qa/text_vqa 等
  - image_generation   : 视觉生图能力（生成新图片），含 t2i
  - （未来可扩展 audio_recognition / audio_generation / video_recognition 等）
"""
import json
import os
from datetime import datetime


# ========== 多模态能力类型常量 ==========
# 能力类型 → 中文说明（用于文档和日志）
CAPABILITY_TYPES = {
    "vision_recognition": "视觉识图（理解已有图片）",
    "image_generation":   "视觉生图（生成新图片）",
    # 未来扩展：
    # "audio_recognition":  "听觉识别（理解已有音频）",
    # "audio_generation":   "听觉生成（生成新音频）",
    # "video_recognition":  "视频理解（理解已有视频）",
}

# ========== 能力状态常量 ==========
# 用于明确标记模型对某能力是否具备，避免"未测试"与"不支持"混淆
CAPABILITY_STATUS_SUPPORTED = "supported"      # 已测试，模型具备该能力
CAPABILITY_STATUS_UNSUPPORTED = "unsupported"  # 已探测，模型/接口不支持该能力
CAPABILITY_STATUS_NOT_TESTED = "not_tested"    # 未测试（默认）

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

# 视觉识图题集显示名 → router domain 短名 映射（归入 vision_recognition 能力）
_VISION_RECOGNITION_BENCHMARK_TO_DOMAIN = {
    "chartqa-图表理解(20)":    "chart_qa",
    "textvqa-文字识别(20)":    "text_vqa",
    "mathvista-视觉数学(20)": "math_vista",
    "vqa-视觉问答(30)":        "vqa",
    "mmmu-多模态理解(30)":     "mmmu",
}

# 视觉生图题集显示名 → router domain 短名 映射（归入 image_generation 能力）
_IMAGE_GENERATION_BENCHMARK_TO_DOMAIN = {
    "t2i-文生图(50)":          "t2i",
}


def _collect_efficiency(eff_result):
    """从效率测试结果提取关键指标"""
    if not eff_result:
        return None

    single = eff_result.get("single_user", {})
    stable = eff_result.get("stable_result", {})

    # stable_throughput_tok_s 使用"每通道平均吞吐"：
    #   总 tokens / (并发数 × 总耗时)
    # 这反映每个通道在并发场景下的真实产出，而非"每请求视角吞吐"
    # （后者会被并行加速放大，不反映每通道实际效率）
    stable_throughput = None
    if stable:
        stable_throughput = stable.get("per_channel_throughput")
        if stable_throughput is None:
            # 兼容旧数据：若无 per_channel_throughput，回退到 avg_throughput
            stable_throughput = stable.get("avg_throughput", 0)

    return {
        "type": "efficiency",
        "single_ttft_ms": round(single.get("avg_ttft_ms", 0), 1),
        "single_throughput_tok_s": round(single.get("avg_throughput", 0), 2),
        "stable_concurrency": eff_result.get("stable_concurrency"),
        "stable_ttft_ms": round(stable.get("avg_ttft_ms", 0), 1) if stable else None,
        "stable_throughput_tok_s": round(stable_throughput, 2) if stable else None,
    }


def _build_router_benchmarks(all_results):
    """从 all_results 构建 router 兼容的 benchmarks 子对象（仅纯文本题集）"""
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


def _build_router_vision_recognition(multimodal_results):
    """从 multimodal_results 构建 router 兼容的视觉识图能力子对象。
    归入 multimodal.vision_recognition 分组。
    若无识图结果则返回 None。
    """
    if not multimodal_results:
        return None
    router_vr = {}
    for data in multimodal_results:
        display = data["benchmark_display"]
        domain = _VISION_RECOGNITION_BENCHMARK_TO_DOMAIN.get(display)
        if not domain:
            continue
        results = data.get("results", [])
        total = len(results)
        if total == 0:
            continue
        # 识图题集均为客观选择题
        correct = sum(1 for r in results if r["score"] == 1)
        router_vr[domain] = round(correct / total, 4)
    return router_vr if router_vr else None


def _build_router_image_generation(t2i_results):
    """从 t2i_results 构建 router 兼容的视觉生图能力子对象。
    归入 multimodal.image_generation 分组。
    文生图得分为 0-10，归一化到 0~1。
    若无文生图结果则返回 None。
    """
    if not t2i_results:
        return None
    router_ig = {}
    for data in t2i_results:
        display = data["benchmark_display"]
        domain = _IMAGE_GENERATION_BENCHMARK_TO_DOMAIN.get(display)
        if not domain:
            continue
        results = data.get("results", [])
        total = len(results)
        if total == 0:
            continue
        # 文生图为打分制（0-10），归一化到 0~1
        score_sum = sum(r["score"] for r in results)
        router_ig[domain] = round(score_sum / total / 10, 4)
    return router_ig if router_ig else None


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


def build_benchmark_summary(all_results, eff_result=None, multimodal_results=None,
                            t2i_results=None, capability_status=None):
    """从 all_results、eff_result、multimodal_results、t2i_results 构建汇总字典。

    参数:
        capability_status : dict，能力类型 → 状态（supported/unsupported/not_tested）
                           用于明确标记模型对某能力是否具备，避免"未测试"与"不支持"混淆
    """
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
        is_visual = questions[0]["type"] == "visual_multiple_choice"
        is_t2i = questions[0]["type"] == "text_to_image"

        if is_sub:
            tscore = sum(r["score"] for r in results)
            avg = tscore / total if total > 0 else 0
            # 统一为正确率：10 分制 → 0~1
            summary[key] = {
                "type": "subjective",
                "total": total,
                "total_score": tscore,
                "avg_score": round(avg, 2),
                "accuracy": round(avg / 10, 4),
            }
        elif is_t2i:
            tscore = sum(r["score"] for r in results)
            avg = tscore / total if total > 0 else 0
            # 统一为正确率：10 分制 → 0~1
            summary[key] = {
                "type": "text_to_image",
                "total": total,
                "total_score": tscore,
                "avg_score": round(avg, 2),
                "accuracy": round(avg / 10, 4),
            }
        elif is_visual:
            correct = sum(1 for r in results if r["score"] == 1)
            acc = correct / total if total > 0 else 0
            summary[key] = {
                "type": "visual_objective",
                "total": total,
                "correct": correct,
                "accuracy": round(acc, 4),
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

    # 多模态能力子结构（按能力类型分组，router 用于多模态任务路由）
    # 结构：multimodal: { vision_recognition: {...}, image_generation: {...} }
    vision_recog = _build_router_vision_recognition(multimodal_results)
    image_gen = _build_router_image_generation(t2i_results)
    # 合并识图 + 生图到 multimodal 子结构（按能力类型分组）
    merged_mm = {}
    if vision_recog:
        merged_mm["vision_recognition"] = vision_recog
    if image_gen:
        merged_mm["image_generation"] = image_gen
    if merged_mm:
        summary["multimodal"] = merged_mm

    # ========== 能力状态标记 ==========
    # 明确记录每个能力类型的状态，路由器据此区分"未测试"与"不支持"
    # - supported  : 已测试，模型具备该能力（multimodal 子结构中有得分）
    # - unsupported : 已探测，模型/接口不支持该能力（不可路由到此模型）
    # - not_tested  : 未测试（默认，路由器可回退处理）
    if capability_status:
        summary["capability_status"] = dict(capability_status)

    return summary


def update_benchmarks_json(filepath, model_name, summary):
    """增量更新 JSON 文件（不覆盖其他模块的已有结果）。

    合并策略：
      - last_updated                : 始终用新值覆盖
      - 各题集显示名键（如 mmlu-知识广度(30)）: 仅当新 summary 中存在时覆盖
      - 模型效率测试                  : 仅当新 summary 中存在时覆盖
      - benchmarks (dict)            : 按子键合并（新值覆盖同 domain，保留未测 domain）
      - efficiency (dict)            : 仅当新 summary 中存在时整体覆盖
      - multimodal (dict)            : 按能力类型子分组合并
        （新测识图不覆盖已有生图，反之亦然）
      - capability_status (dict)     : 按子键合并（新值覆盖同能力类型状态）

    这样用户可分模块测试：先测纯文本，再测视觉识图，再测文生图，
    每次只更新对应模块，已有结果保留。
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {}
    else:
        data = {}

    existing = data.get(model_name, {})

    # 需要按子键合并的 dict 字段
    _MERGE_SUBKEYS = ("benchmarks", "multimodal", "capability_status")
    # 整体覆盖的 dict 字段（仅当新值存在时覆盖）
    _REPLACE_KEYS = ("efficiency",)
    # 各题集显示名键 + 模型效率测试：仅当新值存在时覆盖
    # （这些键是动态的，在循环中按"新值存在则覆盖"处理）

    merged = dict(existing)  # 以旧数据为基底

    for key, new_val in summary.items():
        if new_val is None:
            continue
        if key in _MERGE_SUBKEYS:
            # 按子键合并
            old_val = merged.get(key, {})
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                merged_sub = dict(old_val)
                # multimodal 需要再深一层合并（按能力类型分组）
                if key == "multimodal":
                    for cap, cap_new in new_val.items():
                        cap_old = merged_sub.get(cap, {})
                        if isinstance(cap_old, dict) and isinstance(cap_new, dict):
                            merged_sub[cap] = {**cap_old, **cap_new}
                        else:
                            merged_sub[cap] = cap_new
                else:
                    merged_sub.update(new_val)
                merged[key] = merged_sub
            else:
                merged[key] = new_val
        else:
            # last_updated / 各题集键 / efficiency / 模型效率测试 等：直接覆盖
            merged[key] = new_val

    data[model_name] = merged

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  -> 已增量更新: {filepath}（模型 {model_name}）")
    return filepath
