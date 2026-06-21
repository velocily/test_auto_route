# scoring.py
# ============================================================
# 模型评分与选择 (备用，当 WhoEngine 不可用时使用)
# ============================================================

from model_profiles import MODELS
import logging

logger = logging.getLogger(__name__)

def normalize_latency(x):
    return max(0, 1 - x / 3)

def normalize_tps(x):
    return min(x / 60, 1)

def normalize_concurrency(x):
    return min(x / 20, 1)

TASK_WEIGHTS = {

    "summary": {
        "communication": 0.4,
        "long_context": 0.3,
        "efficiency": 0.3
    },

    "pm": {
        "reasoning": 0.3,
        "execution": 0.4,
        "long_context": 0.3
    },

    "coding": {
        "reasoning": 0.5,
        "execution": 0.3,
        "efficiency": 0.2
    },

    "math": {
        "reasoning": 0.7,
        "intelligence": 0.3
    },

    "secretary": {
        "communication": 0.5,
        "execution": 0.3,
        "efficiency": 0.2
    },

    "chat": {
        "communication": 0.5,
        "efficiency": 0.5
    }
}

def compute_efficiency(eff):

    return (
        0.4 * normalize_tps(eff["tps"]) +
        0.3 * normalize_latency(eff["latency"]) +
        0.3 * normalize_concurrency(eff["concurrency"])
    )

def score_model(model_data, task_type):

    weights = TASK_WEIGHTS.get(
        task_type,
        TASK_WEIGHTS["chat"]
    )

    cap = model_data["capability"]

    score = 0

    for k, w in weights.items():

        if k == "efficiency":
            score += compute_efficiency(
                model_data["efficiency"]
            ) * w
        else:
            score += cap.get(k, 0) * w

    return score

def select_model(task_analysis):
    """
    原有接口，保留供 fallback 使用。
    IR3DE 路由不再调用此函数，但原系统的其他代码可直接复用。
    """
    best_model = None
    best_score = -1

    for model_name, model_data in MODELS.items():

        score = score_model(
            model_data,
            task_analysis["task"]
        )

        if task_analysis["difficulty"] >= 8:
            score += model_data["capability"]["reasoning"] * 0.3

        if score > best_score:
            best_score = score
            best_model = model_name

    return best_model, best_score


# ========== 新增：WhoEngine 直接查表路由（供 whoengine.py 使用）==========

def select_model_by_domain(domain: str, benchmark_data: dict) -> tuple:
    """
    WhoEngine 路由专用：给定 domain，直接查 benchmark_data 中该 domain 分数最高的模型。
    benchmark_data 格式同 model_benchmarks.json。
    """
    best_model = None
    best_score = -1.0
    for model_name, data in benchmark_data.items():
        score = data.get("benchmarks", {}).get(domain, 0.0)
        if score > best_score:
            best_score = score
            best_model = model_name
    if best_model is None:
        logger.warning("[WhoEngine] domain '%s' 未匹配到任何模型，回退到平均分最高", domain)
        best_model, best_score = _fallback_best_overall(benchmark_data)
    return best_model, best_score


def _fallback_best_overall(benchmark_data: dict) -> tuple:
    """计算各模型在所有 benchmark 上的平均分，选最高"""
    best_model = None
    best_avg = -1.0
    for model_name, data in benchmark_data.items():
        scores = list(data.get("benchmarks", {}).values())
        avg = sum(scores) / len(scores) if scores else 0
        if avg > best_avg:
            best_avg = avg
            best_model = model_name
    return best_model, best_avg
