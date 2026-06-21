# router_engine.py
# ============================================================
# 路由引擎 (WhoEngine 版)
# 新路线: query → WhoEngine domain 分类 → best expert
# 保留原有接口，内部替换为 WhoEngine 实现
# ============================================================

import logging
import json
import os

logger = logging.getLogger(__name__)

# ---------- 尝试导入 WhoEngine 路由 ----------
try:
    from whoengine import classify_and_select, get_router
    _USE_WHOENGINE = True
except Exception as e:
    logger.warning("WhoEngine 路由加载失败 (%s)，回退到原有 task_classifier + scoring 路由", e)
    _USE_WHOENGINE = False

# ---------- 原有导入（作为 fallback）----------
if not _USE_WHOENGINE:
    from task_classifier import classify_task
    from scoring import select_model


# 缓存 benchmark 数据
_BENCHMARK_DATA = None

def _load_benchmark_data() -> dict:
    """加载 model_benchmarks.json"""
    global _BENCHMARK_DATA
    if _BENCHMARK_DATA is not None:
        return _BENCHMARK_DATA
    # 使用项目根目录下的 model_benchmarks.json
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "model_benchmarks.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            _BENCHMARK_DATA = json.load(f)
    else:
        _BENCHMARK_DATA = {}
    return _BENCHMARK_DATA


def route(prompt: str) -> dict:
    """
    对用户输入进行路由决策。

    新流程 (WhoEngine):
        1. WhoEngine KNN 路由器判断 query 属于哪个 domain
        2. 直接查 model_benchmarks.json，选该 domain 分数最高的模型

    返回格式和原来完全一致:
        {
            "selected_model": "your-model-name",
            "score": 0.8521,
            "task_analysis": {
                "task": "coding",
                "difficulty": 7,
                "need_reasoning": True,
                "need_long_context": False,
                "route_domain": "bbh",
                "route_confidence": 0.92,
                "route_domain_scores": {"mmlu": 0.1, ...}
            }
        }
    """
    if _USE_WHOENGINE:
        logger.debug("路由引擎: 使用 WhoEngine 路由...")
        benchmark_data = _load_benchmark_data()
        result = classify_and_select(prompt, benchmark_data)
        analysis = result["task_analysis"]
        logger.debug(
            "路由引擎: WhoEngine domain=%s (置信度 %.3f) → 选中模型: %s (得分 %.4f)",
            analysis.get("route_domain"),
            analysis.get("route_confidence", 0),
            result["selected_model"],
            result["score"],
        )
        return result

    # ---------- Fallback: 原有路由 ----------
    logger.debug("路由引擎: 使用原有 task_classifier + scoring 路由...")
    task_analysis = classify_task(prompt)
    model, score = select_model(task_analysis)
    return {
        "selected_model": model,
        "score": round(score, 4),
        "task_analysis": task_analysis,
    }
