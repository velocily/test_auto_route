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
    from whoengine import classify_and_select, classify_and_select_multimodal, get_router
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


def _get_running_models() -> set:
    """获取当前正在运行的模型名集合。

    来源：
      1. 手动配置的 model_routes（router/config.py）
      2. 自动发现的模型缓存（model_client._discovered_cache）

    路由时仅在这些模型中选择，避免选到未部署的模型。
    """
    try:
        from config import REMOTE_SERVER_CONFIG
        models = set(REMOTE_SERVER_CONFIG.get("model_routes", {}).keys())
        # 合并自动发现的模型
        try:
            from model_client import discover_remote_models
            discovery = discover_remote_models(force_refresh=False)
            models.update(discovery["models"].keys())
        except Exception:
            pass
        return models
    except Exception:
        return set()


def route(prompt: str) -> dict:
    """
    对用户输入进行路由决策。

    新流程 (WhoEngine):
        1. WhoEngine KNN 路由器判断 query 属于哪个 domain
        2. 直接查 model_benchmarks.json，选该 domain 分数最高的模型
        3. 仅在当前正在运行（model_routes 已注册）的模型中选择

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
        running_models = _get_running_models()
        result = classify_and_select(
            prompt, benchmark_data,
            running_models=running_models if running_models else None,
        )
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


def route_messages(messages: list) -> dict:
    """
    对用户输入（含多模态消息）进行路由决策。

    多模态判定流程：
        1. 调用 is_multimodal_message 检测是否为多模态任务
        2. 多模态任务：仅在具备对应能力类型（识图/生图）且正在运行的模型中选择
        3. 非多模态任务：走原有纯文本路由（route()）

    参数:
        messages : OpenAI 格式消息列表
                    - 纯文本: [{"role":"user","content":"你好"}]
                    - 多模态: [{"role":"user","content":[
                                 {"type":"text","text":"描述图片"},
                                 {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
                              ]}]

    返回:
        {
            "selected_model": "model-name",
            "score": 0.85,
            "is_multimodal": True/False,
            "task_analysis": { ... }
        }
    """
    if _USE_WHOENGINE:
        logger.debug("路由引擎: 使用 WhoEngine 多模态路由...")
        benchmark_data = _load_benchmark_data()
        running_models = _get_running_models()
        result = classify_and_select_multimodal(
            messages, benchmark_data,
            running_models=running_models if running_models else None,
        )
        analysis = result["task_analysis"]
        logger.debug(
            "路由引擎: 多模态=%s domain=%s → 选中模型: %s (得分 %.4f)",
            result.get("is_multimodal"),
            analysis.get("route_domain"),
            result["selected_model"],
            result["score"],
        )
        return result

    # ---------- Fallback: 提取文本走原有路由 ----------
    logger.debug("路由引擎: WhoEngine 不可用，回退到原有路由（不支持多模态）")
    prompt = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            prompt += content + " "
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    prompt += item.get("text", "") + " "
    result = route(prompt.strip())
    result["is_multimodal"] = False
    return result
