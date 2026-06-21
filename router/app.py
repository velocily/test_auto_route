# app.py
# ============================================================
# 智能路由系统 —— FastAPI 应用主入口 (WhoEngine 版)
# 新流程：接收用户消息 → WhoEngine domain 分类 → 查 benchmark 选 best expert → 远程模型调用 → 返回响应
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from router_engine import route
from model_client import call_remote_model, get_model_url, stream_remote_model, close_http_clients
from config import ROUTER_CONFIG, REMOTE_SERVER_CONFIG

import time
import uuid
import re
import json
import logging

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

# ---------- FastAPI 应用 ----------
app = FastAPI(title="智能路由系统 (WhoEngine)", description="用户输入 → WhoEngine domain 分类 → 模型选型 → 远程调用")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """服务启动时预加载 WhoEngine 路由器"""
    logger.info("=" * 60)
    logger.info("智能路由系统 (WhoEngine) 启动中...")
    logger.info("远程服务器 : %s", REMOTE_SERVER_CONFIG.get("base_url", "（见 model_routes 中各模型 URL）"))
    logger.info("已注册模型 : %s", list(REMOTE_SERVER_CONFIG.get("model_routes", {}).keys()))

    # 预加载 WhoEngine 路由器（自动训练或加载缓存）
    logger.info("预加载 WhoEngine 路由器...")
    try:
        from whoengine import get_router
        get_router()
        logger.info("WhoEngine 路由器加载/训练完成。")
    except Exception as e:
        logger.warning("WhoEngine 路由器加载失败: %s，将回退到原有 task_classifier 路由", e)
        logger.info("预加载原有任务分类模型...")
        from task_classifier import load_model
        load_model()
        logger.info("任务分类模型加载完成。")

    mode = "测试模式（跳过远程调用）" if ROUTER_CONFIG.get("test_mode") else "正常模式"
    logger.info("运行模式   : %s", mode)
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理 HTTP 连接池"""
    logger.info("正在关闭 HTTP 连接池...")
    await close_http_clients()
    logger.info("HTTP 连接池已关闭。")

# ============================================================
# Pydantic 数据模型
# ============================================================

class ChatRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    selected_model: str
    score: float
    task_analysis: Dict[str, Any]
    routing_url: Optional[str] = None
    model_available: Optional[bool] = None


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str


class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]


# ============================================================
# 工具函数
# ============================================================

def _extract_user_message(messages: List[OpenAIMessage]) -> Optional[str]:
    """从消息列表中提取最后一条用户消息"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return None


def _build_openai_error_response(
    model_name: str,
    error_content: str,
) -> dict:
    """构造 OpenAI 格式的错误响应（返回原始 dict，避免 Pydantic 校验问题）"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": error_content},
                "finish_reason": "stop",
            }
        ],
    }


# ============================================================
# 核心接口：OpenAI 兼容的 /v1/chat/completions
# ============================================================

@app.post("/v1/chat/completions")
async def openai_compatible(req: OpenAIRequest):
    """
    OpenAI 兼容接口，供 Cherry Studio / 任意 OpenAI 客户端调用。

    完整流程：
    1. 提取用户消息
    2. 调用路由引擎 → 确定最佳模型
    3. 调用远程模型服务器 → 获取真实回复
    4. 返回 OpenAI 格式的响应
    """

    logger.info("=" * 60)
    logger.info(">>> 收到 /v1/chat/completions 请求")
    logger.info("    消息数量      : %d", len(req.messages))

    # ---------- Step 1: 提取用户消息 ----------
    user_message = _extract_user_message(req.messages)
    if not user_message:
        logger.error("    未找到用户消息，返回 400")
        raise HTTPException(status_code=400, detail="未找到用户消息")

    logger.info("    用户消息      : %s", user_message)

    # ---------- Step 1.5: 检查用户是否通过 @模型名 格式指定了模型 ----------
    # 格式：在消息开头写 @模型名（如 @model-name），后面跟正常问题
    # 示例："@model-name 写一个快排算法"
    model_override = None
    override_notice = ""
    model_routes: dict = REMOTE_SERVER_CONFIG.get("model_routes", {})

    override_match = re.match(r'^@([a-zA-Z0-9_.-]+)', user_message)
    if override_match:
        model_override = override_match.group(1)
        # 从消息中移除 @xxx 指令
        user_message = user_message[override_match.end():].strip()
        if not user_message:
            user_message = "你好"  # 防止空消息

        # 同步更新 req.messages 中最后一条用户消息的内容
        for msg in reversed(req.messages):
            if msg.role == "user":
                msg.content = user_message
                break

        logger.info("    检测到用户指定模型: %s", model_override)
        logger.info("    移除指令后的消息: %s", user_message)

        if model_override not in model_routes:
            # 模型不存在 → 回退到自动路由，并在回复中提示用户
            override_notice = (
                f"[系统提示] 您指定的模型 '{model_override}' 不存在，"
                f"已自动选择最优模型。\n"
                f"可用的模型: {', '.join(model_routes.keys())}\n\n"
            )
            logger.warning("    用户指定模型 '%s' 不存在，回退到自动路由", model_override)

    # ---------- Step 2: 路由决策 ----------
    logger.info("--- 开始路由决策 ---")

    if model_override and model_override in model_routes:
        # 用户指定了有效模型 → 跳过路由，直接使用
        selected_model = model_override
        score = 1.0
        task_analysis = {"task": "user_override", "difficulty": 0, "need_reasoning": False, "need_long_context": False}
        logger.info("    使用用户指定模型: %s（跳过路由）", selected_model)
    else:
        # 正常路由决策
        routing_result = route(user_message)
        selected_model: str = routing_result["selected_model"]
        score: float = routing_result["score"]
        task_analysis: dict = routing_result["task_analysis"]

        logger.info("    WhoEngine域   : %s (置信度 %.3f)", task_analysis.get("route_domain"), task_analysis.get("route_confidence", 0))
        logger.info("    任务分类  : %s", task_analysis.get("task"))
        logger.info("    难度      : %s/10", task_analysis.get("difficulty"))
        logger.info("    选中模型  : %s", selected_model)
        logger.info("    路由得分  : %.4f", score)
    logger.info("--- 路由决策完成 ---")

    # ---------- Step 3: 调用远程模型 / 测试模式 ----------

    # 获取路由 URL 和模型可用性（无论测试还是正常模式都需要）
    routing_url = get_model_url(selected_model)
    model_available = routing_url is not None

    # ========== 测试模式：跳过远程调用，返回路由详情 ==========
    if ROUTER_CONFIG.get("test_mode", False):
        logger.info("--- [测试模式] 跳过远程调用，返回路由详情 ---")
        logger.info("    路由 URL   : %s", routing_url if routing_url else "（模型未在服务器注册）")
        logger.info("    模型可用性 : %s", "可用" if model_available else "不可用")
        logger.info("<<< 测试模式完成")
        logger.info("=" * 60)

        # 组装 WhoEngine 相关信息（兼容旧字段 + 新增字段）
        route_domain = task_analysis.get('route_domain', 'N/A')
        route_conf = task_analysis.get('route_confidence', 0.0)
        route_scores = task_analysis.get('route_domain_scores', {})
        domain_scores_str = ""
        if route_scores:
            sorted_scores = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)
            domain_scores_str = " | ".join([f"{k}: {v:.3f}" for k, v in sorted_scores])

        # 路由详情（Token 级路由 + Entropy + Voting）
        routing_detail = task_analysis.get('routing_detail', {})
        routing_detail_str = ""
        if routing_detail:
            rd = routing_detail
            strategy = rd.get('strategy', '?')
            latency = rd.get('routing_latency_ms', 0)
            routing_detail_str = (
                f"\n"
                f"【路由策略】\n"
                f"  策略: {strategy}\n"
                f"  路由耗时: {latency} ms\n"
            )
            if strategy == 'majority_voting':
                routing_detail_str += (
                    f"  Token 总数: {rd.get('total_tokens', '?')}\n"
                    f"  Top-K Token: {rd.get('top_k_tokens', '?')}（取熵最低的K个高置信度Token）\n"
                    f"  Token 熵范围: [{rd.get('entropy_min', '?')}, {rd.get('entropy_max', '?')}]（均值 {rd.get('entropy_mean', '?')}）\n"
                )
                vote_counts = rd.get('vote_counts', {})
                if vote_counts:
                    vote_str = " | ".join([f"{k}: {v}票" for k, v in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)])
                    routing_detail_str += f"  投票结果: {vote_str}\n"
                top_domains = rd.get('top_token_domains', [])
                if top_domains:
                    routing_detail_str += f"  Top Token 预测: {top_domains[:10]}{'...' if len(top_domains) > 10 else ''}\n"
            routing_detail_str += (
                f"  Domain 分数: {domain_scores_str}\n"
            )

        mock_content = (
            f"{override_notice}"  # 模型不存在时的提示
            f"[测试模式] 路由决策结果 (WhoEngine 升级版)\n"
            f"\n"
            f"【WhoEngine Domain 分类】\n"
            f"  预测 Domain: {route_domain}（置信度 {route_conf:.3f}）\n"
            f"{routing_detail_str}"
            f"\n"
            f"【兼容信息】\n"
            f"  任务分类: {task_analysis.get('task')}（难度 {task_analysis.get('difficulty')}/10）\n"
            f"  需要推理: {'是' if task_analysis.get('need_reasoning') else '否'} | "
            f"需要长上下文: {'是' if task_analysis.get('need_long_context') else '否'}\n"
            f"\n"
            f"【模型选型】\n"
            f"  选中模型: {selected_model}（得分 {score:.4f}）\n"
            f"  路由 URL: {routing_url if routing_url else '（未配置 —— 该模型未在 config.py 的 model_routes 中注册）'}\n"
            f"  模型可用: {'是' if model_available else '否（如需接入，请在 config.py 的 model_routes 中添加该模型）'}\n"
            f"\n"
            f"原始输入: {user_message}\n"
            f"\n"
            f"提示: 将 config.py 中 ROUTER_CONFIG['test_mode'] 设为 False 即可切换到正常模式。"
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": mock_content},
                    "finish_reason": "stop",
                }
            ],
        }

    # ========== 正常模式：调用远程模型 ==========
    logger.info("--- 开始调用远程模型 ---")

    # 客户端是否请求流式
    client_wants_stream = req.stream if req.stream is not None else False

    # 构造发送给远程模型的消息（包含完整对话历史）
    remote_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]

    temp = req.temperature if req.temperature is not None else 0.7
    mt = req.max_tokens if req.max_tokens is not None else 2048

    if client_wants_stream:
        # ===== 流式模式：逐 token 透传 vLLM 的 SSE 流 =====
        async def stream_generator():
            # 若模型不存在回退，先发送提示通知
            if override_notice:
                notice_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                notice_created = int(time.time())
                notice_chunk = json.dumps({
                    "id": notice_id,
                    "object": "chat.completion.chunk",
                    "created": notice_created,
                    "model": selected_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }, ensure_ascii=False)
                yield f"data: {notice_chunk}\n\n"
                content_chunk = json.dumps({
                    "id": notice_id,
                    "object": "chat.completion.chunk",
                    "created": notice_created,
                    "model": selected_model,
                    "choices": [{"index": 0, "delta": {"content": override_notice}, "finish_reason": None}],
                }, ensure_ascii=False)
                yield f"data: {content_chunk}\n\n"

            async for sse_line in stream_remote_model(
                model_name=selected_model,
                messages=remote_messages,
                temperature=temp,
                max_tokens=mt,
            ):
                yield sse_line

        logger.info("<<< 流式传输开始")
        logger.info("=" * 60)
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )
    else:
        # ===== 非流式模式：完整请求-响应 =====
        result = call_remote_model(
            model_name=selected_model,
            messages=remote_messages,
            temperature=temp,
            max_tokens=mt,
            stream=False,
        )

        if result["success"]:
            remote_data = result["data"]
            remote_data["model"] = selected_model

            # 若模型不存在回退，在内容前添加提示
            if override_notice:
                choices = remote_data.get("choices", [])
                if choices:
                    original_content = choices[0].get("message", {}).get("content", "")
                    choices[0]["message"]["content"] = override_notice + original_content

            logger.info("<<< 请求完成，返回远程模型响应")
            logger.info("    返回内容（前200字符）: %s",
                        str(remote_data.get("choices", [{}])[0].get("message", {}).get("content", ""))[:200])
            logger.info("=" * 60)
            return JSONResponse(content=remote_data)
        else:
            logger.warning("<<< 远程调用失败，返回错误提示")
            logger.info("=" * 60)
            return _build_openai_error_response(selected_model, result.get("content", "未知错误"))


# ============================================================
# 模型列表接口
# ============================================================

@app.get("/v1/models")
async def list_models():
    """
    列出可用模型列表（OpenAI 兼容格式）。
    返回所有已注册的远程模型，供客户端（如 Cherry Studio）展示。
    """
    model_routes = REMOTE_SERVER_CONFIG.get("model_routes", {})
    models = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "router",
        }
        for model_id in model_routes
    ]
    logger.info("收到 /v1/models 请求 → 返回 %d 个模型", len(models))
    return {
        "object": "list",
        "data": models,
    }


@app.get("/v1/models/@mention")
async def list_models_for_mention():
    """
    【前端专用】列出所有可用的模型，供前端 @ 输入框弹出下拉菜单使用。

    返回格式（轻量级）:
    [
        {"id": "model-a",      "label": "model-a",      "description": ""},
        {"id": "model-b",      "label": "model-b",      "description": ""},
    ]

    前端接入：
        1. 用户在输入框输入 @ 时，前端调用此接口获取模型列表
        2. 弹出下拉菜单展示所有模型
        3. 用户选中后，在输入框中插入 "@模型名 " 文本
        4. 用户发送消息时，后端 app.py 自动解析 @xxx 指令并路由到对应模型
    """
    model_routes = REMOTE_SERVER_CONFIG.get("model_routes", {})
    models = [
        {
            "id": model_id,
            "label": model_id,
            "description": "",
        }
        for model_id in model_routes
    ]
    logger.info("收到 /v1/models/@mention 请求 → 返回 %d 个模型", len(models))
    return models


# ============================================================
# 路由分析接口（仅分析，不调用远程模型）
# ============================================================

@app.post("/route", response_model=RouteResponse)
def route_api(req: ChatRequest):
    """
    路由分析接口：仅执行任务分类和模型选型，不实际调用远程模型。
    返回选中的模型、得分、任务分析、以及远程调用 URL。
    """
    logger.info(f"收到 /route 请求: {req.prompt[:100]}...")

    if not req.prompt or len(req.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="prompt 不能为空")

    result = route(req.prompt)
    selected_model = result["selected_model"]

    # 附加远程路由信息
    routing_url = get_model_url(selected_model)
    model_available = routing_url is not None

    return RouteResponse(
        selected_model=selected_model,
        score=result["score"],
        task_analysis=result["task_analysis"],
        routing_url=routing_url,
        model_available=model_available,
    )


# ============================================================
# 任务分类接口
# ============================================================

@app.post("/classify")
def classify_only(req: ChatRequest):
    """仅执行任务分类，返回分类结果"""
    from task_classifier import classify_task

    logger.info(f"收到 /classify 请求: {req.prompt[:100]}...")
    result = classify_task(req.prompt)
    return {"task_analysis": result}


# ============================================================
# 基础接口
# ============================================================

@app.get("/")
def root():
    """服务状态页"""
    from task_classifier import is_model_loaded

    model_routes = REMOTE_SERVER_CONFIG.get("model_routes", {})
    return {
        "service": "智能路由系统",
        "version": "3.0.0",
        "status": "running",
        "mode": "模型路由 + 远程调用",
        "local_classifier_loaded": is_model_loaded(),
        "remote_server": REMOTE_SERVER_CONFIG.get("base_url", "（见 model_routes）"),
        "available_models": list(model_routes.keys()),
        "debug": ROUTER_CONFIG.get("enable_debug_log", True),
    }


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "healthy"}


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
    )
