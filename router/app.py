# app.py
# ============================================================
# 智能路由系统 —— FastAPI 应用主入口 (WhoEngine 版)
# 新流程：接收用户消息 → WhoEngine domain 分类 → 查 benchmark 选 best expert → 远程模型调用 → 返回响应
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union

import os
import asyncio

# 优先使用本地 HF 缓存，避免联网检查更新导致启动失败/超时
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from router_engine import route, route_messages
from model_client import (
    call_remote_model, call_remote_model_async, get_model_url,
    stream_remote_model, close_http_clients,
    discover_remote_models, get_model_api_key,
)
from config import ROUTER_CONFIG, REMOTE_SERVER_CONFIG

# 集中式路由参数管理
try:
    import routing_params
    _HAS_PARAMS = True
except Exception as e:
    _HAS_PARAMS = False

import time
import uuid
import re
import json
import logging
import sys
import subprocess
import threading

# 项目根目录（用于定位文档文件）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))

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
    eps_count = len(REMOTE_SERVER_CONFIG.get("remote_endpoints", []))
    mr_count = len(REMOTE_SERVER_CONFIG.get("model_routes", {}))
    logger.info("远程端点 : %d 个（自动发现）", eps_count)
    logger.info("手动路由 : %d 个 %s", mr_count, list(REMOTE_SERVER_CONFIG.get("model_routes", {}).keys()))
    # 启动时若配置了 remote_endpoints，预触发一次自动发现
    if eps_count > 0:
        try:
            result = discover_remote_models(force_refresh=True)
            logger.info("启动自动发现完成：发现 %d 个模型", len(result["models"]))
        except Exception as e:
            logger.warning("启动自动发现失败: %s", e)

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
    # content 支持两种格式：
    #   1. 纯文本: "你好"
    #   2. 多模态: [{"type":"text","text":"描述图片"},
    #              {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
    content: Union[str, List[Dict[str, Any]]]


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

def _is_model_known(model_name: str) -> bool:
    """检查模型是否已知（有可调用 URL 或在 benchmark 中有测试记录）。

    用于强制路由的模型校验：测试模式下 benchmark 模型可能没有 URL，
    但仍应允许被勾选为强制路由目标。
    """
    if get_model_url(model_name) is not None:
        return True
    try:
        import router_engine
        data = router_engine._load_benchmark_data()
        return model_name in data
    except Exception:
        return False


def _extract_user_message(messages: List[OpenAIMessage]) -> Optional[str]:
    """从消息列表中提取最后一条用户消息的文本部分（用于日志和 @ 指令解析）"""
    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            elif isinstance(msg.content, list):
                # 多模态消息：拼接所有 text 项
                texts = []
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return " ".join(texts) if texts else None
    return None


def _messages_to_dicts(messages: List[OpenAIMessage]) -> list:
    """将 OpenAIMessage 列表转换为纯 dict 列表（保留多模态 content 结构）"""
    return [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]


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
    fallback_notice = ""
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

        # 检查模型是否可用（model_routes 或自动发现的模型）
        if get_model_url(model_override) is None:
            # 模型不存在 → 回退到自动路由，并在回复中提示用户
            # 收集所有可用模型名（手动 + 自动发现）
            available_models = list(model_routes.keys())
            try:
                discovery = discover_remote_models(force_refresh=False)
                available_models.extend(discovery["models"].keys())
            except Exception:
                pass
            override_notice = (
                f"[系统提示] 您指定的模型 '{model_override}' 不存在，"
                f"已自动选择最优模型。\n"
                f"可用的模型: {', '.join(available_models) or '（无）'}\n\n"
            )
            logger.warning("    用户指定模型 '%s' 不存在，回退到自动路由", model_override)

    # ---------- Step 2: 路由决策 ----------
    logger.info("--- 开始路由决策 ---")

    if model_override and get_model_url(model_override) is not None:
        # 用户指定了有效模型 → 跳过路由，直接使用（最高优先级）
        selected_model = model_override
        score = 1.0
        is_multimodal = False
        task_analysis = {"task": "user_override", "difficulty": 0, "need_reasoning": False, "need_long_context": False}
        logger.info("    使用用户指定模型: %s（跳过路由）", selected_model)
    elif ROUTER_CONFIG.get("forced_model") and _is_model_known(ROUTER_CONFIG["forced_model"]):
        # 前端勾选了强制路由模型 → 跳过路由，直接使用（第二优先级）
        # 注意：不要求模型有可调用 URL，测试模式下 benchmark 模型也可被强制路由
        selected_model = ROUTER_CONFIG["forced_model"]
        score = 1.0
        is_multimodal = False
        task_analysis = {"task": "forced_route", "difficulty": 0, "need_reasoning": False, "need_long_context": False}
        logger.info("    使用强制路由模型: %s（跳过路由）", selected_model)
    else:
        # 正常路由决策（支持多模态消息识别）
        # 使用线程池执行路由推理，避免阻塞事件循环（KNN 推理是 CPU 密集型操作）
        messages_dicts = _messages_to_dicts(req.messages)
        routing_result = await asyncio.to_thread(route_messages, messages_dicts)
        selected_model: str = routing_result["selected_model"]
        score: float = routing_result["score"]
        task_analysis: dict = routing_result["task_analysis"]
        is_multimodal = routing_result.get("is_multimodal", False)
        task_kind = routing_result.get("task_kind", "text")

        logger.info("    多模态任务: %s", "是" if is_multimodal else "否")
        if is_multimodal:
            kind_zh = {"image_recognition": "识图", "image_generation": "生图"}.get(task_kind, task_kind)
            logger.info("    多模态类型: %s", kind_zh)
        logger.info("    WhoEngine域   : %s (置信度 %.3f)", task_analysis.get("route_domain"), task_analysis.get("route_confidence", 0))
        logger.info("    任务分类  : %s", task_analysis.get("task"))
        logger.info("    难度      : %s/10", task_analysis.get("difficulty"))
        logger.info("    选中模型  : %s", selected_model)
        logger.info("    路由得分  : %.4f", score)
    logger.info("--- 路由决策完成 ---")

    # ---------- Step 2.5: 路由失败防护 ----------
    # 当所有模型都未测试或无可用模型时，selected_model 可能为 None
    if not selected_model:
        logger.warning("    路由未选中任何模型（可能所有模型都未测试或未配置）")
        # 尝试从已发现/已注册模型中取第一个作为兜底
        fallback_models = list(REMOTE_SERVER_CONFIG.get("model_routes", {}).keys())
        try:
            discovery = discover_remote_models(force_refresh=False)
            fallback_models.extend(discovery["models"].keys())
        except Exception:
            pass
        if fallback_models:
            selected_model = sorted(fallback_models)[0]
            logger.info("    兜底选择第一个可用模型: %s", selected_model)
            score = 0.0
            task_analysis = task_analysis or {"task": "fallback", "difficulty": 0}
            fallback_notice = f"[系统提示] 当前无已测试模型匹配，已兜底路由到: {selected_model}\n\n"
        else:
            # 真的没有任何可用模型
            logger.error("    无任何可用模型，返回错误")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "未检测到模型：请先配置远程端点（URL+Key）或手动模型路由，"
                                   "并确保至少检测到一个模型。",
                        "type": "no_model_available",
                    }
                }
            )

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
            f"{fallback_notice}"  # 兜底路由提示
            f"路由选择：{selected_model}\n"
            f"[测试模式] 路由决策结果 (WhoEngine 升级版)\n"
            f"\n"
            f"【任务类型】\n"
            f"  多模态任务: {'是（仅多模态模型参与路由）' if is_multimodal else '否（纯文本路由）'}\n"
        )
        if is_multimodal:
            kind_zh = {"image_recognition": "识图（视觉理解）", "image_generation": "生图（文生图）"}.get(task_kind, task_kind)
            mock_content += f"  多模态类型: {kind_zh}\n"
        mock_content += (
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
    # 正常模式下，如果选中的模型没有可调用 URL，返回错误
    if not model_available:
        logger.error("    选中的模型 '%s' 没有可调用的 API 地址", selected_model)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"未检测到模型：选中的模型 '{selected_model}' 没有可调用的 API 地址，"
                               f"请在路由配置中添加远程端点（URL+Key）或手动模型路由。",
                    "type": "no_model_available",
                }
            }
        )

    logger.info("--- 开始调用远程模型 ---")

    # 客户端是否请求流式
    client_wants_stream = req.stream if req.stream is not None else False

    # 构造发送给远程模型的消息（包含完整对话历史，保留多模态 content 结构）
    remote_messages = _messages_to_dicts(req.messages)

    temp = req.temperature if req.temperature is not None else 0.7
    mt = req.max_tokens if req.max_tokens is not None else 2048

    if client_wants_stream:
        # ===== 流式模式：逐 token 透传 vLLM 的 SSE 流 =====
        async def stream_generator():
            # 若有系统提示（模型不存在回退/兜底路由），先发送提示通知
            system_notice = override_notice + fallback_notice
            if system_notice:
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
                    "choices": [{"index": 0, "delta": {"content": system_notice}, "finish_reason": None}],
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
        # 使用 async 版本避免阻塞事件循环，支持多用户并发
        result = await call_remote_model_async(
            model_name=selected_model,
            messages=remote_messages,
            temperature=temp,
            max_tokens=mt,
            stream=False,
        )

        if result["success"]:
            remote_data = result["data"]
            remote_data["model"] = selected_model

            # 若有系统提示（模型不存在回退/兜底路由），在内容前添加提示
            system_notice = override_notice + fallback_notice
            if system_notice:
                choices = remote_data.get("choices", [])
                if choices:
                    original_content = choices[0].get("message", {}).get("content", "")
                    choices[0]["message"]["content"] = system_notice + original_content

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
# 路由参数调控台（可视化界面 + REST API）
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """路由参数可视化调控台 HTML 页面"""
    html_path = os.path.join(_ROUTER_DIR, "static", "dashboard.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="dashboard.html 不存在")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/params")
def api_get_params():
    """获取所有路由参数当前值"""
    if not _HAS_PARAMS:
        raise HTTPException(status_code=500, detail="routing_params 模块未加载")
    return routing_params.get_all_params()


@app.get("/api/params/meta")
def api_get_params_meta():
    """获取所有参数的元数据（含范围、默认值、说明），供前端渲染滑块"""
    if not _HAS_PARAMS:
        raise HTTPException(status_code=500, detail="routing_params 模块未加载")
    return routing_params.get_param_meta()


@app.post("/api/params")
def api_set_params(updates: Dict[str, Any]):
    """批量更新路由参数（即时生效，并记录到 params_changes.log）"""
    if not _HAS_PARAMS:
        raise HTTPException(status_code=500, detail="routing_params 模块未加载")
    try:
        routing_params.set_params(updates)
        logger.info("路由参数已更新并记录日志：%s", list(updates.keys()))
        return {"ok": True, "updated": list(updates.keys()), "current": routing_params.get_all_params(), "logged": True}
    except KeyError as e:
        return {"ok": False, "error": f"未知参数: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/params/preset")
def api_get_preset():
    """获取当前预设方案名"""
    if not _HAS_PARAMS:
        raise HTTPException(status_code=500, detail="routing_params 模块未加载")
    return {"current": routing_params.current_preset(), "available": list(routing_params.get_presets().keys())}


@app.post("/api/params/preset")
def api_apply_preset(body: Dict[str, Any]):
    """应用预设方案"""
    if not _HAS_PARAMS:
        raise HTTPException(status_code=500, detail="routing_params 模块未加载")
    preset = body.get("preset")
    if not preset:
        return {"ok": False, "error": "缺少 preset 字段"}
    try:
        routing_params.apply_preset(preset)
        logger.info("已应用预设方案：%s", preset)
        return {"ok": True, "preset": preset, "current": routing_params.get_all_params()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/params/reset")
def api_reset_params():
    """重置为默认预设"""
    if not _HAS_PARAMS:
        raise HTTPException(status_code=500, detail="routing_params 模块未加载")
    routing_params.reset_params()
    logger.info("路由参数已重置为默认预设")
    return {"ok": True, "current": routing_params.get_all_params()}


# ============================================================
# 路由服务配置接口（test_mode / model_routes / verify_ssl 等）
# ------------------------------------------------------------
# 这些配置存储在 config.py 的 ROUTER_CONFIG / REMOTE_SERVER_CONFIG 中，
# 运行时修改内存中的字典即可即时生效（无需重启服务），但重启后会恢复
# config.py 中的原始值。如需持久化，请同时修改 config.py。
# ============================================================

_route_config_lock = threading.Lock()


@app.get("/api/route/config")
def api_route_config_get():
    """获取当前路由服务配置（test_mode、remote_endpoints、model_routes、verify_ssl、request_timeout）"""
    with _route_config_lock:
        # 复制 remote_endpoints（避免外部修改内存）
        eps = []
        for ep in REMOTE_SERVER_CONFIG.get("remote_endpoints", []):
            eps.append({
                "url": ep.get("url", "") if isinstance(ep, dict) else "",
                "api_key": ep.get("api_key", "") if isinstance(ep, dict) else "",
            })
        return {
            "test_mode": bool(ROUTER_CONFIG.get("test_mode", False)),
            "verify_ssl": bool(REMOTE_SERVER_CONFIG.get("verify_ssl", True)),
            "request_timeout": int(REMOTE_SERVER_CONFIG.get("request_timeout", 120)),
            "remote_endpoints": eps,
            "model_routes": dict(REMOTE_SERVER_CONFIG.get("model_routes", {})),
        }


@app.post("/api/route/config")
def api_route_config_set(body: dict):
    """更新路由服务配置（内存级，即时生效，重启后恢复 config.py 原值）

    可选字段：
      - test_mode: bool           是否测试模式（跳过远程调用）
      - verify_ssl: bool          是否校验 SSL 证书
      - request_timeout: int     请求超时秒数
      - remote_endpoints: list   远程端点列表 [{"url": str, "api_key": str}]
      - model_routes: dict        模型名 → API URL 映射（手动配置，向后兼容）
    """
    with _route_config_lock:
        changed = []
        if "test_mode" in body:
            ROUTER_CONFIG["test_mode"] = bool(body["test_mode"])
            changed.append("test_mode")
        if "verify_ssl" in body:
            REMOTE_SERVER_CONFIG["verify_ssl"] = bool(body["verify_ssl"])
            changed.append("verify_ssl")
        if "request_timeout" in body:
            try:
                t = int(body["request_timeout"])
                if t > 0:
                    REMOTE_SERVER_CONFIG["request_timeout"] = t
                    changed.append("request_timeout")
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="request_timeout 必须是正整数")
        if "remote_endpoints" in body:
            eps = body["remote_endpoints"]
            if not isinstance(eps, list):
                raise HTTPException(status_code=400, detail="remote_endpoints 必须是列表")
            # 清理：去除空 URL 的项
            clean_eps = []
            for ep in eps:
                if not isinstance(ep, dict):
                    continue
                url = (ep.get("url") or "").strip()
                api_key = (ep.get("api_key") or "").strip()
                if url:
                    clean_eps.append({"url": url, "api_key": api_key})
            REMOTE_SERVER_CONFIG["remote_endpoints"] = clean_eps
            changed.append("remote_endpoints")
        if "model_routes" in body:
            mr = body["model_routes"]
            if not isinstance(mr, dict):
                raise HTTPException(status_code=400, detail="model_routes 必须是字典")
            # 清理：去除空键空值
            clean_mr = {}
            for k, v in mr.items():
                k = (k or "").strip()
                v = (v or "").strip()
                if k and v:
                    clean_mr[k] = v
            REMOTE_SERVER_CONFIG["model_routes"] = clean_mr
            changed.append("model_routes")

    mode = "测试模式（跳过远程调用）" if ROUTER_CONFIG.get("test_mode") else "正常模式"
    logger.info("路由服务配置已更新（内存级）: %s → 运行模式: %s, "
                "remote_endpoints: %d 个, model_routes: %d 个",
                ", ".join(changed) or "无变更", mode,
                len(REMOTE_SERVER_CONFIG.get("remote_endpoints", [])),
                len(REMOTE_SERVER_CONFIG.get("model_routes", {})))
    # 返回前复制 remote_endpoints
    eps_out = []
    for ep in REMOTE_SERVER_CONFIG.get("remote_endpoints", []):
        eps_out.append({"url": ep.get("url", ""), "api_key": ep.get("api_key", "")})
    return {
        "ok": True,
        "changed": changed,
        "current": {
            "test_mode": bool(ROUTER_CONFIG.get("test_mode", False)),
            "verify_ssl": bool(REMOTE_SERVER_CONFIG.get("verify_ssl", True)),
            "request_timeout": int(REMOTE_SERVER_CONFIG.get("request_timeout", 120)),
            "remote_endpoints": eps_out,
            "model_routes": dict(REMOTE_SERVER_CONFIG.get("model_routes", {})),
        },
    }


@app.post("/api/route/discover")
def api_route_discover():
    """触发远程模型自动发现。

    遍历 REMOTE_SERVER_CONFIG["remote_endpoints"]，对每个端点调用
    GET {url}/models 检测可用模型，结果缓存到内存中供路由使用。

    返回:
      {
        "ok": True,
        "models": [model_name, ...],     # 发现的模型名列表
        "errors": [str, ...],             # 检测失败的端点信息
        "endpoints_count": int,           # 配置的端点数
        "cached": bool,                   # 是否使用了缓存
      }
    """
    result = discover_remote_models(force_refresh=True)
    return {
        "ok": True,
        "models": sorted(result["models"].keys()),
        "errors": result["errors"],
        "endpoints_count": result["endpoints_count"],
        "cached": result["cached"],
    }


@app.get("/api/route/forced-model")
def api_forced_model_get():
    """获取当前强制路由模型设置"""
    return {
        "forced_model": ROUTER_CONFIG.get("forced_model", ""),
        "active": bool(ROUTER_CONFIG.get("forced_model", "")),
    }


@app.post("/api/route/forced-model")
def api_forced_model_set(body: dict):
    """设置或清除强制路由模型

    body: {"model": "model-name"} 设置强制路由
    body: {"model": ""} 清除强制路由，恢复自动路由
    """
    model = (body.get("model") or "").strip() if isinstance(body, dict) else ""
    if model:
        # 验证模型是否已知（有可调用 URL 或在 benchmark 中有测试记录）
        if not _is_model_known(model):
            raise HTTPException(status_code=400, detail=f"模型 '{model}' 不可用（未在路由配置或测试记录中找到）")
        ROUTER_CONFIG["forced_model"] = model
        logger.info("强制路由已设置: %s", model)
    else:
        ROUTER_CONFIG["forced_model"] = ""
        logger.info("强制路由已清除，恢复自动路由")
    return {
        "ok": True,
        "forced_model": ROUTER_CONFIG.get("forced_model", ""),
        "active": bool(ROUTER_CONFIG.get("forced_model", "")),
    }


@app.get("/api/route/status")
def api_route_status():
    """返回路由服务当前运行状态（模式、已注册模型、端口）"""
    # 统计已发现模型数（从缓存读取，不触发网络请求）
    discovery = discover_remote_models(force_refresh=False)
    discovered_count = len(discovery["models"])
    return {
        "mode": "测试模式（跳过远程调用）" if ROUTER_CONFIG.get("test_mode") else "正常模式",
        "test_mode": bool(ROUTER_CONFIG.get("test_mode", False)),
        "forced_model": ROUTER_CONFIG.get("forced_model", ""),
        "registered_models": list(REMOTE_SERVER_CONFIG.get("model_routes", {}).keys()),
        "remote_endpoints_count": len(REMOTE_SERVER_CONFIG.get("remote_endpoints", [])),
        "discovered_models_count": discovered_count,
        "service_host": "0.0.0.0",
        "service_port": 8000,
    }


@app.get("/api/route/models")
def api_route_models():
    """返回模型列表及能力类型

    - 测试模式：从 model_benchmarks.json 读取所有已测试模型，显示能力类型
    - 非测试模式：通过 remote_endpoints 自动发现（调用 /v1/models 端点），
      并合并手动配置的 model_routes，对照 model_benchmarks.json 显示能力类型；
      未测试的模型只显示名字
    - 每个模型附带 details 字段（已测试时），包含各题库测试详情、效率指标、
      归一化 benchmarks/multimodal，供前端点击展开查看
    """
    import json as _json

    # 加载 model_benchmarks.json
    bench_path = os.path.join(_PROJECT_ROOT, "model_benchmarks.json")
    benchmarks = {}
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            benchmarks = _json.load(f)

    def _get_capabilities(model_data):
        """从模型测试数据判断能力类型列表"""
        caps = []
        if "benchmarks" in model_data and model_data["benchmarks"]:
            caps.append("语言")
        mm = model_data.get("multimodal", {})
        if mm.get("vision_recognition"):
            caps.append("识图")
        if mm.get("image_generation"):
            caps.append("生图")
        return caps

    # 聚合字段名（这些不作为单独的题库测试项展示）
    _AGG_KEYS = {"benchmarks", "efficiency", "multimodal", "image_generation",
                 "last_updated", "模型效率测试"}

    def _build_details(model_data: dict) -> dict:
        """从单个模型的测试数据中提取详情（供前端展开查看）
        统一用 accuracy 表示得分（主观题的 avg_score/10 转换为正确率）"""
        tests = []
        for key, val in model_data.items():
            if key in _AGG_KEYS:
                continue
            if not isinstance(val, dict):
                continue
            # 跳过没有 type 字段的非测试项
            t = val.get("type", "")
            if not t:
                continue
            item = {"name": key, "type": t}
            if t in ("objective", "visual_objective"):
                item["total"] = val.get("total", 0)
                item["correct"] = val.get("correct", 0)
                item["accuracy"] = val.get("accuracy", 0.0)
            elif t in ("subjective", "text_to_image"):
                item["total"] = val.get("total", 0)
                item["total_score"] = val.get("total_score", 0)
                item["avg_score"] = val.get("avg_score", 0.0)
                # 统一为正确率：avg_score/10（满分为 10 分制）
                if "accuracy" in val:
                    item["accuracy"] = val["accuracy"]
                else:
                    item["accuracy"] = round(val.get("avg_score", 0) / 10, 4)
            tests.append(item)

        # 效率测试（单独提取）
        eff_test = model_data.get("模型效率测试", {})
        efficiency_summary = model_data.get("efficiency", {})
        benchmarks_summary = model_data.get("benchmarks", {})
        multimodal_summary = model_data.get("multimodal", {})

        return {
            "tests": tests,
            "efficiency_test": eff_test if eff_test else None,
            "benchmarks": benchmarks_summary,
            "efficiency": efficiency_summary,
            "multimodal": multimodal_summary,
        }

    if ROUTER_CONFIG.get("test_mode", False):
        # 测试模式：从 benchmarks 读
        models = []
        for name, data in benchmarks.items():
            caps = _get_capabilities(data)
            models.append({
                "name": name,
                "capabilities": caps,
                "tested": True,
                "last_updated": data.get("last_updated", ""),
                "details": _build_details(data),
            })
        return {"mode": "test", "models": models, "errors": []}

    # 非测试模式：使用自动发现（remote_endpoints → /v1/models）
    # 同时合并手动配置的 model_routes（向后兼容）
    discovery = discover_remote_models(force_refresh=False)
    remote_models = set(discovery["models"].keys())
    # 合并 model_routes 中手动配置的模型名
    for name in REMOTE_SERVER_CONFIG.get("model_routes", {}).keys():
        remote_models.add(name)
    errors = discovery.get("errors", [])

    # 对照 benchmarks 组装结果
    models = []
    for name in sorted(remote_models):
        if name in benchmarks:
            caps = _get_capabilities(benchmarks[name])
            models.append({
                "name": name,
                "capabilities": caps,
                "tested": True,
                "last_updated": benchmarks[name].get("last_updated", ""),
                "details": _build_details(benchmarks[name]),
            })
        else:
            models.append({
                "name": name,
                "capabilities": [],
                "tested": False,
                "last_updated": "",
                "details": None,
            })

    return {"mode": "normal", "models": models, "errors": errors}


# ============================================================
# 文档查看接口（在 Dashboard 内嵌查看，无需跳转）
# ============================================================

@app.get("/api/docs/markdown")
def api_docs_markdown():
    """返回项目说明文档.md 的纯文本内容"""
    path = os.path.join(_PROJECT_ROOT, "项目说明文档.md")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="项目说明文档.md 不存在")
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(content={"content": f.read(), "format": "markdown"})


@app.get("/api/docs/readme")
def api_docs_readme():
    """返回 README.md 的纯文本内容"""
    path = os.path.join(_PROJECT_ROOT, "README.md")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="README.md 不存在")
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(content={"content": f.read(), "format": "markdown"})


@app.get("/api/docs/docx/download")
def api_docs_docx_download():
    """下载项目说明文档.docx"""
    # 优先返回项目内的副本
    inner_path = os.path.join(_PROJECT_ROOT, "docs", "项目说明文档.docx")
    outer_path = os.path.join(os.path.dirname(_PROJECT_ROOT), "test_auto_route项目说明文档.docx")
    path = inner_path if os.path.exists(inner_path) else outer_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="项目说明文档.docx 不存在")
    return FileResponse(path, filename="项目说明文档.docx",
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ============================================================
# 首页 + 测试 UI
# ============================================================

# 测试任务状态（全局，单实例）
_test_state = {
    "running": False,
    "process": None,
    "logs": [],
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "model_name": None,   # 最近一次测试的模型名（用于导出文档定位 results 子目录）
}
_test_lock = threading.Lock()


def _read_test_output(proc):
    """后台线程：逐行读取子进程输出，存入日志缓冲区"""
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            line = line.rstrip('\n').rstrip('\r')
            with _test_lock:
                _test_state["logs"].append(line)
                # 限制日志缓冲区大小
                if len(_test_state["logs"]) > 8000:
                    _test_state["logs"] = _test_state["logs"][-5000:]
    finally:
        proc.wait()
        with _test_lock:
            _test_state["running"] = False
            _test_state["finished_at"] = time.time()
            _test_state["exit_code"] = proc.returncode
            _test_state["process"] = None


@app.get("/home", response_class=HTMLResponse)
def home_page():
    """首页：选择运行测试或进入路由调参台"""
    html_path = os.path.join(_ROUTER_DIR, "static", "home.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="home.html 不存在")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/test", response_class=HTMLResponse)
def test_ui_page():
    """测试 UI：填写 API 信息、选择模块、运行测试、查看实时日志"""
    html_path = os.path.join(_ROUTER_DIR, "static", "test.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="test.html 不存在")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/test/config")
def api_test_config():
    """获取 autotest/config.py 中的默认配置，供测试 UI 预填"""
    import importlib.util
    cfg_path = os.path.join(_PROJECT_ROOT, "autotest", "config.py")
    spec = importlib.util.spec_from_file_location("autotest_config", cfg_path)
    acfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acfg)
    return {
        "model_name": getattr(acfg, "TEST_MODEL_NAME", ""),
        "api_key": getattr(acfg, "TEST_API_KEY", ""),
        "base_url": getattr(acfg, "TEST_BASE_URL", ""),
        "judge_model": getattr(acfg, "JUDGE_MODEL_NAME", ""),
        "judge_api_key": getattr(acfg, "JUDGE_API_KEY", ""),
        "judge_base_url": getattr(acfg, "JUDGE_BASE_URL", ""),
    }


@app.get("/api/test/benchmarks")
def api_test_benchmarks():
    """返回各题库元信息（用于前端分题库采样配置 UI）

    返回结构：按模块分组，每模块含题库列表，每个题库含 key/name/default_samples。
    name 优先使用 BENCHMARK_NAMES 中的简短显示名（如 "mmlu-知识广度(30)"），
    其次取文件 basename 去扩展名。default_samples 优先从文件名括号数字解析，
    其次从 BENCHMARK_NAMES[key] 的括号数字兜底（如 workplace_pm/secretary 默认 20）。
    """
    import re
    import importlib.util
    cfg_path = os.path.join(_PROJECT_ROOT, "autotest", "config.py")
    spec = importlib.util.spec_from_file_location("autotest_config", cfg_path)
    acfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acfg)

    def _parse_int(s):
        """从字符串中解析第一个括号内的数字"""
        m = re.search(r"\((\d+)\)", s or "")
        return int(m.group(1)) if m else None

    def _short_name(key, fname, names_dict):
        """优先用 BENCHMARK_NAMES 中的简短名，否则 basename 去扩展名"""
        if key in names_dict:
            return names_dict[key]
        return os.path.splitext(os.path.basename(fname))[0]

    def _default_samples(key, fname, names_dict):
        """默认题数：优先从 BENCHMARK_NAMES[key] 解析（显示名中的数字即期望默认值），
        其次从文件名括号数字解析"""
        n = _parse_int(names_dict.get(key, ""))
        if n is not None:
            return n
        return _parse_int(os.path.basename(fname))

    text_names = getattr(acfg, "BENCHMARK_NAMES", {})
    mm_names = getattr(acfg, "MULTIMODAL_BENCHMARK_NAMES", {})
    t2i_names = getattr(acfg, "T2I_BENCHMARK_NAMES", {})

    result = {
        "text": [
            {
                "key": k,
                "name": _short_name(k, v, text_names),
                "default_samples": _default_samples(k, v, text_names),
            }
            for k, v in getattr(acfg, "BENCHMARK_FILES", {}).items()
        ],
        "vision_recognition": [
            {
                "key": k,
                "name": _short_name(k, v, mm_names),
                "default_samples": _default_samples(k, v, mm_names),
            }
            for k, v in getattr(acfg, "MULTIMODAL_BENCHMARK_FILES", {}).items()
        ],
        "image_generation": [
            {
                "key": k,
                "name": _short_name(k, v, t2i_names),
                "default_samples": _default_samples(k, v, t2i_names),
            }
            for k, v in getattr(acfg, "T2I_BENCHMARK_FILES", {}).items()
        ],
        # efficiency 模块无题库概念（按并发数测试）
        "efficiency": [],
    }
    return result


@app.post("/api/test/run")
def api_test_run(body: dict):
    """启动测试子进程（python run.py test ...）"""
    with _test_lock:
        if _test_state["running"]:
            raise HTTPException(status_code=409, detail="已有测试正在运行，请先停止或等待完成")

    model = (body.get("model") or "").strip()
    api_key = (body.get("api_key") or "").strip()
    base_url = (body.get("base_url") or "").strip()
    modules = body.get("modules") or []
    num_samples = body.get("num_samples")
    num_samples_map = body.get("num_samples_map") or {}
    skip_probe = bool(body.get("skip_probe", False))

    if not model:
        raise HTTPException(status_code=400, detail="请填写待测模型名")

    # 构造命令行
    cmd = [sys.executable, os.path.join(_PROJECT_ROOT, "run.py"), "test",
           "--model", model]
    if api_key:
        cmd += ["--api-key", api_key]
    if base_url:
        cmd += ["--base-url", base_url]
    if modules:
        cmd += ["--modules", ",".join(modules)]
    # 分题库采样（优先级高于全局 num_samples）
    if num_samples_map and isinstance(num_samples_map, dict):
        # 清理空值，保留有效正整数
        clean_map = {}
        for k, v in num_samples_map.items():
            try:
                n = int(v)
                if n > 0:
                    clean_map[k] = n
            except (ValueError, TypeError):
                pass
        if clean_map:
            cmd += ["--num-samples-map", json.dumps(clean_map, ensure_ascii=False)]
    elif num_samples:
        try:
            n = int(num_samples)
            if n > 0:
                cmd += ["--num-samples", str(n)]
        except (ValueError, TypeError):
            pass
    if skip_probe:
        cmd += ["--skip-probe"]

    # 启动子进程（合并 stderr 到 stdout，UTF-8 输出）
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        cwd=_PROJECT_ROOT,
        env=env,
    )

    with _test_lock:
        _test_state["running"] = True
        _test_state["process"] = proc
        _test_state["logs"] = []
        _test_state["started_at"] = time.time()
        _test_state["finished_at"] = None
        _test_state["exit_code"] = None
        _test_state["model_name"] = model

    # 后台线程读取输出
    threading.Thread(target=_read_test_output, args=(proc,), daemon=True).start()

    logger.info("测试子进程已启动 (pid=%d): %s", proc.pid, " ".join(cmd))
    return {"status": "started", "pid": proc.pid, "command": " ".join(cmd)}


@app.get("/api/test/status")
def api_test_status(since: int = 0):
    """获取测试状态和增量日志（前端轮询，since=上次获取的日志总数）"""
    with _test_lock:
        total = len(_test_state["logs"])
        new_logs = _test_state["logs"][since:] if since < total else []
        return {
            "running": _test_state["running"],
            "total_logs": total,
            "logs": new_logs,
            "exit_code": _test_state["exit_code"],
            "started_at": _test_state["started_at"],
            "finished_at": _test_state["finished_at"],
        }


@app.post("/api/test/stop")
def api_test_stop():
    """停止正在运行的测试子进程"""
    with _test_lock:
        proc = _test_state["process"]
        if not proc or not _test_state["running"]:
            raise HTTPException(status_code=400, detail="没有正在运行的测试")
    try:
        proc.terminate()
        logger.info("测试子进程已发送终止信号")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止失败: {e}")
    return {"status": "stopped"}


# ============================================================
# 测试文档导出（XLSX）—— 供 Web UI 调用
# ============================================================

def _get_test_results_dir(model_name):
    """根据模型名定位 results/{模型名} 目录（取最新的一个，处理 _1 _2 后缀）"""
    results_base = os.path.join(_PROJECT_ROOT, "results")
    if not os.path.isdir(results_base):
        return None, []
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    # 候选目录：原名 或 原名_1 / 原名_2 ...，取最新的（修改时间最大）
    candidates = []
    for d in os.listdir(results_base):
        full = os.path.join(results_base, d)
        if os.path.isdir(full) and (d == safe_name or d.startswith(safe_name + "_")):
            candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None, []
    candidates.sort(reverse=True)
    return candidates[0][1], [c[1] for c in candidates]


@app.get("/api/test/export/list")
def api_test_export_list():
    """列出最近一次测试生成的 xlsx 文件，供 UI 展示和选择导出"""
    with _test_lock:
        model = _test_state.get("model_name")
        exit_code = _test_state.get("exit_code")
        running = _test_state.get("running")

    # 未测试过
    if not model:
        return {"has_results": False, "reason": "no_test",
                "message": "尚未进行测试，请先运行测试后再导出文档"}

    results_dir, _ = _get_test_results_dir(model)
    if not results_dir:
        return {"has_results": False, "reason": "no_results",
                "message": f"未找到模型 {model} 的测试结果目录（results/{model}），请先完成测试"}

    xlsx_files = []
    for f in sorted(os.listdir(results_dir)):
        if f.lower().endswith(".xlsx"):
            xlsx_files.append({
                "name": f,
                "path": os.path.join(results_dir, f),
                "size": os.path.getsize(os.path.join(results_dir, f)),
                "mtime": os.path.getmtime(os.path.join(results_dir, f)),
            })

    if not xlsx_files:
        return {"has_results": False, "reason": "no_xlsx",
                "message": f"测试结果目录 {results_dir} 中没有 xlsx 文件"}

    return {
        "has_results": True,
        "model": model,
        "results_dir": results_dir,
        "files": xlsx_files,
        "running": running,
        "exit_code": exit_code,
    }


@app.post("/api/test/export")
def api_test_export(body: dict):
    """
    将测试生成的 xlsx 文件导出到用户指定的目录。
    body: { save_dir: str, overwrite: bool }
    处理边界情况：未测试、无结果文件、目标已存在、文件被占用
    """
    with _test_lock:
        model = _test_state.get("model_name")

    if not model:
        raise HTTPException(status_code=400, detail="尚未进行测试，请先运行测试后再导出文档")

    results_dir, _ = _get_test_results_dir(model)
    if not results_dir:
        raise HTTPException(status_code=400, detail=f"未找到模型 {model} 的测试结果目录，请先完成测试")

    save_dir = (body.get("save_dir") or "").strip()
    overwrite = bool(body.get("overwrite", False))

    if not save_dir:
        raise HTTPException(status_code=400, detail="请指定保存目录")

    # 收集源 xlsx 文件
    src_files = []
    for f in sorted(os.listdir(results_dir)):
        if f.lower().endswith(".xlsx"):
            src_files.append(os.path.join(results_dir, f))
    if not src_files:
        raise HTTPException(status_code=400, detail=f"测试结果目录 {results_dir} 中没有 xlsx 文件")

    # 创建目标目录
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"无法创建保存目录 {save_dir}：{e}")

    # 检查目标文件是否已存在
    existing = []
    for src in src_files:
        dst = os.path.join(save_dir, os.path.basename(src))
        if os.path.exists(dst):
            existing.append(os.path.basename(src))
    if existing and not overwrite:
        return JSONResponse(status_code=409, content={
            "status": "exists",
            "message": f"保存目录中已存在同名文件：{', '.join(existing)}。是否覆盖？",
            "existing_files": existing,
        })

    # 执行复制（捕获文件被占用错误）
    import shutil
    copied = []
    failed = []
    for src in src_files:
        dst = os.path.join(save_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
            copied.append(os.path.basename(src))
        except PermissionError:
            failed.append({
                "file": os.path.basename(src),
                "reason": "文件被占用（可能在 Excel 中打开），请关闭后重试",
            })
        except OSError as e:
            failed.append({"file": os.path.basename(src), "reason": str(e)})

    if failed and not copied:
        raise HTTPException(status_code=500, detail={
            "message": "所有文件导出失败",
            "failed": failed,
        })

    return {
        "status": "ok",
        "saved_dir": save_dir,
        "copied": copied,
        "failed": failed,
        "message": f"已导出 {len(copied)} 个文件到 {save_dir}" + (
            f"，{len(failed)} 个失败（文件被占用）" if failed else ""),
    }


@app.get("/api/test/export/download")
def api_test_export_download():
    """将测试生成的所有 xlsx 打包成 zip 下载（浏览器下载对话框让用户选保存位置）"""
    import io
    import zipfile

    with _test_lock:
        model = _test_state.get("model_name")
    if not model:
        raise HTTPException(status_code=400, detail="尚未进行测试，请先运行测试后再导出文档")

    results_dir, _ = _get_test_results_dir(model)
    if not results_dir:
        raise HTTPException(status_code=400, detail=f"未找到模型 {model} 的测试结果目录")

    src_files = [os.path.join(results_dir, f) for f in sorted(os.listdir(results_dir))
                 if f.lower().endswith(".xlsx")]
    if not src_files:
        raise HTTPException(status_code=400, detail="测试结果目录中没有 xlsx 文件")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src_files:
            zf.write(f, os.path.basename(f))
    buf.seek(0)

    safe_model = model.replace("/", "_").replace("\\", "_").replace(":", "_")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_model}_results.zip"'},
    )


# ============================================================
# 启动入口
# ============================================================

def _open_browser_when_ready(dashboard_url: str, health_url: str, max_wait: float = 120.0):
    """轮询服务是否就绪，就绪后再打开浏览器（避免在模型加载期间打开导致无法访问）"""
    import threading
    import time
    import urllib.request
    import webbrowser

    def _wait_and_open():
        poll_interval = 1.0
        waited = 0.0
        while waited < max_wait:
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=2):
                    webbrowser.open(dashboard_url)
                    logger.info("服务已就绪，已在浏览器中打开可视化调参台：%s", dashboard_url)
                    return
            except Exception:
                pass
            time.sleep(poll_interval)
            waited += poll_interval
            if int(waited) % 10 == 0 and int(waited) > 0:
                logger.info("正在加载模型，等待服务就绪... (%ds)", int(waited))
        logger.warning("等待服务就绪超时（%ss），请手动访问：%s", int(max_wait), dashboard_url)

    threading.Thread(target=_wait_and_open, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    port = 8000
    dashboard_url = f"http://localhost:{port}/home"
    health_url = f"http://localhost:{port}/api/params"

    # 服务就绪后自动打开可视化调参台
    _open_browser_when_ready(dashboard_url, health_url, max_wait=120.0)

    logger.info("=" * 60)
    logger.info("智能路由服务启动中...")
    logger.info("可视化调参台: %s（服务就绪后自动打开）", dashboard_url)
    logger.info("API 文档:      http://localhost:%d/docs", port)
    logger.info("=" * 60)

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
    )
