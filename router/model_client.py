# model_client.py
# ============================================================
# 远程模型客户端
# 负责：根据模型名称查找服务器 URL → 发送 OpenAI 兼容请求 → 返回模型输出
# 所有路由和请求信息均输出到后台日志
# ============================================================

import httpx
import json as _json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from config import REMOTE_SERVER_CONFIG

logger = logging.getLogger(__name__)

# ============================================================
# 持久化 HTTP 客户端（连接池复用，避免每次请求都 TLS 握手）
# ============================================================
_stream_client: Optional[httpx.AsyncClient] = None
_sync_client: Optional[httpx.Client] = None


def _get_stream_client() -> httpx.AsyncClient:
    """获取或创建持久化的异步 HTTP 客户端（连接池复用）"""
    global _stream_client
    if _stream_client is None:
        verify_ssl: bool = REMOTE_SERVER_CONFIG.get("verify_ssl", True)
        timeout: int = REMOTE_SERVER_CONFIG.get("request_timeout", 120)
        _stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(timeout)),
            verify=verify_ssl,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
        )
        logger.info("持久化异步 HTTP 客户端已创建（连接池: %d keepalive / %d max）", 10, 50)
    return _stream_client


def _get_sync_client() -> httpx.Client:
    """获取或创建持久化的同步 HTTP 客户端（连接池复用）"""
    global _sync_client
    if _sync_client is None:
        verify_ssl: bool = REMOTE_SERVER_CONFIG.get("verify_ssl", True)
        timeout: int = REMOTE_SERVER_CONFIG.get("request_timeout", 120)
        _sync_client = httpx.Client(
            timeout=httpx.Timeout(float(timeout)),
            verify=verify_ssl,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
        )
        logger.info("持久化同步 HTTP 客户端已创建（连接池: %d keepalive / %d max）", 10, 50)
    return _sync_client


async def close_http_clients():
    """关闭所有持久化 HTTP 客户端"""
    global _stream_client, _sync_client
    if _stream_client is not None:
        await _stream_client.aclose()
        _stream_client = None
        logger.info("异步 HTTP 客户端已关闭")
    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None
        logger.info("同步 HTTP 客户端已关闭")


def get_model_url(model_name: str) -> Optional[str]:
    """
    根据模型名称，获取远程服务器上该模型的完整请求 URL。

    返回 None 表示该模型未在 model_routes 中配置（即服务器上不存在该模型）。
    """
    model_routes: dict = REMOTE_SERVER_CONFIG.get("model_routes", {})

    if model_name not in model_routes:
        return None

    route: str = model_routes[model_name]
    route = route.strip()

    # 判断是完整 URL 还是路径后缀
    if route.startswith("http://") or route.startswith("https://"):
        return route
    else:
        base_url: str = REMOTE_SERVER_CONFIG.get("base_url", "").rstrip("/")
        return f"{base_url}/{route}/v1/chat/completions"


def call_remote_model(
    model_name: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    调用远程服务器上的指定模型。

    参数:
        model_name  : 程序内部模型名（与 model_benchmarks.json 一致）
        messages    : 对话消息列表 [{"role": "system/user/assistant", "content": "..."}]
        temperature : 生成温度
        max_tokens  : 最大生成 token 数
        stream      : 是否流式输出（当前仅支持非流式）

    返回:
        {
            "success": True/False,
            "data":    { ... }            # 成功时：OpenAI 格式的完整响应 JSON
            "content": "错误提示文本",      # 失败时：友好的错误信息
            "error":   "错误原因"          # 失败时：技术错误描述
        }
    """

    # ---------- 构造请求 URL ----------
    url = get_model_url(model_name)
    verify_ssl = REMOTE_SERVER_CONFIG.get("verify_ssl", True)

    # ---------- 路由日志 ----------
    logger.info("=" * 60)
    logger.info("[路由日志] 目标模型  : %s", model_name)

    if url is None:
        logger.warning(
            "[路由日志] ❌ 模型 '%s' 在服务器上不存在 —— "
            "请在 config.py → REMOTE_SERVER_CONFIG → model_routes 中添加该模型的路径映射",
            model_name
        )
        logger.info("=" * 60)
        return {
            "success": False,
            "error": f"模型 '{model_name}' 未在 model_routes 中配置",
            "content": (
                f"【路由提示】智能路由选择了模型 '{model_name}'，"
                f"但该模型在服务器上不存在。\n"
                f"请在 config.py 的 model_routes 中添加该模型的路径映射后重试。"
            ),
        }

    logger.info("[路由日志] 请求 URL  : %s", url)
    logger.info("[路由日志] 请求参数  : temperature=%.2f, max_tokens=%d, stream=%s",
                temperature, max_tokens, stream)

    # ---------- 构造 OpenAI 兼容请求体 ----------
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    timeout: int = REMOTE_SERVER_CONFIG.get("request_timeout", 120)

    # ---------- 发送请求（使用持久化连接池） ----------
    try:
        client = _get_sync_client()
        response = client.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            logger.info("[路由日志] ✅ 请求成功，模型已返回响应")
            logger.info("=" * 60)
            return {
                "success": True,
                "data": result,
            }

        else:
            logger.error("[路由日志] ❌ HTTP 错误，状态码: %d", response.status_code)
            logger.error("[路由日志] 响应内容（前500字符）: %s", response.text[:500])
            logger.info("=" * 60)
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "content": (
                    f"【错误】模型服务器返回 HTTP {response.status_code} 错误。\n"
                    f"请检查服务器上的模型是否正常运行。"
                ),
            }

    except httpx.ConnectError:
        logger.error("[路由日志] ❌ 无法连接到服务器: %s", url)
        logger.info("=" * 60)
        return {
            "success": False,
            "error": "连接失败",
            "content": (
                "【错误】无法连接到模型服务器。\n"
                "请检查 config.py 中 model_routes 的 URL 地址是否正确，以及服务器是否在运行。"
            ),
        }

    except httpx.TimeoutException:
        logger.error("[路由日志] ❌ 请求超时（%d 秒）", timeout)
        logger.info("=" * 60)
        return {
            "success": False,
            "error": "请求超时",
            "content": f"【错误】模型请求超时（{timeout} 秒），请稍后重试或增大 request_timeout。",
        }

    except Exception as e:
        logger.error("[路由日志] ❌ 未知错误: %s", e)
        logger.info("=" * 60)
        return {
            "success": False,
            "error": str(e),
            "content": f"【错误】请求模型时发生未知错误: {e}",
        }


async def stream_remote_model(
    model_name: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """
    流式调用远程模型，逐 token 返回 SSE 数据块。

    用法:
        async for chunk in stream_remote_model("model-name", messages):
            yield chunk  # 直接透传给客户端

    返回的每个 chunk 已经是完整的 SSE 行（含 "data: ...\\n\\n"），
    并且 model 字段已替换为路由选择的模型名。
    """
    url = get_model_url(model_name)
    verify_ssl = REMOTE_SERVER_CONFIG.get("verify_ssl", True)
    timeout: int = REMOTE_SERVER_CONFIG.get("request_timeout", 120)

    logger.info("=" * 60)
    logger.info("[路由日志] 目标模型  : %s", model_name)

    if url is None:
        logger.warning(
            "[路由日志] ❌ 模型 '%s' 在服务器上不存在",
            model_name
        )
        logger.info("=" * 60)
        # 返回一个错误 SSE 块
        error_data = _json.dumps({
            "error": f"模型 '{model_name}' 未在 model_routes 中配置",
            "content": f"【路由提示】模型 '{model_name}' 在服务器上不存在。",
        }, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
        return

    logger.info("[路由日志] 请求 URL  : %s", url)
    logger.info("[路由日志] 请求参数  : temperature=%.2f, max_tokens=%d, stream=True",
                temperature, max_tokens)

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        client = _get_stream_client()
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                logger.error("[路由日志] ❌ HTTP 错误，状态码: %d", response.status_code)
                error_data = _json.dumps({
                    "error": f"HTTP {response.status_code}",
                    "content": f"【错误】模型服务器返回 HTTP {response.status_code}。",
                }, ensure_ascii=False)
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
                logger.info("=" * 60)
                return

            logger.info("[路由日志] ✅ 流式连接建立，开始接收数据...")
            chunk_count = 0

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # 去掉 "data: " 前缀

                # 遇到 [DONE] 直接透传
                if data_str.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                # 解析 JSON，替换 model 字段
                try:
                    chunk_data = _json.loads(data_str)
                    chunk_data["model"] = model_name
                    yield f"data: {_json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    chunk_count += 1
                except _json.JSONDecodeError:
                    # 非 JSON 行直接透传
                    yield f"data: {data_str}\n\n"

            logger.info("[路由日志] ✅ 流式传输完成，共 %d 个 chunk", chunk_count)
            logger.info("=" * 60)

    except httpx.ConnectError:
        logger.error("[路由日志] ❌ 无法连接到服务器: %s", url)
        error_data = _json.dumps({
            "error": "连接失败",
            "content": "【错误】无法连接到模型服务器。",
        }, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
        logger.info("=" * 60)

    except httpx.TimeoutException:
        logger.error("[路由日志] ❌ 请求超时（%d 秒）", timeout)
        error_data = _json.dumps({
            "error": "请求超时",
            "content": f"【错误】模型请求超时（{timeout} 秒）。",
        }, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
        logger.info("=" * 60)

    except Exception as e:
        logger.error("[路由日志] ❌ 未知错误: %s", e)
        error_data = _json.dumps({
            "error": str(e),
            "content": f"【错误】请求模型时发生未知错误: {e}",
        }, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
        logger.info("=" * 60)
