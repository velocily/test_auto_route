"""
============================================================================
模型 API 调用模块
============================================================================
提供统一的 ask_model() 接口，兼容 OpenAI 风格 API。
同时封装 提示词构建 与 答案提取 逻辑。
支持纯文本和多模态（视觉）两种调用模式。
"""
import re
import os
import time
import base64
import requests
import urllib3

# 禁用 SSL 警告（自签名证书场景）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# 持久化 Session（连接池复用，避免每次请求 TLS 握手）
# =========================
_SESSION = None


def _get_session():
    """获取或创建持久化 requests.Session，复用 TCP 连接"""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        # 禁用 SSL 验证（自签名证书场景）
        _SESSION.verify = False
        # 连接池大小：适配并发场景
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=1,
        )
        _SESSION.mount("https://", adapter)
        _SESSION.mount("http://", adapter)
    return _SESSION


def _warm_up(api_key, base_url, model_name, timeout=30):
    """
    预热 API 连接：发送一条极短请求，建立 TLS 会话并触发服务端模型加载。
    失败不阻塞主流程。
    """
    try:
        print("  [预热] 正在建立 API 连接...", end=" ", flush=True)
        session = _get_session()
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "1+1=?"}],
            "temperature": 0,
            "max_tokens": 5,
        }
        r = session.post(base_url, headers=headers, json=payload,
                         timeout=(5, timeout))
        if r.status_code == 200:
            print("完成")
        else:
            print(f"返回 {r.status_code}，继续...")
    except Exception as e:
        print(f"跳过（{e}）")


# =========================
# 能力探测：明确判断模型/接口是否支持某能力
# =========================
# 探测结果：
#   True  = 支持（可继续完整测试）
#   False = 不支持（API 返回 404/405/400 明确拒绝，或返回错误信息含"not support"等）
#   None  = 无法确定（网络超时等，建议按"未测试"处理，不轻易判定为不支持）
#
# 设计目标：避免"程序认为不具备多模态能力但实际是有的"误判
#   - 仅当 API 明确返回 404/405 或错误信息明确说不支持时才判定 False
#   - 超时/5xx 等临时错误返回 None，不轻易判定
def probe_vision_capability(api_key, base_url, model_name, timeout=60):
    """
    探测模型是否支持视觉识图（Vision 输入）。
    发送一个最小化的多模态请求（1x1 测试图 + 简短问题），
    根据响应判断接口是否支持 image_url 输入。

    返回:
        True  = 支持
        False = 明确不支持（404/405/400 + not support 等关键词）
        None  = 无法确定（超时/5xx/解析异常）
    """
    session = _get_session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 1x1 红色 PNG 的 base64（最小测试图，避免传输大图）
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    image_url = f"data:image/png;base64,{tiny_png_b64}"

    content = [
        {"type": "text", "text": "What color is this image? Answer in one word."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 16,
    }

    try:
        r = session.post(base_url, headers=headers, json=payload, timeout=(10, timeout))
    except requests.exceptions.Timeout:
        print("  [能力探测-识图] 请求超时，无法确定是否支持")
        return None
    except Exception as e:
        print(f"  [能力探测-识图] 请求异常: {e}")
        return None

    status = r.status_code
    # 200 / 4xx-with-content 都视为支持（部分服务对 1x1 图返回 200 但内容为空）
    if status == 200:
        print("  [能力探测-识图] 接口返回 200，支持视觉输入")
        return True

    # 非 200：检查是否明确"不支持"
    body = ""
    try:
        body = r.text[:500]
    except Exception:
        pass
    body_lower = body.lower()

    # 明确不支持的信号：404/405（接口不存在）、400 + not support 关键词
    unsupported_signals = [
        "not support", "unsupported", "image input", "vision",
        "multimodal", "no such", "not found", "invalid_request",
        "does not support", "can only support",
    ]
    if status in (404, 405):
        print(f"  [能力探测-识图] 接口返回 {status}，判定为不支持")
        return False
    if status == 400 and any(sig in body_lower for sig in unsupported_signals):
        print(f"  [能力探测-识图] 接口返回 400 且含不支持关键词，判定为不支持")
        return False
    if status in (401, 403):
        print(f"  [能力探测-识图] 接口返回 {status}（鉴权问题），无法确定")
        return None

    # 其他 4xx/5xx：保守起见返回 None（可能是临时错误）
    print(f"  [能力探测-识图] 接口返回 {status}，无法确定（body: {body[:120]}）")
    return None


def probe_t2i_capability(api_key, base_url, model_name, timeout=60, size="1024x1024"):
    """
    探测模型是否支持文生图（/v1/images/generations 接口）。
    发送一个最小化的生图请求，根据响应判断接口是否存在且可用。

    返回:
        True  = 支持
        False = 明确不支持（404/405，或 400 + not support 关键词）
        None  = 无法确定（超时/5xx/解析异常）
    """
    session = _get_session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 构造 images/generations URL
    if "/v1/chat/completions" in base_url:
        images_url = base_url.replace("/v1/chat/completions", "/v1/images/generations")
    elif base_url.rstrip("/").endswith("/v1"):
        images_url = base_url.rstrip("/") + "/images/generations"
    elif "/v1" in base_url:
        idx = base_url.find("/v1")
        images_url = base_url[:idx + 3] + "/images/generations"
    else:
        images_url = base_url.rstrip("/") + "/v1/images/generations"

    payload = {
        "model": model_name,
        "prompt": "a red circle",
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    }

    try:
        r = session.post(images_url, headers=headers, json=payload, timeout=(10, timeout))
    except requests.exceptions.Timeout:
        print("  [能力探测-生图] 请求超时，无法确定是否支持")
        return None
    except Exception as e:
        print(f"  [能力探测-生图] 请求异常: {e}")
        return None

    status = r.status_code
    if status == 200:
        print("  [能力探测-生图] 接口返回 200，支持文生图")
        return True

    body = ""
    try:
        body = r.text[:500]
    except Exception:
        pass
    body_lower = body.lower()

    unsupported_signals = [
        "not support", "unsupported", "image generation", "no such",
        "not found", "does not support", "can only support",
        "model does not", "not available",
    ]
    if status in (404, 405):
        print(f"  [能力探测-生图] 接口返回 {status}，判定为不支持")
        return False
    if status == 400 and any(sig in body_lower for sig in unsupported_signals):
        print(f"  [能力探测-生图] 接口返回 400 且含不支持关键词，判定为不支持")
        return False
    if status in (401, 403):
        print(f"  [能力探测-生图] 接口返回 {status}（鉴权问题），无法确定")
        return None

    print(f"  [能力探测-生图] 接口返回 {status}，无法确定（body: {body[:120]}）")
    return None


def ask_model(prompt, api_key, base_url, model_name, timeout=600, max_tokens=None):
    """
    调用模型 API（兼容 OpenAI chat/completions 风格）
    使用持久化 Session 复用连接，避免每次 TLS 握手开销。

    返回模型输出的原始文本；失败返回空字符串。
    """
    session = _get_session()
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    last_error = None
    for attempt in range(1, 4):  # 最多重试 3 次
        try:
            # timeout=(connect_timeout, read_timeout)
            r = session.post(base_url, headers=headers, json=payload,
                             timeout=(15, timeout))
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content")
            # 推理模型可能 content 为空字符串（所有 token 被 reasoning 消耗）
            if not content:
                content = msg.get("reasoning") or msg.get("reasoning_content") or ""
            return content
        except requests.exceptions.Timeout:
            last_error = f"请求超时（{timeout}s）"
            if attempt < 3:
                wait = attempt * 2
                print(f"  [重试 {attempt}/3] {last_error}，{wait}s 后重试...")
                time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            last_error = f"连接失败: {e}"
            if attempt < 3:
                wait = attempt * 2
                print(f"  [重试 {attempt}/3] {last_error}，{wait}s 后重试...")
                time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}"
            print(f"  [API异常] {last_error}: {e.response.text[:200]}")
            break  # HTTP 错误不重试
        except Exception as e:
            last_error = str(e)
            print(f"  [API异常] {last_error}")
            if attempt < 3:
                wait = attempt * 2
                print(f"  [重试 {attempt}/3] {wait}s 后重试...")
                time.sleep(wait)

    print(f"  [API失败] {last_error}")
    return ""


def _encode_image(image_path):
    """将本地图片编码为 base64 data URL。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else (ext or "png")
    return f"data:image/{mime};base64,{b64}"


def ask_model_multimodal(prompt, image_path, api_key, base_url, model_name,
                         timeout=600, max_tokens=None):
    """
    调用多模态模型 API（OpenAI Vision 兼容接口）。
    将文本 prompt 和本地图片一起发送给模型。

    返回模型输出的原始文本；失败返回空字符串。
    推理模型可能将内容放在 reasoning 字段，content 为空时回退到 reasoning。
    """
    session = _get_session()
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        image_url = _encode_image(image_path)
    except Exception as e:
        print(f"  [图片编码失败] {image_path}: {e}")
        return ""

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        # 推理模型需要足够的 token 完成推理后输出答案
        "max_tokens": max_tokens or 4096,
    }

    last_error = None
    for attempt in range(1, 4):
        try:
            r = session.post(base_url, headers=headers, json=payload,
                             timeout=(15, timeout))
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content")
            # 推理模型可能 content 为空字符串（所有 token 被 reasoning 消耗）
            if not content:
                content = msg.get("reasoning") or msg.get("reasoning_content") or ""
            return content
        except requests.exceptions.Timeout:
            last_error = f"请求超时（{timeout}s）"
            if attempt < 3:
                wait = attempt * 2
                print(f"  [重试 {attempt}/3] {last_error}，{wait}s 后重试...")
                time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}"
            print(f"  [API异常] {last_error}: {e.response.text[:200]}")
            break
        except Exception as e:
            last_error = str(e)
            print(f"  [API异常] {last_error}")
            if attempt < 3:
                wait = attempt * 2
                print(f"  [重试 {attempt}/3] {wait}s 后重试...")
                time.sleep(wait)

    print(f"  [API失败] {last_error}")
    return ""


def ask_model_t2i(prompt, api_key, base_url, model_name, timeout=600, size="1024x1024"):
    """
    调用文生图模型 API（OpenAI 兼容 /v1/images/generations 接口）。

    参数:
        prompt     : 生图提示词
        api_key    : API 密钥
        base_url   : 完整的 chat/completions URL（自动转换为 images/generations URL）
        model_name : 模型名
        timeout    : 超时秒数
        size       : 图片尺寸

    返回:
        (image_url_or_b64, raw_response)
        - image_url_or_b64: 图片 URL 或 base64 数据（失败为空字符串）
        - raw_response: 完整响应 JSON（用于调试）
    """
    session = _get_session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 将 /v1/chat/completions 转换为 /v1/images/generations
    # 兼容完整 URL 和 base URL 两种情况
    if "/v1/chat/completions" in base_url:
        images_url = base_url.replace("/v1/chat/completions", "/v1/images/generations")
    elif base_url.rstrip("/").endswith("/v1"):
        images_url = base_url.rstrip("/") + "/images/generations"
    elif "/v1" in base_url:
        # 截断到 /v1 后追加
        idx = base_url.find("/v1")
        images_url = base_url[:idx + 3] + "/images/generations"
    else:
        # 兜底：直接拼接
        images_url = base_url.rstrip("/") + "/v1/images/generations"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    }

    last_error = None
    for attempt in range(1, 4):
        try:
            r = session.post(images_url, headers=headers, json=payload,
                             timeout=(15, timeout))
            r.raise_for_status()
            data = r.json()
            # OpenAI 标准格式：data[0].b64_json 或 data[0].url
            items = data.get("data", [])
            if items:
                item = items[0]
                if "b64_json" in item:
                    return item["b64_json"], data
                if "url" in item:
                    return item["url"], data
            last_error = "响应中未找到图片数据"
        except requests.exceptions.Timeout:
            last_error = f"请求超时（{timeout}s）"
            if attempt < 3:
                wait = attempt * 2
                print(f"  [重试 {attempt}/3] {last_error}，{wait}s 后重试...")
                time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}"
            print(f"  [API异常] {last_error}: {e.response.text[:200]}")
            break
        except Exception as e:
            last_error = str(e)
            print(f"  [API异常] {last_error}")
            if attempt < 3:
                wait = attempt * 2
                print(f"  [重试 {attempt}/3] {wait}s 后重试...")
                time.sleep(wait)

    print(f"  [文生图API失败] {last_error}")
    return "", {}


def build_t2i_prompt(question):
    """
    构建文生图提示词。
    题干本身就是提示词，这里直接返回。
    """
    return question["question_text"]


def judge_t2i_answer(question, model_output, judge_api_key, judge_base_url, judge_model, timeout=600):
    """
    文生图评分：调用打分模型对生成的图片做 0-10 分评估。

    评分流程：
      1. 将生成的图片（base64 或 URL）作为多模态输入发送给打分模型
      2. 打分模型根据提示词和评分维度给出 0-10 分
      3. 提取分数返回

    参数:
        model_output : ask_model_t2i 返回的图片标识（base64 或 URL）
    """
    prompt_text = question["question_text"]
    eval_dims = question.get("eval_dims", ["主体准确度", "场景契合度", "画面质量"])

    if not model_output:
        return 0, "模型未生成图片"

    # 构造图片 URL（base64 或远程 URL）
    if model_output.startswith("http"):
        image_url = model_output
    else:
        image_url = f"data:image/png;base64,{model_output}"

    dims_str = "、".join(eval_dims)
    judge_prompt = (
        f"你是一位严格的图像生成评审专家。请根据以下提示词和评分维度，"
        f"对生成的图片进行打分。\n\n"
        f"生图提示词：{prompt_text}\n\n"
        f"评分维度：{dims_str}\n\n"
        "评分标准（0-10分）：\n"
        "- 10分：图片完全符合提示词，画面质量极高，所有维度均优秀\n"
        "- 7-9分：图片基本符合提示词，有少量瑕疵\n"
        "- 4-6分：图片部分符合提示词，有明显问题\n"
        "- 1-3分：图片与提示词差距较大\n"
        "- 0分：未生成图片或完全不符合提示词\n\n"
        "【重要规则】你必须只输出一个0-10之间的整数分数，"
        "然后换行给出一句简短的评分理由。格式如下：\n"
        "分数\n理由\n"
        "不要输出任何其他内容。"
    )

    # 调用打分模型（多模态）
    content = [
        {"type": "text", "text": judge_prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 2048,
    }

    session = _get_session()
    headers = {"Authorization": f"Bearer {judge_api_key}", "Content-Type": "application/json"}

    for attempt in range(3):
        try:
            r = session.post(judge_base_url, headers=headers, json=payload,
                             timeout=(15, timeout))
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            raw_judge = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
            if not raw_judge:
                continue
            score, comment = _extract_subjective_score(raw_judge)
            if score >= 0:
                return score, comment
            print(f"    评分提取失败 {attempt+1}/3: {raw_judge[:150]}")
        except Exception as e:
            print(f"    打分模型调用失败 {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(3)

    return 0, "[评分失败-需人工审核] 打分模型多次返回不可解析"


def build_mc_prompt(question):
    """
    构建选择题（MMLU / HellaSwag）提示词。
    通过 system + user 双重约束，要求模型仅输出选项字母。
    """
    q_text = question["question_text"]
    options = question["options"]

    opts_str = ""
    for label in ["A", "B", "C", "D"]:
        if label in options:
            opts_str += f"{label}. {options[label]}\n"

    prompt = (
        "【重要规则】你必须只输出一个字母（A/B/C/D），不能输出任何其他文字、"
        "解释、分析、标点或换行。违反规则将被视为错误。\n\n"
        f"问题：{q_text}\n\n"
        f"选项：\n{opts_str}\n"
        "答案（仅一个字母）："
    )
    return prompt


def build_math_prompt(question):
    """
    构建数学题（GSM8K）提示词。
    要求模型仅输出最终数字。
    """
    q_text = question["question_text"]

    prompt = (
        "【重要规则】你必须只输出最终的数字答案，不能输出任何解释、步骤、"
        "推理过程、单位、标点或换行。违反规则将被视为错误。\n\n"
        f"问题：{q_text}\n\n"
        "答案（仅数字）："
    )
    return prompt


def build_subjective_prompt(question):
    """
    构建主观题提示词。
    允许模型自由发挥，但要求结构化输出以便评分。
    """
    q_text = question["question_text"]
    role = question.get("category", "")

    prompt = (
        f"你是一名{role}。请认真回答以下问题。\n"
        "回答要求：结构清晰、条理分明，直接给出你的方案或建议，"
        "不要添加问候语、自我介绍等无关内容。\n\n"
        f"问题：{q_text}\n\n"
        "你的回答："
    )
    return prompt


def build_longbench_prompt(question):
    """
    构建长上下文阅读理解提示词。
    包含完整的上下文文本 + 问题，要求模型仅用一句话回答。
    """
    q_text = question["question_text"]
    context = question.get("context", "")

    if context:
        prompt = (
            "请仔细阅读以下长文本内容，然后回答后续问题。\n\n"
            "========================================\n"
            f"{context}\n"
            "========================================\n\n"
            f"问题：{q_text}\n\n"
            "【重要规则】你必须只回答一句话，不能输出任何多余的解释、"
            "分析或不相关内容。直接给出最终答案。\n\n"
            "答案（一句话）："
        )
    else:
        prompt = (
            f"问题：{q_text}\n\n"
            "【重要规则】你必须只回答一句话，不能输出任何多余的解释、"
            "分析或不相关内容。直接给出最终答案。\n\n"
            "答案（一句话）："
        )
    return prompt


def build_visual_mc_prompt(question):
    """
    构建视觉选择题（ChartQA / TextVQA / MathVista / VQA / MMMU）提示词。
    图片通过 ask_model_multimodal 单独发送，此处仅构建文本部分。
    """
    q_text = question["question_text"]
    options = question["options"]

    opts_str = ""
    for label in ["A", "B", "C", "D"]:
        if label in options:
            opts_str += f"{label}. {options[label]}\n"

    prompt = (
        "请仔细观察图片并回答以下问题。\n\n"
        "【重要规则】你必须只输出一个字母（A/B/C/D），不能输出任何其他文字、"
        "解释、分析、标点或换行。违反规则将被视为错误。\n\n"
        f"问题：{q_text}\n\n"
        f"选项：\n{opts_str}\n"
        "答案（仅一个字母）："
    )
    return prompt


# =========================
# 答案提取（从模型输出中提取最终答案）
# =========================

def extract_mc_answer(raw_output):
    """从选择题模型输出中提取选项字母 A/B/C/D"""
    if not raw_output:
        return ""
    text = raw_output.strip()

    # 1. 尝试匹配 "答案" 后面的字母
    m = re.search(r'答案\s*[：:]\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2. 尝试匹配 "正确选项是" / "最终输出" / "答案是" 后面的字母（推理模型常见格式）
    m = re.search(r'(?:正确选项[是是]?|最终输出|答案[是是]?|选)\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. 尝试匹配以 A/B/C/D 开头的行（content 字段常见格式）
    m = re.search(r'(?:^|\n)\s*([A-D])\s*(?:[.\s,]|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4. 对于 reasoning 文本，从末尾 500 字符中提取最后一个 A/B/C/D
    tail = text[-500:] if len(text) > 500 else text
    letters = re.findall(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', tail, re.IGNORECASE)
    if letters:
        return letters[-1].upper()

    # 5. 兜底：取第一个字符
    first_char = text[:1].upper()
    if first_char in "ABCD":
        return first_char
    return ""


def extract_math_answer(raw_output):
    """从数学题模型输出中提取最终数字"""
    if not raw_output:
        return ""
    text = raw_output.strip()

    # 1. 尝试匹配 #### NNN 格式
    m = re.search(r'####\s*([\d,.]+)', text)
    if m:
        return m.group(1).replace(",", "")

    # 2. 尝试匹配 "答案" 后面的数字
    m = re.search(r'答案\s*[：:]\s*([\d,.]+)', text)
    if m:
        return m.group(1).replace(",", "")

    # 3. 尝试匹配最后一段数字（包括小数、逗号分隔）
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        numbers = re.findall(r'[\d,]+\.?\d*', line)
        if numbers:
            return numbers[-1].replace(",", "")

    # 4. 兜底：全文找数字
    numbers = re.findall(r'[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text


def extract_subjective_answer(raw_output):
    """主观题不需要提取，直接返回完整输出"""
    return raw_output.strip() if raw_output else ""


# =========================
# 评分模块
# =========================

def judge_mc_answer(question, model_output, judge_api_key, judge_base_url, judge_model, timeout=600):
    """
    选择题打分：直接字符串比对模型答案与正确答案。
    返回 (得分: int 0/1, 评语: str)
    """
    extracted = extract_mc_answer(model_output)
    correct = question["correct_answer"].strip().upper()
    score = 1 if extracted.upper() == correct else 0
    comment = f"模型输出: {extracted}, 正确答案: {correct}"
    return score, comment


def judge_math_answer(question, model_output, judge_api_key, judge_base_url, judge_model, timeout=600):
    """
    数学题打分：提取数字后与参考答案比对。
    返回 (得分: int 0/1, 评语: str)
    """
    extracted = extract_math_answer(model_output)
    correct = question["correct_answer"].strip()

    # 数值标准化比较
    try:
        val_extracted = float(extracted.replace(",", ""))
        val_correct = float(correct.replace(",", ""))
        score = 1 if abs(val_extracted - val_correct) < 1e-6 else 0
    except (ValueError, AttributeError):
        score = 1 if extracted.strip() == correct.strip() else 0

    comment = f"模型输出: {extracted}, 正确答案: {correct}"
    return score, comment


def judge_subjective_answer(question, model_output, judge_api_key, judge_base_url, judge_model, timeout=600):
    """
    主观题打分：调用打分模型API进行0-10分评分。
    内置重试机制：如果打分模型返回异常或无法提取分数，自动重试最多2次。
    返回 (得分: int 0-10, 评语: str)
    """
    q_text = question["question_text"]
    role = question.get("category", "")
    answer = extract_subjective_answer(model_output)

    if not answer:
        return 0, "模型未输出任何内容"

    judge_prompt = (
        f"你是一位严格的职场能力评审专家。请根据以下评分标准对{role}的回答进行打分。\n\n"
        f"题目：{q_text}\n\n"
        f"{role}的回答：\n{answer}\n\n"
        "评分标准（0-10分）：\n"
        "- 10分：回答完美，逻辑严密，方案完整可执行，专业性强\n"
        "- 7-9分：回答优秀，有少量可改进之处\n"
        "- 4-6分：回答一般，有明显不足但仍有一定价值\n"
        "- 1-3分：回答较差，逻辑混乱或内容空洞\n"
        "- 0分：完全未回答或答非所问\n\n"
        "【重要规则】你必须只输出一个0-10之间的整数分数，"
        "然后换行给出一句简短的评分理由。格式如下：\n"
        "分数\n理由\n"
        "不要输出任何其他内容。"
    )

    return _judge_with_retry(
        judge_prompt, judge_api_key, judge_base_url, judge_model, timeout,
        question_type="主观题"
    )


def judge_long_context_answer(question, model_output, judge_api_key, judge_base_url, judge_model, timeout=600):
    """
    长上下文阅读理解题打分：调用打分模型API进行0-10分评分。
    内置重试机制：如果打分模型返回异常或无法提取分数，自动重试最多2次。
    """
    q_text = question["question_text"]
    answer = extract_subjective_answer(model_output)

    if not answer:
        return 0, "模型未输出任何内容"

    judge_prompt = (
        f"你是一位严格的阅读理解评审专家。请根据以下评分标准对模型的回答进行打分。\n\n"
        f"问题：{q_text}\n\n"
        f"模型回答：\n{answer}\n\n"
        "评分标准（0-10分）：\n"
        "- 10分：答案完全正确，基于上下文准确回答，简洁明了\n"
        "- 7-9分：答案基本正确，有少量偏差或不完整\n"
        "- 4-6分：答案部分正确，有较大遗漏或表述不够简洁\n"
        "- 1-3分：答案大部分错误或完全偏离问题\n"
        "- 0分：完全未回答或答非所问\n\n"
        "【重要规则】你必须只输出一个0-10之间的整数分数，"
        "然后换行给出一句简短的评分理由。格式如下：\n"
        "分数\n理由\n"
        "不要输出任何其他内容。"
    )

    return _judge_with_retry(
        judge_prompt, judge_api_key, judge_base_url, judge_model, timeout,
        question_type="长上下文"
    )


def _judge_with_retry(judge_prompt, api_key, base_url, model_name, timeout,
                      question_type="主观题", max_retries=2):
    """
    带重试的评分调用：调用打分模型 → 提取分数 → 如果失败则重试。
    重试时使用更强的格式约束提示，并逐步增加等待时间。
    """
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait = attempt * 3  # 逐步增加: 3s, 6s
            print(f"    评分重试 {attempt}/{max_retries}（等待{wait}s）...")
            time.sleep(wait)
            # 重试时加强格式约束
            retry_suffix = (
                "\n\n【再次强调】你上一次的输出格式不正确。"
                "请严格按照以下格式，只输出两行，不要有任何其他内容：\n"
                "第一行：一个0-10的整数\n"
                "第二行：一句简短的理由"
            )
            current_prompt = judge_prompt + retry_suffix
        else:
            current_prompt = judge_prompt

        raw_judge = ask_model(
            current_prompt, api_key, base_url, model_name,
            timeout=timeout, max_tokens=2048
        )

        if not raw_judge:
            print(f"    打分模型第{attempt+1}次调用返回空")
            continue

        score, comment = _extract_subjective_score(raw_judge)

        if score >= 0:
            return score, comment

        # 提取失败
        print(f"    第{attempt+1}次评分提取失败，原始输出: {raw_judge[:150]}")

    # 所有重试失败 → 降级：返回0分并标记
    print(f"  [评分放弃] {question_type}评分全部失败，标记为0分需人工审核")
    return 0, "[评分失败-需人工审核] 打分模型多次返回不可解析"


def _extract_subjective_score(raw_judge_output):
    """
    从打分模型输出中提取0-10的整数分数和评语。

    增强版：支持多种常见输出格式，包括：
      "8"          纯数字
      "8分"        带"分"字
      "得分：8"     带标签
      "8/10"       分数格式
      "**8**"      markdown 加粗
      "8. 理由..."  编号后跟理由
    返回 (score: int 0-10, comment: str)。如果提取失败返回 (-1, raw_text) 供上层判断。
    """
    if not raw_judge_output:
        return -1, "打分模型未返回结果"

    text = raw_judge_output.strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    score = -1

    # 策略 1: 逐行尝试，取第一个有效数字
    for line in lines:
        # 去除 markdown 标记
        clean = line.strip().lstrip('#').strip('*').strip()

        # 尝试 "得分：8" / "分数：8" / "评分：8" 等模式
        m = re.search(r'(?:得分|分数|评分|总分)\s*[：:]\s*(\d{1,2})', clean)
        if m:
            score = int(m.group(1))
            break

        # 尝试 "8/10" 格式
        m = re.search(r'\b(\d{1,2})\s*/\s*10\b', clean)
        if m:
            score = int(m.group(1))
            break

        # 尝试纯数字 "8" 或 "8分" 或 "8." 开头
        m = re.search(r'\b(\d{1,2})\b', clean)
        if m:
            candidate = int(m.group(1))
            if 0 <= candidate <= 10:
                score = candidate
                break

    # 策略 2: 全文搜索（兜底）
    if score == -1:
        numbers = re.findall(r'\b(\d{1,2})\b', text)
        valid = [int(n) for n in numbers if 0 <= int(n) <= 10]
        if valid:
            score = valid[0]

    # 策略 3: 仍然失败 → 返回 -1 标记，由上层处理
    if score == -1:
        print(f"  [评分警告] 无法从打分模型输出中提取分数，原始输出: {text[:200]}")
        return -1, text[:200]

    score = max(0, min(10, score))

    # 评语提取：跳过第一行（分数行），取剩余内容
    comment_lines = []
    found_score_line = False
    for line in lines:
        clean = line.strip().lstrip('#').strip('*').strip()
        if not found_score_line:
            # 判断当前行是否为分数行
            has_score = bool(re.search(r'\b' + str(score) + r'\b', clean))
            if has_score:
                found_score_line = True
                continue
        comment_lines.append(line)

    comment = ' '.join(comment_lines).strip() if comment_lines else text
    comment = comment[:300]

    return score, comment


# =========================
# 评分调度
# =========================

SCORER_MAP = {
    "multiple_choice": judge_mc_answer,
    "math_fill": judge_math_answer,
    "subjective": judge_subjective_answer,
    "long_context": judge_long_context_answer,
    "visual_multiple_choice": judge_mc_answer,
    "text_to_image": judge_t2i_answer,
}


def score_answer(question, model_output, judge_api_key, judge_base_url, judge_model, timeout=600):
    """统一评分入口"""
    qtype = question.get("type", "multiple_choice")
    scorer = SCORER_MAP.get(qtype, judge_mc_answer)
    return scorer(question, model_output, judge_api_key, judge_base_url, judge_model, timeout)


# 提示词构建映射
PROMPT_BUILDER_MAP = {
    "multiple_choice": build_mc_prompt,
    "math_fill": build_math_prompt,
    "subjective": build_subjective_prompt,
    "long_context": build_longbench_prompt,
    "visual_multiple_choice": build_visual_mc_prompt,
    "text_to_image": build_t2i_prompt,
}

# 答案提取映射
EXTRACTOR_MAP = {
    "multiple_choice": extract_mc_answer,
    "math_fill": extract_math_answer,
    "subjective": extract_subjective_answer,
    "long_context": extract_subjective_answer,
    "visual_multiple_choice": extract_mc_answer,
    # 文生图：模型输出即图片标识（base64 或 URL），无需提取
    "text_to_image": extract_subjective_answer,
}
