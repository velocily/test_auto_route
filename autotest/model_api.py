"""
============================================================================
模型 API 调用模块
============================================================================
提供统一的 ask_model() 接口，兼容 OpenAI 风格 API。
同时封装 提示词构建 与 答案提取 逻辑。
"""
import re
import time
import requests

# =========================
# 持久化 Session（连接池复用，避免每次请求 TLS 握手）
# =========================
_SESSION = None


def _get_session():
    """获取或创建持久化 requests.Session，复用 TCP 连接"""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
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
            return r.json()["choices"][0]["message"]["content"]
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


# =========================
# 提示词构建
# =========================

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

    # 2. 尝试匹配以 A/B/C/D 开头的行
    m = re.search(r'(?:^|\n)\s*([A-D])\s*(?:[.\s,]|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. 提取文本中最后一个独立出现的 A/B/C/D
    letters = re.findall(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', text, re.IGNORECASE)
    if letters:
        return letters[-1].upper()

    # 4. 兜底：取第一个字符
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
            timeout=timeout, max_tokens=300
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
}

# 答案提取映射
EXTRACTOR_MAP = {
    "multiple_choice": extract_mc_answer,
    "math_fill": extract_math_answer,
    "subjective": extract_subjective_answer,
    "long_context": extract_subjective_answer,
}
