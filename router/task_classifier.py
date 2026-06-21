# task_classifier.py
# ============================================================
# 任务分类器 —— 使用本地 Qwen2.5-3B 模型进行任务类型分类
# 模型在 app.py 启动时通过 startup 事件预加载，避免 import 时重复加载。
# classify_task() 内部有 _ensure_model_loaded() 作为安全兜底。
# ============================================================

import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_PATH, ROUTER_CONFIG

# ---------- 延迟加载：模块级变量初始为 None，首次调用时加载 ----------
_tokenizer = None
_model = None


def is_model_loaded() -> bool:
    """检查模型是否已加载（不会触发加载）"""
    return _model is not None


def _ensure_model_loaded():
    """确保模型已加载（首次调用时加载，后续调用直接返回）"""
    global _tokenizer, _model
    if _model is not None:
        return
    print("Loading router model...")
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    _model.eval()
    print("Router model loaded successfully.")


def load_model():
    """公开的模型加载函数，供 app.py startup 事件调用。
    在服务器启动时预加载模型，确保第一个请求到来前模型已就绪。"""
    _ensure_model_loaded()

# 定义合法的任务类型
VALID_TASKS = {"coding", "summary", "chat", "math", "pm", "secretary"}

# 同义词映射到标准任务名
TASK_SYNONYMS = {
    "project planning": "pm",
    "planning": "pm",
    "project": "pm",
    "代码": "coding",
    "编程": "coding",
    "总结": "summary",
    "汇总": "summary",
    "数学": "math",
    "证明": "math",
    "邮件": "secretary",
    "文档": "secretary",
    "办公": "secretary",
    "闲聊": "chat",
    "对话": "chat",
}

SYSTEM_PROMPT = """
你是AI任务分类器。

你必须严格按照以下任务类型之一输出，不能创造新的类型：

可选类型（必须从中选择）：
- coding: 编程、代码、debug、算法
- summary: 总结、提炼、会议纪要  
- chat: 闲聊、普通对话
- math: 数学、证明、计算
- pm: 项目规划、任务拆解、风险分析、项目路线
- secretary: 邮件、文档、安排、办公

你必须严格输出以下JSON格式，不要添加任何额外内容：
{"task": "任务类型", "difficulty": 1-10, "need_reasoning": true, "need_long_context": true}

其中：
- difficulty: 1-10的整数
- need_reasoning: true或false
- need_long_context: true或false

只输出JSON，不要输出解释，不要输出markdown标记。
"""

def _clean_llm_response(text: str) -> str:
    """清洗 LLM 输出：去掉 markdown 代码块标记和多余空格"""
    # 去掉 ```json 和 ``` 包裹
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()
    # 找到第一个 { 和最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return text


def _robust_parse_json(text: str) -> dict:
    """
    鲁棒 JSON 解析：先清洗再 json.loads，失败则用正则提取。
    这是纯解析逻辑，不涉及关键词规则回退。
    """
    # 1. 清洗后尝试标准解析
    cleaned = _clean_llm_response(text)
    if cleaned:
        try:
            return json.loads(cleaned)
        except:
            pass
    # 2. 正则提取
    result = extract_json(text)
    if result:
        return result
    return None


def normalize_task(task_str: str) -> str:
    """将模型输出的任务名映射到标准任务名"""
    task_lower = task_str.lower().strip()
    
    # 直接匹配
    if task_lower in VALID_TASKS:
        return task_lower
    
    # 同义词映射
    for synonym, standard in TASK_SYNONYMS.items():
        if synonym in task_lower or task_lower in synonym:
            return standard
    
    # 规则回退：仅在 enable_rule_fallback 为 True 时启用模糊关键词匹配
    if ROUTER_CONFIG.get("enable_rule_fallback", False):
        if any(word in task_lower for word in ["code", "coding", "程序", "debug"]):
            return "coding"
        if any(word in task_lower for word in ["summary", "summar", "总结"]):
            return "summary"
        if any(word in task_lower for word in ["math", "数学", "计算"]):
            return "math"
        if any(word in task_lower for word in ["pm", "project", "项目", "规划", "管理"]):
            return "pm"
        if any(word in task_lower for word in ["secretary", "邮件", "办公", "文档"]):
            return "secretary"
    
    # 默认返回chat
    return "chat"

def extract_json(text: str) -> dict:
    """从可能包含额外内容的文本中提取JSON"""
    # 尝试匹配JSON对象
    json_pattern = r'\{[^{}]*"task"[^{}]*\}'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    # 如果失败，尝试手动解析
    task_match = re.search(r'"task"\s*:\s*"([^"]+)"', text)
    difficulty_match = re.search(r'"difficulty"\s*:\s*(\d+)', text)
    reasoning_match = re.search(r'"need_reasoning"\s*:\s*(true|false)', text, re.IGNORECASE)
    context_match = re.search(r'"need_long_context"\s*:\s*(true|false)', text, re.IGNORECASE)
    
    result = {}
    if task_match:
        result["task"] = task_match.group(1)
    if difficulty_match:
        result["difficulty"] = int(difficulty_match.group(1))
    if reasoning_match:
        result["need_reasoning"] = reasoning_match.group(1).lower() == "true"
    if context_match:
        result["need_long_context"] = context_match.group(1).lower() == "true"
    
    return result if result else None

def classify_task(user_prompt):
    # 延迟加载模型（首次调用时加载，后续直接复用）
    _ensure_model_loaded()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt"
    ).to(_model.device)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,  # 减少重复
        )

    response = _tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # 尝试解析JSON（鲁棒解析：自动处理 markdown 标记、额外文本等）
    result = _robust_parse_json(response)
    if result is None:
        # 规则回退：仅在 enable_rule_fallback 为 True 时启用关键词匹配
        if ROUTER_CONFIG.get("enable_rule_fallback", False):
            # 用 normalize_task 的关键词规则兜底
            task_guess = normalize_task(response)
            result = {
                "task": task_guess,
                "difficulty": 5,
                "need_reasoning": False,
                "need_long_context": False,
            }
        else:
            # 纯 LLM 模式：LLM 输出解析失败，返回默认值
            result = {"task": "chat", "difficulty": 5,
                      "need_reasoning": False, "need_long_context": False}
    
    # 标准化任务名称
    if "task" in result:
        result["task"] = normalize_task(result["task"])
    else:
        result["task"] = "chat"
    
    # 确保difficulty在1-10范围内
    difficulty = result.get("difficulty", 5)
    result["difficulty"] = max(1, min(10, int(difficulty)))
    
    # 确保布尔字段存在
    result.setdefault("need_reasoning", False)
    result.setdefault("need_long_context", False)
    
    return result
