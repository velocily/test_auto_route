# config.py
# ============================================================
# 智能路由系统 —— 集中配置文件
# 所有可调参数、路径、地址均在此文件中配置，修改后重启程序即可生效
# ============================================================

import os

# 项目根目录（自动检测：本文件在 router/ 目录下，上级即项目根目录）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. 本地任务分类模型配置
# ============================================================
# 用于任务分类的本地模型路径（可选，WhoEngine 不依赖此模型）
# 如需启用 task_classifier.py 中的本地分类器，请填入本地模型目录
MODEL_PATH = r"/path/to/your/local/model"

# ============================================================
# 2. 路由器行为配置
# ============================================================
ROUTER_CONFIG = {
    # 是否开启调试日志（True=输出详细日志，False=仅输出关键日志）
    "enable_debug_log": True,

    # 是否启用规则（正则/关键词）回退分类
    # False = 纯 LLM 分类，LLM 失败时直接返回默认值 "chat"
    # True  = LLM 分类失败时，使用 normalize_task() 中的关键词规则进行回退匹配
    # 当前为关闭状态，保留正则匹配代码以备后续启用
    "enable_rule_fallback": False,

    # 测试模式（不连接远程服务器，仅验证路由逻辑）
    # True  = 跳过远程模型调用，返回包含路由详情的 mock 响应
    #         （任务分类、模型选型、路由 URL 等全部正常执行并输出日志）
    # False = 正常模式，实际调用远程服务器上的模型
    "test_mode": True,

    # 强制路由模型（由前端勾选设置，留空=自动路由）
    # 设置后所有请求都路由到该模型，跳过 WhoEngine 路由决策
    # 优先级低于用户 @model-name 指令
    "forced_model": "",
}

# ============================================================
# 3. 本程序 API 服务配置
# ============================================================
# 本 FastAPI 程序的监听地址与端口（对外提供路由接口）
SERVICE_CONFIG = {
    "host": "0.0.0.0",          # 监听地址，0.0.0.0 表示接受所有来源的请求
    "port": 8000,               # 监听端口
}

# ============================================================
# 4. 远程模型服务器配置
# ============================================================
#
# 所有模型可共用一个 API 端点，也可分别配置不同地址。
# 模型名通过请求 body 的 "model" 字段区分，不在 URL 路径中。
#
# 请求格式：
#   ✅ POST {model_url}
#      Body: {"model": "model-name", "messages": [...]}
# ============================================================
REMOTE_SERVER_CONFIG = {
    # ---------- 请求超时（秒）----------
    "request_timeout": 120,

    # ---------- SSL 证书校验 ----------
    # 自签名证书服务器需设为 False 跳过校验
    "verify_ssl": False,

    # ---------- 远程端点（自动发现模型，推荐）----------
    # 给出 URL 和 API Key，程序会调用 GET {url}/models 自动检测该端点上正在运行的模型，
    # 无需手动填写 model_routes。
    #
    # URL 可以是基础地址（如 https://your-server:8443/v1），
    # 也可以是完整的 chat completions 地址，程序会自动规范化。
    # api_key 为空字符串表示不携带 Authorization 头（适用于无需鉴权的内网服务）。
    "remote_endpoints": [
        # {"url": "https://your-server:8443/v1", "api_key": "sk-xxxx"},
        # {"url": "https://your-server:8443/v1/chat/completions", "api_key": "sk-yyyy"},
    ],

    # ---------- 模型路由映射（手动配置，向后兼容）----------
    # 键（key）  ：程序内部模型名称，必须与 model_profiles.py 中的名称一致
    #               同时也是发送给服务器的 "model" 字段值
    # 值（value）：该模型的 OpenAI 兼容 API 完整地址
    #
    # 所有模型共用同一个 URL，服务器通过 body 中的 model 字段区分。
    # 如果某个模型暂未部署，注释掉对应行即可。
    # 程序会在日志中提示"模型不存在"，不会报错崩溃。
    #
    # 注意：配置了 remote_endpoints 后，通常无需再手动填写 model_routes。
    # 此处保留用于手动覆盖或无需鉴权的场景。
    "model_routes": {
        # 在此填入你的模型 API 地址（OpenAI 兼容接口）
        # 示例：
        # "model-a": "https://your-server:8443/v1/chat/completions",
        # "model-b": "https://your-server:8443/v1/chat/completions",
    },
}

# ============================================================
# 5. WhoEngine 路由器配置
# ============================================================
# WhoEngine 使用 KNN 近邻软投票 + 岭回归路由器，无需加载 3B 大模型即可做 domain 分类。
# 如果 WhoEngine 初始化失败，程序会自动回退到原有的 Qwen2.5-3B 分类器。

WHOENGINE_CONFIG = {
    # Embedding 模型
    # 选项 1: sentence-transformers 模型（自动从 HF Hub 缓存到本地）
    #   例: "sentence-transformers/all-MiniLM-L6-v2"（约 80MB，速度快，英文优）
    #   例: "BAAI/bge-small-zh-v1.5"（中文更好，需翻墙下载一次）
    # 选项 2: 本地 transformers 模型（不推荐，因果语言模型的 hidden states 不适合做语义相似度）
    #   例: r"D:\Qwen2.5-3B\base"（指向包含 config.json 的目录）
    #   注意：因果语言模型（如 Qwen、GPT）的 hidden states 未经过对比学习优化，
    #   语义聚类能力远弱于专门的 sentence-transformers 模型，不建议使用。
    "embedder": "BAAI/bge-large-zh-v1.5",

    # 岭回归正则化系数 λ，默认 1e-2
    # 若过拟合（训练 domain 内准确但泛化差）→ 增大 λ
    # 若欠拟合 → 减小 λ
    "lambda": 1e-2,

    # 路由策略
    #   "knn_prior":        KNN + 关键词先验混合（推荐，准确率高）
    #   "knn":              KNN 近邻软投票（准确率 83.9%）
    #   "majority_voting": Token 级路由 + entropy top-k + 多数投票（精度最高）
    #   "average":         句级 embedding → 单次 argmax（快速，基线）
    "routing_strategy": "knn_prior",

    # 路由模式
    #   "token":    始终使用 Token 级路由（精度最高）
    #   "sentence": 始终使用句级路由（速度最快）
    #   "auto":     自动选择（短文本用 token 级，长文本用句级）
    "routing_mode": "token",

    # SRS: 取熵最小的 top-k 个 token 投票决定 domain
    "top_k_entropy": 10,

    # KNN 近邻数：实验测得 k=20 在 8 domain 上取得最佳准确率
    # 较大的 k 对边界样本更鲁棒，减少噪声影响
    "knn_k": 20,

    # KNN 相似度温度缩放：将余弦相似度放大后再 softmax，使近邻权重更集中
    "knn_sim_temp": 10.0,

    # KNN + 关键词先验混合系数 alpha
    # 最终概率 = alpha × KNN + (1-alpha) × 关键词先验
    # 实验测得 alpha=0.5 时准确率显著提升（+12.9%）
    # alpha=1.0 退化为纯 KNN，alpha=0.0 退化为纯关键词先验
    "knn_prior_alpha": 0.5,

    # 是否启用 KNN 路由策略
    "use_knn_router": True,

    # 路由器缓存文件（训练一次后自动保存，下次直接加载）
    "cache_file": os.path.join(_PROJECT_ROOT, "whoengine.pt"),

    # 各 benchmark 题目文件路径（用于训练 domain 分类器）
    # 键 = domain 名称（必须与 model_benchmarks.json 中的 benchmark 键一致）
    # 值 = 题目文本文件的路径（相对于项目根目录）
    "benchmark_files": {
        "mmlu": os.path.join(_PROJECT_ROOT, "benchmarks", "mmlu_gsm8k_hellaswag", "mmlu-知识广度(30).txt"),
        "gsm8k": os.path.join(_PROJECT_ROOT, "benchmarks", "mmlu_gsm8k_hellaswag", "gsm8k-数学推理(10).txt"),
        "hellaswag": os.path.join(_PROJECT_ROOT, "benchmarks", "mmlu_gsm8k_hellaswag", "hellaswag-常识推理(20).txt"),
        "bbh_semantic": os.path.join(_PROJECT_ROOT, "benchmarks", "bbh_longbench", "bbh-语义理解(10).txt"),
        "bbh_math": os.path.join(_PROJECT_ROOT, "benchmarks", "bbh_longbench", "bbh-数学推理(10).txt"),
        "longbench": os.path.join(_PROJECT_ROOT, "benchmarks", "bbh_longbench", "longbench-长上下文(10).txt"),
        "project_manager": os.path.join(_PROJECT_ROOT, "benchmarks", "bbh_longbench", "project_manager-项目管理.txt"),
        "secretary": os.path.join(_PROJECT_ROOT, "benchmarks", "bbh_longbench", "secretary-秘书工作.txt"),
        # 如需新增 domain，在此添加路径即可，WhoEngine 支持增量更新
    },

    # ---------- 多模态配置 ----------
    # 多模态任务识别开关：True 时启用多模态任务检测和分支路由
    # 启用后，含图片/视觉关键词的请求将路由到多模态模型
    # 含生图关键词的请求将路由到具备 image_generation 能力的模型
    "enable_multimodal_routing": True,

    # 多模态识图 benchmark 题集文件路径（用于多模态任务难度评估参考）
    # 注意：多模态 domain 不参与 WhoEngine KNN 训练（图片无法直接嵌入），
    #       仅用于配置记录和未来扩展。多模态路由基于 model_benchmarks.json
    #       中的 multimodal 子结构进行模型选型。
    "multimodal_benchmark_files": {
        "chart_qa":   os.path.join(_PROJECT_ROOT, "benchmarks", "multimodal", "chartqa-图表理解(20).txt"),
        "text_vqa":   os.path.join(_PROJECT_ROOT, "benchmarks", "multimodal", "textvqa-文字识别(20).txt"),
        "math_vista": os.path.join(_PROJECT_ROOT, "benchmarks", "multimodal", "mathvista-视觉数学(20).txt"),
        "vqa":        os.path.join(_PROJECT_ROOT, "benchmarks", "multimodal", "vqa-视觉问答(30).txt"),
        "mmmu":       os.path.join(_PROJECT_ROOT, "benchmarks", "multimodal", "mmmu-多模态理解(30).txt"),
    },

    # 文生图（text-to-image）benchmark 题集文件路径
    # 文生图任务通过关键词识别（画一张/生成图片等），路由到具备
    # image_generation 能力的模型。同样不参与 KNN 训练。
    "t2i_benchmark_files": {
        "image_generation": os.path.join(_PROJECT_ROOT, "benchmarks", "multimodal", "t2i-文生图(50).txt"),
    },
}

# ============================================================
# 6. 文件路径配置
# ============================================================
# 所有外部文件的路径统一在此管理

# 模型基准测试数据文件（JSON 格式，记录各模型在 benchmark 上的得分）
BENCHMARK_FILE = os.path.join(_PROJECT_ROOT, "model_benchmarks.json")

# 自动生成的模型能力配置文件（由 generate_model_profiles.py 根据 benchmark 生成）
MODEL_PROFILES_FILE = "model_profiles.py"
