# routing_params.py
# ============================================================
# 路由参数集中管理模块
# ------------------------------------------------------------
# 所有影响路由决策的可调参数集中在此处，支持：
#   1. 运行时读取（whoengine.py、router_engine.py 等模块共享）
#   2. 运行时修改（通过 /dashboard 可视化界面或 /api/params 接口）
#   3. 参数校验（min/max 范围限制，防止非法值）
#   4. 默认值预设（"efficiency_first" / "balanced" / "accuracy_first"）
#   5. 变更日志（每次保存自动追加到 params_changes.log，记录时间与参数值）
#
# 参数分组：
#   A. 选型权重 (domain_alpha)        —— 各 domain 的能力/效率权衡
#   B. 难度自适应 (difficulty_adjust)  —— 根据难度动态调整 alpha
#   C. 效率归一化 (efficiency)         —— TPS/延迟/并发的归一化参数
#   D. KNN 路由  (knn)                 —— KNN 近邻路由参数
#   E. 关键词先验 (prior)              —— KNN + 关键词先验混合系数
#   F. 难度评估 (difficulty)           —— 各 domain 基础难度与置信度影响
# ============================================================

import os
import json
import threading
import copy
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 变更日志文件路径（与本模块同目录）
_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_changes.log")

# ============================================================
# 参数定义（含范围、默认值、说明）
# ============================================================
# 每个参数的元数据：key -> (default, min, max, step, category, description)
PARAM_META = {
    # ---------- A. 选型权重 α ----------
    # α 越大越看重 benchmark 能力，越小越看重效率
    # 公式：combined = α × benchmark[domain] + (1-α) × efficiency
    # 默认值依据：数学/语义类任务对模型能力要求高（α≥0.75），
    #           知识/长文类任务适中（α≈0.45-0.65），
    #           常识/应用类任务多数模型都能胜任，效率优先（α≤0.40）
    "domain_alpha.mmlu":             (0.50, 0.10, 0.95, 0.01, "选型权重", "MMLU 知识广度：能力/效率权重"),
    "domain_alpha.gsm8k":            (0.80, 0.10, 0.95, 0.01, "选型权重", "GSM8K 数学推理：能力/效率权重"),
    "domain_alpha.hellaswag":        (0.35, 0.10, 0.95, 0.01, "选型权重", "HellaSwag 常识推理：能力/效率权重"),
    "domain_alpha.bbh_semantic":     (0.75, 0.10, 0.95, 0.01, "选型权重", "BBH 语义理解：能力/效率权重"),
    "domain_alpha.bbh_math":         (0.85, 0.10, 0.95, 0.01, "选型权重", "BBH 数学推理：能力/效率权重"),
    "domain_alpha.longbench":        (0.60, 0.10, 0.95, 0.01, "选型权重", "LongBench 长上下文：能力/效率权重"),
    "domain_alpha.project_manager":  (0.45, 0.10, 0.95, 0.01, "选型权重", "项目经理：能力/效率权重"),
    "domain_alpha.secretary":        (0.35, 0.10, 0.95, 0.01, "选型权重", "秘书工作：能力/效率权重"),

    # ---------- B. 难度自适应 ----------
    # 根据题目难度动态调整 α：难题加 α，简单题减 α
    # difficulty >= 8  →  α += boost_high
    # difficulty >= 6  →  α += boost_medium
    # difficulty <= 5  →  α -= reduce_mid_low
    # difficulty <= 3  →  α -= reduce_low
    "difficulty_adjust.boost_high":      (0.15, 0.00, 0.30, 0.01, "难度自适应", "高难度题（≥8）α 提升量"),
    "difficulty_adjust.boost_medium":    (0.08, 0.00, 0.20, 0.01, "难度自适应", "中等难度题（≥6）α 提升量"),
    "difficulty_adjust.reduce_mid_low":  (0.10, 0.00, 0.25, 0.01, "难度自适应", "简单题（≤5）α 降低量"),
    "difficulty_adjust.reduce_low":      (0.20, 0.00, 0.40, 0.01, "难度自适应", "极简题（≤3）α 降低量"),
    # α 的上下限（防止极端值）
    "difficulty_adjust.alpha_min":       (0.20, 0.05, 0.50, 0.01, "难度自适应", "α 下限（效率最大权重）"),
    "difficulty_adjust.alpha_max":       (0.95, 0.50, 1.00, 0.01, "难度自适应", "α 上限（能力最大权重）"),

    # ---------- C. 效率归一化 ----------
    # efficiency = w_tps × norm(tps) + w_lat × norm(latency) + w_con × norm(concurrency)
    # 三个权重之和无须等于 1，但建议总和接近 1 以保持归一化效率分在 [0,1]
    "efficiency.weight_tps":         (0.40, 0.00, 1.00, 0.01, "效率归一化", "吞吐量 TPS 权重"),
    "efficiency.weight_latency":     (0.30, 0.00, 1.00, 0.01, "效率归一化", "延迟权重"),
    "efficiency.weight_concurrency": (0.30, 0.00, 1.00, 0.01, "效率归一化", "并发能力权重"),
    # 饱和点：达到该值即得满分
    "efficiency.tps_max":            (60.0,  10.0, 300.0, 1.0,  "效率归一化", "TPS 饱和点（tok/s，≥此值得满分）"),
    "efficiency.latency_max":        (3.0,   0.5,  10.0,  0.1,  "效率归一化", "延迟饱和点（秒，≥此值得 0 分）"),
    "efficiency.concurrency_max":    (20.0,  5.0,  100.0, 1.0,  "效率归一化", "并发饱和点（≥此值得满分）"),
    # 无效率数据时的默认效率分
    "efficiency.default_score":      (0.5,   0.0,  1.0,  0.01, "效率归一化", "无效率数据时的默认效率分"),

    # ---------- D. KNN 路由 ----------
    # k=20 在准确率和鲁棒性间取得平衡；sim_temp=10.0 使近邻权重适度集中
    "knn.k":                         (20,    1,    50,   1,    "KNN 路由",   "KNN 近邻数 k（越大越鲁棒）"),
    "knn.sim_temp":                  (10.0,  1.0,  30.0, 0.5,  "KNN 路由",   "相似度温度（越大近邻权重越集中）"),

    # ---------- E. 关键词先验混合 ----------
    # 最终概率 = α × KNN + (1-α) × 关键词先验
    # 0.7 偏向 KNN（数据驱动），保留 30% 关键词先验作为冷启动兜底
    "prior.alpha":                   (0.7,   0.0,  1.0,  0.01, "关键词先验", "KNN/先验混合系数（1.0=纯KNN，0.0=纯先验）"),

    # ---------- F. 难度评估 ----------
    # 每个 domain 的基础难度（1-10），影响 _get_adaptive_alpha 的输入
    # 依据各 benchmark 的实际难度设定
    "difficulty.base.mmlu":            (5, 1, 10, 1, "难度评估", "MMLU 基础难度"),
    "difficulty.base.gsm8k":           (7, 1, 10, 1, "难度评估", "GSM8K 基础难度"),
    "difficulty.base.hellaswag":       (3, 1, 10, 1, "难度评估", "HellaSwag 基础难度"),
    "difficulty.base.bbh_semantic":    (6, 1, 10, 1, "难度评估", "BBH 语义理解基础难度"),
    "difficulty.base.bbh_math":        (8, 1, 10, 1, "难度评估", "BBH 数学推理基础难度"),
    "difficulty.base.longbench":       (6, 1, 10, 1, "难度评估", "LongBench 基础难度"),
    "difficulty.base.project_manager": (5, 1, 10, 1, "难度评估", "项目经理基础难度"),
    "difficulty.base.secretary":       (3, 1, 10, 1, "难度评估", "秘书工作基础难度"),
    # 置信度对难度的影响：delta = (confidence_base - confidence) × confidence_factor
    "difficulty.confidence_base":      (0.5, 0.0, 1.0,  0.01, "难度评估", "置信度基准值（高于此值不加分）"),
    "difficulty.confidence_factor":    (4,   1,    10,  1,    "难度评估", "置信度对难度的影响系数"),
}

# ============================================================
# 预设方案
# ============================================================
PRESETS = {
    # 效率优先：80% 任务使用高效模型，仅极难题才用专家模型
    "efficiency_first": {
        "domain_alpha.mmlu": 0.35, "domain_alpha.gsm8k": 0.70,
        "domain_alpha.hellaswag": 0.20, "domain_alpha.bbh_semantic": 0.65,
        "domain_alpha.bbh_math": 0.80, "domain_alpha.longbench": 0.55,
        "domain_alpha.project_manager": 0.30, "domain_alpha.secretary": 0.20,
        "difficulty_adjust.boost_high": 0.20, "difficulty_adjust.boost_medium": 0.10,
        "difficulty_adjust.reduce_mid_low": 0.15, "difficulty_adjust.reduce_low": 0.30,
        "difficulty_adjust.alpha_min": 0.10, "difficulty_adjust.alpha_max": 0.95,
    },
    # 平衡：使用 PARAM_META 中的默认值（推荐）
    "balanced": {key: meta[0] for key, meta in PARAM_META.items()},
    # 能力优先：尽量选能力最强的模型
    "accuracy_first": {
        "domain_alpha.mmlu": 0.75, "domain_alpha.gsm8k": 0.90,
        "domain_alpha.hellaswag": 0.65, "domain_alpha.bbh_semantic": 0.90,
        "domain_alpha.bbh_math": 0.95, "domain_alpha.longbench": 0.85,
        "domain_alpha.project_manager": 0.70, "domain_alpha.secretary": 0.65,
        "difficulty_adjust.boost_high": 0.10, "difficulty_adjust.boost_medium": 0.05,
        "difficulty_adjust.reduce_mid_low": 0.05, "difficulty_adjust.reduce_low": 0.10,
        "difficulty_adjust.alpha_min": 0.40, "difficulty_adjust.alpha_max": 0.98,
    },
}

# 默认预设：balanced（在能力和效率间取得平衡，适合大多数场景）
DEFAULT_PRESET = "balanced"


# ============================================================
# 运行时参数存储（线程安全，仅内存，不持久化）
# ============================================================
class _ParamStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._values = {}
        self._reset_to_preset(DEFAULT_PRESET, log=False)

    def _reset_to_preset(self, preset_name: str, log: bool = True):
        """重置为指定预设"""
        if preset_name not in PRESETS:
            logger.warning("预设 '%s' 不存在，使用 balanced", preset_name)
            preset_name = "balanced"
        with self._lock:
            self._values = copy.deepcopy(PRESETS[preset_name])
            # 补全未在预设中显式声明的参数（用 PARAM_META 默认值）
            for key, meta in PARAM_META.items():
                if key not in self._values:
                    self._values[key] = meta[0]
            self._current_preset = preset_name
        if log:
            self._append_log({"action": "apply_preset", "preset": preset_name})

    def _append_log(self, entry: dict):
        """将一次变更追加到日志文件"""
        try:
            entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self._lock:
                snapshot = copy.deepcopy(self._values)
            entry["params"] = snapshot
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info("参数变更已记录到日志 %s", _LOG_FILE)
        except Exception as e:
            logger.warning("写入参数变更日志失败: %s", e)

    def get(self, key: str, default=None):
        with self._lock:
            return self._values.get(key, default)

    def get_all(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._values)

    def _coerce(self, key: str, value):
        """类型转换 + 范围裁剪"""
        if key not in PARAM_META:
            raise KeyError(f"未知参数: {key}")
        _, vmin, vmax, _, _, _ = PARAM_META[key]
        if isinstance(vmin, int) and isinstance(vmax, int):
            value = int(round(value))
        else:
            value = float(value)
        if value < vmin:
            value = vmin
        elif value > vmax:
            value = vmax
        return value

    def set(self, key: str, value):
        """设置单个参数，自动校验范围（仅内存，重启后恢复默认）"""
        value = self._coerce(key, value)
        with self._lock:
            self._values[key] = value
            self._current_preset = "custom"
        self._append_log({"action": "set", "changes": {key: value}})

    def set_many(self, updates: dict):
        """批量设置参数（仅内存，重启后恢复默认）"""
        changes = {}
        for k, v in updates.items():
            v = self._coerce(k, v)
            changes[k] = v
            with self._lock:
                self._values[k] = v
        with self._lock:
            self._current_preset = "custom"
        self._append_log({"action": "set_many", "changes": changes})

    def apply_preset(self, preset_name: str):
        """应用预设方案"""
        self._reset_to_preset(preset_name, log=True)

    def current_preset(self) -> str:
        with self._lock:
            return self._current_preset

    def reset(self):
        """重置为默认预设"""
        self._reset_to_preset(DEFAULT_PRESET, log=True)


# 全局单例
_store = _ParamStore()


# ============================================================
# 对外接口函数
# ============================================================
def get_param(key: str, default=None):
    """获取单个参数值"""
    return _store.get(key, default)


def get_all_params() -> dict:
    """获取所有参数（key → value）"""
    return _store.get_all()


def set_param(key: str, value):
    """设置单个参数（自动范围校验）"""
    _store.set(key, value)


def set_params(updates: dict):
    """批量设置参数"""
    _store.set_many(updates)


def apply_preset(preset_name: str):
    """应用预设方案"""
    _store.apply_preset(preset_name)


def reset_params():
    """重置为默认预设"""
    _store.reset()


def current_preset() -> str:
    """当前预设名"""
    return _store.current_preset()


def get_param_meta() -> dict:
    """获取参数元数据（用于前端渲染滑块）"""
    result = {}
    for key, (default, vmin, vmax, vstep, category, desc) in PARAM_META.items():
        result[key] = {
            "default": default,
            "min": vmin,
            "max": vmax,
            "step": vstep,
            "category": category,
            "description": desc,
            "current": _store.get(key),
        }
    return result


def get_presets() -> dict:
    """获取所有预设方案"""
    return copy.deepcopy(PRESETS)


# ============================================================
# 便捷访问函数（供 whoengine.py 等模块使用）
# ============================================================
def get_domain_alpha(domain: str) -> float:
    """获取指定 domain 的基础 α"""
    return _store.get(f"domain_alpha.{domain}", 0.5)


def get_difficulty_adjust_params() -> dict:
    """获取难度自适应参数"""
    return {
        "boost_high":     _store.get("difficulty_adjust.boost_high"),
        "boost_medium":   _store.get("difficulty_adjust.boost_medium"),
        "reduce_mid_low": _store.get("difficulty_adjust.reduce_mid_low"),
        "reduce_low":     _store.get("difficulty_adjust.reduce_low"),
        "alpha_min":      _store.get("difficulty_adjust.alpha_min"),
        "alpha_max":      _store.get("difficulty_adjust.alpha_max"),
    }


def get_efficiency_params() -> dict:
    """获取效率归一化参数"""
    return {
        "w_tps":         _store.get("efficiency.weight_tps"),
        "w_latency":     _store.get("efficiency.weight_latency"),
        "w_concurrency": _store.get("efficiency.weight_concurrency"),
        "tps_max":       _store.get("efficiency.tps_max"),
        "latency_max":   _store.get("efficiency.latency_max"),
        "concurrency_max": _store.get("efficiency.concurrency_max"),
        "default_score": _store.get("efficiency.default_score"),
    }


def get_knn_params() -> dict:
    """获取 KNN 路由参数"""
    return {
        "k":        _store.get("knn.k"),
        "sim_temp": _store.get("knn.sim_temp"),
    }


def get_prior_alpha() -> float:
    """获取关键词先验混合系数"""
    return _store.get("prior.alpha")


def get_difficulty_base(domain: str) -> int:
    """获取指定 domain 的基础难度值"""
    return _store.get(f"difficulty.base.{domain}", 5)


def get_difficulty_confidence_params() -> dict:
    """获取难度评估的置信度参数"""
    return {
        "confidence_base":   _store.get("difficulty.confidence_base"),
        "confidence_factor": _store.get("difficulty.confidence_factor"),
    }
