# 更新日志

本项目所有重要变更记录于此文件。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased] - 2026-06-22

### 项目总体变更概述

本次更新聚焦于**路由参数管理机制优化**和**默认参数调优**，主要变更包括：

1. **移除参数持久化机制**：原方案将修改后的参数写入 `params_runtime.json`，重启后自动加载。改为**仅内存生效 + 日志记录**模式，重启后恢复默认预设，更符合"调参实验"的使用场景。
2. **新增变更日志**：每次保存参数修改时，自动追加一条记录到 `router/params_changes.log`，包含时间戳、操作类型（set/set_many/apply_preset/reset）和参数值快照，便于审计和回溯。
3. **默认参数优化**：将默认预设从 `efficiency_first` 改为 `balanced`，并调整多个参数的默认值，使开箱即用的效果更均衡。
4. **文档完善**：更新 README.md、项目说明文档.md、README_EN.md，新增调参界面 PNG 截图，新增 CHANGELOG.md。

---

### 各文件修改详情

#### 1. `router/routing_params.py`（核心参数管理模块）

**变更类型**：重构

**变更内容**：

- **移除持久化机制**：
  - 删除 `_save_to_file()` 和 `_load_from_file()` 方法
  - 删除 `clear_persisted_params()` 对外接口
  - 删除模块级常量 `_PARAMS_FILE`
  - 启动时不再尝试加载持久化文件，直接使用默认预设

- **新增变更日志机制**：
  - 新增模块级常量 `_LOG_FILE`（指向 `router/params_changes.log`）
  - 新增 `_append_log(entry)` 方法，将变更记录以 JSON Lines 格式追加到日志文件
  - 每条日志包含：时间戳、操作类型、变更内容、当前参数全量快照
  - 在 `set()`、`set_many()`、`apply_preset()`、`reset()` 中自动调用日志记录

- **默认参数优化**（`PARAM_META` 和 `PRESETS["balanced"]`）：

  | 参数 | 原值 | 新值 | 调整理由 |
  |------|------|------|----------|
  | `domain_alpha.mmlu` | 0.45 | **0.50** | 知识广度任务，略微提升能力权重 |
  | `domain_alpha.hellaswag` | 0.30 | **0.35** | 常识推理，适度提升 |
  | `domain_alpha.longbench` | 0.65 | **0.60** | 长上下文，略微降低能力权重 |
  | `domain_alpha.project_manager` | 0.40 | **0.45** | 项目经理任务，略微提升 |
  | `domain_alpha.secretary` | 0.30 | **0.35** | 秘书工作，略微提升 |
  | `difficulty_adjust.alpha_min` | 0.15 | **0.20** | 提高 α 下限，避免过度偏向效率 |
  | `prior.alpha` | 0.5 | **0.7** | 偏向 KNN（数据驱动），保留 30% 关键词先验作冷启动兜底 |

- **默认预设变更**：`DEFAULT_PRESET` 从 `"efficiency_first"` 改为 `"balanced"`

- **代码重构**：
  - 提取 `_coerce(key, value)` 方法，统一类型转换和范围裁剪逻辑（原 `set` 和 `set_many` 中重复代码）
  - 简化 `_reset_to_preset()` 签名：`save: bool` → `log: bool`
  - 新增 `from datetime import datetime` 导入

- **模块头部注释更新**：第 5 点从"持久化保存"改为"变更日志"

---

#### 2. `router/app.py`（API 接口层）

**变更类型**：修改

**变更内容**：

- **`POST /api/params` 接口**：
  - 接口文档注释从"持久化到 params_runtime.json，重启后保留"改为"记录到 params_changes.log"
  - 日志信息从"路由参数已更新并持久化"改为"路由参数已更新并记录日志"
  - 返回值字段 `persisted: True` 改为 `logged: True`

---

#### 3. `router/static/dashboard.html`（前端界面）

**变更类型**：修改

**变更内容**：

- **保存成功提示**：从"（已持久化，重启后保留）"改为"（已记录到日志）"
- **底部说明栏**：从"调节滑块后点击「保存并应用」生效 · 所有参数自动限制在合法范围内"改为"调节滑块后点击「保存并应用」即时生效 · 修改将记录到 params_changes.log · 重启后恢复默认预设"

---

#### 4. `README.md`（项目说明 - 中文）

**变更类型**：修改

**变更内容**：

- **默认预设说明**：从"默认预设：效率优先"改为"默认预设：平衡（balanced）"，并补充日志记录说明
- **界面截图**：从 `dashboard_preview.svg` 改为 `dashboard_preview.png`
- **功能列表**：新增"变更日志"功能项；预设方案中标注"平衡（默认）"
- **参数清单表**：更新默认值列，反映 balanced 预设的实际值
- **配置指南**：参数管理描述从"28 个参数 + 3 套预设"改为"34 个参数 + 3 套预设 + 变更日志"
- **WhoEngine 配置示例**：`embedder` 从 `bge-large-zh-v1.5` 改为 `bge-small-zh-v1.5`；`knn_prior_alpha` 从 `0.5` 改为 `0.7`

---

#### 5. `项目说明文档.md`（详细技术文档）

**变更类型**：修改

**变更内容**：

- **9.1 选型权重表**：列名从"efficiency_first 默认"改为"balanced 默认"，更新所有 domain_alpha 默认值
- **9.1 策略说明**：从"效率优先策略"改为"平衡策略"
- **9.2 难度自适应表**：更新 boost_high、boost_medium、reduce_mid_low、reduce_low、alpha_min 的默认值
- **9.2 关键词先验表**：`prior.alpha` 默认值从 0.5 改为 0.7，新增默认值说明
- **9.3 预设方案表**：标注 `balanced` 为"**默认预设**"，`efficiency_first` 改为"生产环境、成本敏感"
- **9.3 新增说明**：参数修改仅保存在内存中，重启后恢复默认预设；每次修改自动记录到日志
- **9.4 界面截图**：从 SVG 改为 PNG
- **9.4 界面布局图**：预设名从 `efficiency_first` 改为 `balanced`；mmlu 值从 0.35 改为 0.50；新增"修改记录到 params_changes.log"提示
- **9.4 功能说明**：第 3 点新增"并追加日志到 params_changes.log"；第 5 点重置目标改为 `balanced`
- **9.6 章节标题**：从"效率优先策略说明"改为"平衡策略说明"
- **9.6 策略说明**：更新所有 domain α 值，更新难度自适应调整量，更新 α 下限保护值，更新效果示例

---

#### 6. `README_EN.md`（项目说明 - 英文）

**变更类型**：修改

**变更内容**：

- **路由算法说明**：`knn_prior_alpha` 从 "α=0.5 works best" 改为 "α=0.7 by default, biased toward KNN"
- **配置指南**：新增 `routing_params.py` 和 `dashboard.html` 到关键配置文件表
- **WhoEngine 配置示例**：`embedder` 从 `bge-large-zh-v1.5` 改为 `bge-small-zh-v1.5`；`knn_prior_alpha` 从 `0.5` 改为 `0.7`

---

#### 7. `docs/images/dashboard_preview.png`（新增）

**变更类型**：新增

**变更内容**：调参界面预览图（PNG 格式，1280×1700），展示 6 大参数分组、滑块控件、预设按钮、底部操作栏的完整布局。

---

#### 8. `CHANGELOG.md`（新增）

**变更类型**：新增

**变更内容**：本文件，记录本次所有变更。

---

### 日志文件格式说明

`router/params_changes.log` 采用 JSON Lines 格式，每行一条记录：

```json
{"action": "set", "changes": {"knn.k": 25}, "timestamp": "2026-06-22 21:00:00", "params": {"domain_alpha.mmlu": 0.5, "...": "..."}}
{"action": "apply_preset", "preset": "balanced", "timestamp": "2026-06-22 21:01:00", "params": {"...": "..."}}
{"action": "reset", "timestamp": "2026-06-22 21:02:00", "params": {"...": "..."}}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | string | 操作类型：`set` / `set_many` / `apply_preset` / `reset` |
| `changes` | object | 仅 `set` / `set_many` 时存在，记录本次修改的参数键值对 |
| `preset` | string | 仅 `apply_preset` 时存在，记录应用的预设名 |
| `timestamp` | string | 操作时间，格式 `YYYY-MM-DD HH:MM:SS` |
| `params` | object | 操作完成后的全量参数快照 |

---

### 升级指南

如果你正在从旧版本升级：

1. **删除旧的持久化文件**（如果存在）：
   ```bash
   rm router/params_runtime.json
   ```

2. **重启服务**：参数将自动使用新的 `balanced` 默认预设。

3. **查看历史变更**（如果之前生成过日志）：
   ```bash
   cat router/params_changes.log
   ```

4. **如需恢复 `efficiency_first` 预设**：在调参台点击"⚡ 效率优先"按钮，或调用 API：
   ```bash
   curl -X POST http://localhost:8000/api/params/preset -H "Content-Type: application/json" -d '{"preset":"efficiency_first"}'
   ```
