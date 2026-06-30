# 更新日志

本项目所有重要变更记录于此文件。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased] - 2026-06-30

### 远程端点自动发现 + 并发支持 + 界面优化

本次更新聚焦于**自动发现模型**、**多用户并发安全**和**界面体验优化**，主要变更包括：

1. **远程端点自动发现模型（推荐方式）**：新增 `remote_endpoints` 配置，用户只需提供 URL 和 API Key，程序自动调用 `GET {url}/models` 检测端点上正在运行的模型，无需手动填写模型名。URL 支持基础地址（如 `https://xxx/v1`）或完整 chat 地址，程序自动规范化。手动 `model_routes` 保留为折叠的"高级选项"，向后兼容。
2. **并发访问支持**：将路由推理和非流式远程调用改为异步执行，支持多用户并发访问：
   - `route_messages()` → `await asyncio.to_thread(route_messages, ...)`：KNN 推理在线程池中执行，不阻塞事件循环
   - 新增 `call_remote_model_async()`：使用共享的 `httpx.AsyncClient`（线程安全、连接池复用），替代同步的 `call_remote_model()`
   - 流式调用 `stream_remote_model()` 本已是 async，无需改动
   - **回答不会混掉**：每个请求有独立的闭包和 HTTP 连接，完全隔离
3. **模式切换自动刷新**：修复测试模式 ↔ 正常模式切换时模型信息不自动刷新的问题。`onRouteModeChange()` 改为 async，等后端模式保存成功后再调用 `loadRouteModels()` 刷新，确保显示的模型列表与当前模式一致。
4. **强制路由勾选框**：模型信息列表中每个模型旁新增"强制"勾选框，勾选后所有请求强制路由到该模型（优先级仅次于 `@模型名` 指令）。勾选的模型行高亮为黄色，视觉反馈清晰。
5. **未测试模型兜底处理**：当所有检测到的模型都未测试时，自动兜底到第一个可用模型并在回复中提示 `[系统提示] 当前无已测试模型匹配，已兜底路由到: {模型名}`。正常模式下若无任何可用模型，返回 503 错误并提示"未检测到模型"。测试模式下返回"路由选择：{模型名}"格式的路由详情。
6. **Embedder 修复**：将 `config.py` 中 `embedder` 从 `bge-small-zh-v1.5`（512 维）恢复为 `bge-large-zh-v1.5`（1024 维），与 `whoengine.pt` 检查点训练时使用的模型一致。此前的不一致会导致 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1306x3072 and 1536x1)`。
7. **文档与可视化图片同步更新**：更新所有说明文档以反映新功能，并重新生成 Dashboard 预览图：
   - `README.md` / `README_EN.md`：核心特性新增自动发现/并发安全/强制路由；快速开始配置改为 `remote_endpoints` 推荐；API 接口表新增 `discover` 和 `forced-model`
   - `项目说明文档.md`：核心能力表新增 3 项；API 表新增 `discover`/`forced-model` 接口；新增 FAQ 11.11 多用户并发访问说明
   - `docs/Web_UI操作手册.md`：4.1 节新增远程端点/手动路由折叠/保存按钮说明/强制路由勾选框；运行模式开关补充自动刷新说明；新增 FAQ Q7/Q8/Q9
   - `docs/images/home_page.png` / `test_page.png` / `dashboard_1.png` ~ `dashboard_3.png`：新增 5 张界面截图（首页、测试页、调参台 3 张），替代旧的 `dashboard_preview`
   - `docs/项目说明文档.docx`：由 .md 重新生成（含 9 张嵌入图片）

#### 涉及文件

**新增**：
- `benchmarks/multimodal/`：5 类识图题库（ChartQA/TextVQA/MathVista/VQA/MMMU，共 120 题）+ 1 类文生图题库（T2I，50 题）+ 120 张配图 + `generate_images.py` 生成脚本
- `router/static/home.html`：Web 首页（选择测试/路由入口）
- `router/static/test.html`：Web 模型评测界面
- `router/test_route.py`：路由准确率测试脚本
- `docs/Web_UI操作手册.md`：Web UI 详细操作手册
- `docs/项目说明文档.docx` / `docs/功能清单.docx`：Word 版文档
- `docs/images/home_page.png` / `test_page.png`：首页和测试页界面截图
- `docs/images/dashboard_1.png` ~ `dashboard_3.png`：调参台界面截图（3 张）
- `docs/images/system_architecture.svg` / `routing_flow.png` / `params_architecture.png`：架构图 PNG 版本
- `workflow.png`：工作流程图 PNG 版本

**修改**：
- `autotest/config.py`：新增多模态/文生图开关和题库路径配置
- `autotest/main.py`：新增模块化选测、多模态测试、能力探测逻辑
- `autotest/model_api.py`：新增多模态和文生图 API 调用函数
- `autotest/parser.py`：新增多模态题集解析器
- `autotest/benchmarks_json.py`：`multimodal` 子结构按能力类型分组、`capability_status` 字段、增量合并
- `autotest/benchmark_efficiency.py`：并发策略和吞吐计算优化
- `autotest/utils.py`：结果导出适配多模态
- `router/config.py`：新增 `remote_endpoints`、多模态路由配置；修复 `embedder`
- `router/app.py`：路由推理和非流式调用改为异步；新增 `/api/route/discover`、`/api/route/forced-model` 端点
- `router/model_client.py`：新增异步调用和远程端点自动发现
- `router/router_engine.py`：新增 `route_messages` 函数
- `router/whoengine.py`：新增多模态路由逻辑（识图/生图分支）
- `router/routing_params.py`：新增多模态路由参数
- `router/static/dashboard.html`：新增路由服务配置 UI、强制路由勾选框
- `run.py`：新增 `--modules`、`--model`、`--num-samples` 等命令行参数
- `model_benchmarks.json`：数据结构更新为能力类型分组
- `README.md` / `README_EN.md`：核心特性、配置指南、API 接口表同步更新
- `项目说明文档.md`：多模态测试、路由、Web UI 章节同步更新
- `workflow.svg` / `docs/images/system_architecture.png` / `routing_flow.svg` / `params_architecture.svg`：架构图重新生成

**删除**：
- `docs/images/dashboard_preview.png` / `.svg`：被新截图替代

---

## [Unreleased] - 2026-06-25

### Web UI + 效率测试优化

1. **Web 首页（/home）**：新增首页，用户可选择进入「模型评测」或「路由调参」。启动服务后浏览器默认打开首页。
2. **模型评测 UI（/test）**：新增 Web 测试界面，支持在浏览器中填写 API 信息、选择测试模块、设置采样题数和能力探测开关，点击开始后实时查看运行日志，支持停止。测试完成后结果自动增量写入 `model_benchmarks.json`。
3. **测试运行 API**：新增 `/api/test/config`（获取默认配置）、`/api/test/run`（启动测试子进程）、`/api/test/status`（轮询状态和增量日志）、`/api/test/stop`（停止测试）。
4. **效率测试并发策略优化**：`EFFICIENCY_MAX_CONCURRENCY` 从 32 改为 `None`（无上限），32 之后每次 +10 递增（42, 52, 62...），直到吞吐跌穿 65% 阈值或全部失败，安全上限 200。
5. **效率测试吞吐计算说明**：明确吞吐 = 平均每请求 token 输出速率（token/s），对多轮/多并发请求取算术平均。
6. **调参台导航**：dashboard.html 顶部新增「首页」「模型评测」链接。
7. **文档更新**：README.md / README_EN.md 新增 Web UI 章节和效率测试说明；API 接口表新增测试相关接口。

---

## [Unreleased] - 2026-06-25

### 模块化测试 + 能力探测 + 路由过滤

本次更新聚焦于**按需选测模块**、**准确识别能力**和**路由仅选运行中模型**，主要变更包括：

1. **模块化测试（按需选测）**：`run.py test` 新增 `--modules` 参数，可选 `text` / `vision_recognition` / `image_generation` / `efficiency`，逗号分隔。测完一个模块后只写入该模块结果，后续再测其他模块时**增量合并**到 `model_benchmarks.json`，互不覆盖。
2. **增量结果合并**：重写 `benchmarks_json.update_benchmarks_json`，按子键合并 `benchmarks` / `multimodal` / `capability_status`，仅替换 `efficiency` 等整体字段，保留未测试模块的已有结果。
3. **能力探测（避免误判）**：新增 `probe_vision_capability` / `probe_t2i_capability`，测试多模态模块前发送最小测试请求探测模型是否真正支持。仅在明确 404/405 或 400 含 "not support" 等信号时标记为 `unsupported`；超时/5xx 返回 `None`（不确定），避免把实际支持的模型误判为不支持。
4. **capability_status 字段**：`model_benchmarks.json` 新增 `capability_status`，取值 `supported` / `unsupported` / `not_tested`，明确区分"未测试"和"已探测不支持"。
5. **路由仅选运行中模型**：`select_expert_by_domain` / `select_multimodal_model` 新增 `running_models` 参数，仅从 `router/config.py` 的 `model_routes` 已注册模型中选择；多模态任务额外跳过 `capability_status` 为 `unsupported` 的模型。
6. **命令行参数扩展**：`run.py test` 新增 `--model` / `--api-key` / `--base-url` / `--modules` / `--num-samples` / `--skip-probe`，支持灵活配置。
7. **文档更新**：README.md / README_EN.md 新增「模块化测试」章节和端到端使用示例；项目说明文档.md 同步更新。

---

## [Unreleased] - 2026-06-22

### 项目总体变更概述

本次更新聚焦于**路由参数管理机制优化**和**默认参数调优**，主要变更包括：

1. **移除参数持久化机制**：原方案将修改后的参数写入 `params_runtime.json`，重启后自动加载。改为**仅内存生效 + 日志记录**模式，重启后恢复默认预设，更符合"调参实验"的使用场景。
2. **新增变更日志**：每次保存参数修改时，自动追加一条记录到 `router/params_changes.log`，包含时间戳、操作类型（set/set_many/apply_preset/reset）和参数值快照，便于审计和回溯。
3. **默认参数优化**：将默认预设从 `efficiency_first` 改为 `balanced`，并调整多个参数的默认值，使开箱即用的效果更均衡。
4. **文档完善**：更新 README.md、项目说明文档.md、README_EN.md，新增调参界面 PNG 截图，新增 CHANGELOG.md。

---

### 文生图（text-to-image）测试与路由（新增功能）

本次更新新增**文生图测试与路由**能力，扩展多模态测试从识图到生图，主要变更包括：

1. **新增 1 类文生图题库**（共 50 题）：T2I，位于 `benchmarks/multimodal/t2i-文生图(50).txt`，覆盖动物/风景/物体/抽象/场景 5 类。
2. **扩展自动评测系统**：新增 `ask_model_t2i` 调用 OpenAI 兼容 `/v1/images/generations` 接口生成图片；新增 `judge_t2i_answer` 由打分模型对生成图片做 0-10 分评估（评分维度：主体准确度、场景契合度、画面质量）。
3. **扩展路由系统**：新增 `is_image_generation_message` 识别生图任务（关键词优先于识图判定），`select_multimodal_model` 新增 `task_kind` 参数区分识图（`image_recognition`）和生图（`image_generation`），生图任务仅路由到具备 `multimodal.image_generation` 的模型。
4. **新增配置项**：`ENABLE_T2I_TEST`（autotest）、`T2I_BENCHMARK_FILES`、`T2I_IMAGE_SIZE`、`t2i_benchmark_files`（router）。
5. **文档完善**：更新项目说明文档.md，新增文生图题库、配置、路由流程、请求示例说明。

---

### 多模态能力类型分组重构（结构优化）

本次更新将 `multimodal` 子结构从扁平格式重构为**按能力类型分组**，明确区分视觉识图、视觉生图等不同能力，主要变更包括：

1. **数据结构重构**：`model_benchmarks.json` 的 `multimodal` 子结构从扁平格式改为按能力类型分组：
   ```json
   "multimodal": {
     "vision_recognition": { "chart_qa": 0.8, "text_vqa": 0.9, ... },
     "image_generation":   { "t2i": 0.72 }
   }
   ```
2. **能力类型常量**：在 `autotest/benchmarks_json.py` 和 `router/whoengine.py` 中定义 `CAPABILITY_TYPES`、`CAPABILITY_VISION_RECOGNITION`、`CAPABILITY_IMAGE_GENERATION` 常量，未来可扩展 `audio_recognition`/`audio_generation`/`video_recognition` 等。
3. **测试写入分组**：`_build_router_vision_recognition` 写入 `multimodal.vision_recognition`，`_build_router_image_generation` 写入 `multimodal.image_generation`。
4. **路由按能力查询**：`select_multimodal_model` 按 `task_kind` 查询对应能力类型分组，新增 `_get_capability_scores` 辅助函数。
5. **向后兼容**：路由查询时自动识别旧扁平格式并按 domain 归类到对应能力类型，无需迁移即可兼容旧数据。
6. **数据迁移**：现有 `model_benchmarks.json` 中两个模型的扁平 `multimodal` 已迁移为分组结构。

---

### 多模态视觉测试与路由（新增功能）

本次更新新增**视觉多模态测试与路由**能力，主要变更包括：

1. **新增 5 类视觉多模态题库**（共 120 题）：ChartQA、TextVQA、MathVista、VQA、MMMU，位于 `benchmarks/multimodal/`。
2. **新增图片生成脚本** `benchmarks/multimodal/generate_images.py`，使用 Pillow 自动生成所有题目配图。
3. **扩展自动评测系统**：支持多模态模型测试（OpenAI Vision 兼容接口），结果写入 `model_benchmarks.json` 的 `multimodal` 子结构。
4. **扩展路由系统**：自动识别多模态任务（含图片/视觉关键词），仅对具备 `multimodal` 子结构的模型进行选型；非多模态任务路由行为不变。
5. **新增配置项**：`ENABLE_MULTIMODAL_TEST`（autotest）、`ENABLE_MULTIMODAL_ROUTING`（router）、`MULTIMODAL_BENCHMARK_FILES`。
6. **文档完善**：更新 README.md、项目说明文档.md，新增多模态使用指南、路由流程、FAQ。

#### 涉及文件

**新增**：
- `benchmarks/multimodal/chartqa-图表理解(20).txt`
- `benchmarks/multimodal/textvqa-文字识别(20).txt`
- `benchmarks/multimodal/mathvista-视觉数学(20).txt`
- `benchmarks/multimodal/vqa-视觉问答(30).txt`
- `benchmarks/multimodal/mmmu-多模态理解(30).txt`
- `benchmarks/multimodal/generate_images.py`

**修改**：
- `autotest/parser.py`：新增 `parse_multimodal` 解析器
- `autotest/model_api.py`：新增 `ask_model_multimodal`、`build_visual_mc_prompt`、`_encode_image`
- `autotest/main.py`：新增 `run_multimodal_benchmark` 函数
- `autotest/benchmarks_json.py`：新增 `multimodal` 子结构输出
- `autotest/config.py`：新增 `ENABLE_MULTIMODAL_TEST`、`MULTIMODAL_BENCHMARK_FILES`
- `router/whoengine.py`：新增 `is_multimodal_message`、`select_multimodal_model`、`classify_and_select_multimodal`
- `router/router_engine.py`：新增 `route_messages` 函数
- `router/app.py`：支持多模态消息透传
- `router/config.py`：新增 `ENABLE_MULTIMODAL_ROUTING`、`MULTIMODAL_BENCHMARK_FILES`
- `router/routing_params.py`：新增 `domain_alpha.multimodal`、`difficulty.base.multimodal`

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
