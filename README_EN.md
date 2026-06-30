<p align="center">
  <h1 align="center">test_auto_route</h1>
  <p align="center"><strong>Automated LLM Evaluation + Intelligent Routing Program</strong></p>
  <p align="center">
    <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python version"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg" alt="FastAPI"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  </p>
</p>

---

## Introduction

**test_auto_route** is an automated LLM evaluation and intelligent routing program, solving two core problems:

- **Automated Evaluation**: Objectively and comprehensively evaluate LLM capabilities across diverse benchmark suites
- **Intelligent Routing**: Automatically select the best model based on user input when multiple models are available

These two functions are connected through `model_benchmarks.json`: the evaluation system produces model capability data, and the routing system consumes this data for intelligent model selection.

**Key Features:**
- **KNN + Keyword Prior Routing** — High domain classification accuracy, corrects embedding confusion
- **8 Text + 5 Visual Recognition + 1 Image Generation Benchmark Suites** — From multiple-choice and math reasoning to chart understanding, visual QA, and text-to-image
- **Capability-Based Multimodal Routing** — Distinguishes vision recognition from image generation, routes only to models with the corresponding capability type
- **6 Routing Strategies** — Average, Majority Voting, KNN, KNN+Prior, Ensemble, etc.
- **Efficiency-First Selection** — 80% of low-difficulty tasks auto-select efficient models; only hard tasks use expert models
- **Remote Endpoint Auto-Discovery** — Provide URL + Key to auto-detect running models; no need to manually fill in model names
- **Multi-User Concurrent Safe** — Routing inference and non-streaming calls are fully async, supporting concurrent access without response mixing
- **Visual Tuning Dashboard** — Web UI with sliders to adjust routing weights in real-time, no restart needed
- **OpenAI-Compatible API** — Standard `/v1/chat/completions`, works with any OpenAI client out of the box
- **`@model-name` Override** — Use `@model-name` at the start of a message to bypass routing and specify a model directly
- **Force-Route Checkbox** — Check a model in the Web UI to force-route all requests; priority below `@model-name`
- **GPU Acceleration** — Auto-detects GPU, embedding inference ~10ms

---

## Table of Contents

- [Quick Start](#quick-start)
- [Module-specific Testing](#module-specific-testing)
- [Web UI Guide (For End Users)](#web-ui-guide-for-end-users)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Routing Algorithm](#routing-algorithm)
- [Model Selection Formula](#model-selection-formula)
- [Visual Tuning Dashboard](#visual-tuning-dashboard)
- [Supported Benchmarks](#supported-benchmarks)
- [Multimodal Routing Principles](#multimodal-routing-principles)
- [API Endpoints](#api-endpoints)
- [Configuration Guide](#configuration-guide)
- [FAQ](#faq)
- [License](#license)

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8+ recommended for GPU acceleration)
- 8GB+ VRAM (optional, for GPU-accelerated embedding inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/test_auto_route.git
cd test_auto_route

# Create a virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

**Step 1: Configure Evaluation API**

Edit `autotest/config.py`:

```python
TEST_API_KEY = "sk-your-api-key-here"         # API key for the model under test
TEST_BASE_URL = "https://api.example.com/v1/chat/completions"
TEST_MODEL_NAME = "your-model-name"
```

**Step 2: Configure Routing Model Server**

Edit `router/config.py`. The recommended approach is **remote endpoint auto-discovery** (just provide URL + Key, and the program automatically detects running models):

```python
REMOTE_SERVER_CONFIG = {
    # Option 1 (Recommended): Remote endpoint auto-discovery — provide URL + Key,
    # the program calls GET {url}/models to auto-detect running models
    "remote_endpoints": [
        {"url": "https://your-server:8443/v1", "api_key": "sk-xxxx"},
    ],

    # Option 2 (Advanced, backward-compatible): Manual model route mapping —
    # fill in model name → API URL one by one
    # "model_routes": {
    #     "model-a": "https://your-server:8443/v1/chat/completions",
    # },
}
```

> You can also skip editing the config file and configure directly in the Web UI (`/dashboard`): switch to Normal Mode, fill in URL + Key, and click "🔍 Save & Detect Models".

**Step 3: Run Automated Evaluation**

```bash
python run.py test
```

After completion, results are saved in the `results/` directory, and capability data is aggregated into `model_benchmarks.json`.

> **Module-specific testing**: Use `--modules` to test only certain modules. Results are **merged incrementally** into `model_benchmarks.json` (existing results from other modules are preserved). See [Module-specific Testing](#module-specific-testing).

**Step 4: Start the Routing Service**

```bash
python run.py route
```

The service runs at `http://0.0.0.0:8000` by default. On startup, the browser automatically opens the **home page** (`http://localhost:8000/home`) where you can choose between "Model Evaluation" or "Route Tuning".

---

## Module-specific Testing

The evaluation system supports **module-specific testing**: after a module is tested, only that module's results are written. Subsequent tests of other modules are **merged incrementally** into `model_benchmarks.json` without overwriting existing results. During routing, only models that are **currently running and have the corresponding module's test results** are selected.

### Available Modules

| Module | Description | Written to |
|--------|-------------|-----------|
| `text` | Text benchmarks (8 suites) | `benchmarks` substructure |
| `vision_recognition` | Vision recognition (understanding existing images) | `multimodal.vision_recognition` |
| `image_generation` | Image generation (text-to-image) | `multimodal.image_generation` |
| `efficiency` | Efficiency (TTFT/throughput/concurrency) | `efficiency` substructure |

### End-to-end Example

```bash
# 1) Test text module first, results written to benchmarks substructure
python run.py test --model qwen36-35b-a3b --modules text

# 2) Test vision recognition module, merged into multimodal.vision_recognition
python run.py test --model qwen36-35b-a3b --modules vision_recognition

# 3) Test image generation module, merged into multimodal.image_generation
python run.py test --model qwen36-35b-a3b --modules image_generation

# 4) Start routing: automatically selects only "running + capable" models
python run.py route
```

All test results are aggregated into **`model_benchmarks.json`** in the project root, which the routing service reads on startup.

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| `--model NAME` | Model name to test (overrides `TEST_MODEL_NAME` in `autotest/config.py`) |
| `--modules a,b,c` | Modules to test, comma-separated (see table above) |
| `--num-samples N` | Max questions per benchmark suite (sampling for speed) |
| `--api-key KEY` | API key (overrides `autotest/config.py`) |
| `--base-url URL` | API endpoint (overrides `autotest/config.py`) |
| `--skip-probe` | Skip capability probing (faster if you know the model supports it) |

### Capability Probing (avoid false negatives)

Before testing multimodal modules, the program **automatically probes** whether the model truly supports the capability (by sending a minimal test request):

- **Supported** → marks `capability_status: "supported"` and proceeds with testing
- **Explicitly unsupported** (404/405, or 400 with "not support" etc.) → marks `capability_status: "unsupported"` and skips that module
- **Uncertain** (timeout/5xx) → does NOT mark as unsupported, to avoid false negatives where a supported model is incorrectly flagged

`capability_status` is written to `model_benchmarks.json`. During routing, models marked as `unsupported` for a capability are **skipped** for that task type.

### Enabling Multimodal Testing

Before testing vision recognition / image generation modules, enable the corresponding switches in `autotest/config.py`:

1. Vision recognition: set `ENABLE_MULTIMODAL_TEST = True`
2. Image generation: set `ENABLE_T2I_TEST = True`
3. Ensure the model supports the corresponding interface (OpenAI Vision-compatible interface / `/v1/images/generations` endpoint)
4. Run `python run.py test --modules vision_recognition,image_generation`. Capability is auto-probed before testing; results are written by capability type into the `multimodal` substructure and `capability_status` of `model_benchmarks.json`

> You can also check "Vision Recognition" / "Image Generation" modules directly in the Web UI Model Evaluation page — same effect as command line.

---

## Web UI Guide (For End Users)

The system provides a complete web interface. **No command line required** for model evaluation or route tuning. The steps below follow the actual usage order.

### 1. Start the Service and Open Home Page

Run in the project root directory:

```bash
python run.py route
```

On startup, the browser **automatically opens the home page** (`http://localhost:8000/home`). If it doesn't open automatically, manually enter the address in the browser.

### 2. Home Page (/home)

The home page is the entry point of the entire Web UI, with two large buttons:

| Button | Page to enter | Purpose |
|--------|---------------|---------|
| **Model Evaluation** | `/test` | Test model capabilities (multiple choice, math, vision recognition, image generation, etc.) |
| **Route Tuning** | `/dashboard` | Visually tune routing weights via sliders, takes effect immediately |

Click the corresponding button to enter the respective page. Navigation links at the top allow switching between all three pages at any time.

### 3. Model Evaluation UI (/test)

Complete all test configuration in the browser without editing any code or config files.

**Steps:**

1. **Fill in API Info**
   - Model name (e.g., `qwen36-35b-a3b`)
   - API key (e.g., `sk-xxxx`)
   - API URL (e.g., `https://api.example.com/v1/chat/completions`)
   - The above three fields are auto-prefilled from `autotest/config.py` — can be used directly without manual input

2. **Select Test Modules** (checkboxes)
   - **Text**: Pure text capabilities (8 question banks including multiple choice, math, long-context comprehension)
   - **Vision Recognition**: Understanding existing images (5 banks: chart, text, math, VQA, MMMU)
   - **Image Generation**: Text-to-image capability (generating images from text descriptions)
   - **Efficiency**: Response speed testing (TTFT, throughput, concurrency limit)

3. **Set Test Parameters**
   - **Global sample count**: Sets a unified sample count for all banks (leave empty for default per-bank values; a number overrides all banks unless per-bank values are set below)
   - **Per-bank sample settings (optional)**: Expand the "⚙ Per-bank sample settings" panel to set samples individually for each bank. Empty fields use the bank default or global value. Priority: per-bank > global > default
   - **Capability probe**: When checked, sends a minimal request before multimodal modules to detect if the model supports it

4. **Start Test**
   - Click "Start Test" button, real-time logs appear below
   - "Stop" button available during the test to terminate

5. **View Results**
   - On completion, results auto-merge into `model_benchmarks.json`
   - Testing different modules multiple times won't overwrite each other — results are merged

> **Tip**: To test only one module without affecting other modules' existing results, just check that module — results will merge incrementally.

### 6. Export Test Documents (XLSX)

After testing, click the "📤 Export Test Documents" button to export results as Excel files:

- **Download ZIP**: Browser download dialog opens, user chooses save location (recommended for general users)
- **Export to specific directory**: Enter a server path, xlsx files are copied there (for server-side operations)

**Edge case handling**:

| Case | System prompt |
|------|--------------|
| No test run yet | "No test has been run, please run a test first" |
| Results directory missing | "Results directory for model X not found, please complete a test first" |
| Same-name file exists in save dir | Asks "Overwrite?", overwrites on confirm |
| File occupied by Excel | "File is occupied, please close it and retry" |
| Save directory doesn't exist | Auto-created |

> For more detailed UI instructions, see [docs/Web_UI操作手册.md](./docs/Web_UI操作手册.md).

### 4. Route Dashboard (/dashboard)

On this page, tune routing weights in real-time by dragging sliders, no service restart needed. See [Visualization Dashboard](#visualization-dashboard) below for details.

### 5. Test Routing Only (No API Cost)

To verify routing selection accuracy without actually calling models (no API cost), use the command-line route analysis endpoint:

```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt":"3x+5=20, solve for x"}'
```

The response shows which model the router selected and why, but does not actually call the remote model.

---

## Workflow

![Workflow](./workflow.svg)

---

## Project Structure

```
test_auto_route/
├── run.py                          # [Entry] Unified entry script
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md                       # Project README (Chinese)
├── README_EN.md                    # Project README (English)
├── 项目说明文档.md                  # Detailed technical docs (Chinese)
├── model_benchmarks.json           # Model evaluation data (routing basis)
│
├── autotest/                       # ===== Automated Evaluation =====
│   ├── config.py                   # Evaluation config (API keys, paths)
│   ├── main.py                     # Evaluation controller
│   ├── model_api.py                # Model API calls + scoring logic
│   ├── parser.py                   # Benchmark parsers (8 formats)
│   ├── utils.py                    # Result export (XLSX)
│   ├── benchmark_efficiency.py     # Efficiency tests (TTFT/throughput/concurrency)
│   └── benchmarks_json.py          # JSON aggregation
│
├── router/                         # ===== Intelligent Routing =====
│   ├── config.py                   # Routing config
│   ├── app.py                      # FastAPI service entry
│   ├── whoengine.py                # [Core] WhoEngine router
│   ├── router_engine.py            # Routing engine
│   ├── model_client.py             # Remote model client
│   ├── task_classifier.py          # Task classifier (backup)
│   ├── scoring.py                  # Model scoring & selection
│   ├── model_profiles.py           # Model capability profiles
│   └── static/
│       ├── home.html                # Home page (choose test/route)
│       ├── test.html                # Model evaluation UI
│       └── dashboard.html           # Visual tuning dashboard frontend
│
├── benchmarks/                     # ===== Benchmark Suites =====
│   ├── mmlu_gsm8k_hellaswag/       # Basic benchmarks
│   ├── bbh_longbench/              # Advanced benchmarks
│   ├── training_extra/             # Router training augmentation samples
│   ├── workplace/                  # Workplace subjective questions
│   └── multimodal/                 # Visual multimodal benchmarks (5 types, 120 questions)
│       ├── chartqa-图表理解(20).txt
│       ├── textvqa-文字识别(20).txt
│       ├── mathvista-视觉数学(20).txt
│       ├── vqa-视觉问答(30).txt
│       ├── mmmu-多模态理解(30).txt
│       ├── generate_images.py     # Image generation script
│       └── images/                # Question images (PNG)
│
├── results/                        # Test results (auto-generated, gitignored)
└── models/                         # Embedding model cache (auto-generated, gitignored)
```

---

## Routing Algorithm

### Algorithm Evolution

| Algorithm | Accuracy | Characteristics |
|-----------|----------|-----------------|
| Ridge Regression (baseline) | 71.0% | Linear decision boundary |
| Token-level Voting | 67.7% | Per-token classification + voting |
| KNN (k=20) | 83.9% | Non-parametric, fits non-linear boundaries |
| **KNN + Keyword Prior (recommended)** | **High** | **KNN + keyword bias, corrects embedding confusion** |

### Core Principle

**KNN + Keyword Prior Hybrid Routing** is the core innovation of this project:

1. **KNN Soft Voting**: Encode the query into a multi-pooled sentence vector (mean+max+cls), compute cosine similarity with all training samples, take top-k neighbors for softmax soft voting
2. **Keyword Prior Bias**: Maintain a strong-signal keyword table for each domain; generate a prior probability distribution when keywords are detected
3. **Hybrid Decision**: Final probability = α × KNN probability + (1-α) × keyword prior (α=0.7 by default, biased toward KNN)

> Why keyword priors? Error analysis revealed that the embedding model misclassifies "chemical formula" as a math question (because the training data contains many English math problems). Keyword priors provide strong signals to correct such systematic confusion without breaking the pure ML capability.

For detailed experimental data, see [项目说明文档](./项目说明文档.md).

---

## Supported Benchmarks

### Text-only Benchmarks (8 types)

| Benchmark | Type | Count | Scoring Method |
|-----------|------|-------|----------------|
| MMLU | Multiple choice | 30 | Answer comparison |
| GSM8K | Math fill-in | 10 | Answer comparison |
| HellaSwag | Multiple choice | 20 | Answer comparison |
| BBH Semantic | Multiple choice | 10 | Answer comparison |
| BBH Math | Math computation | 10 | Answer comparison |
| LongBench | Long-context understanding | 10 | LLM scoring (0-10) |
| Workplace - PM | Subjective | 20 | LLM scoring (0-10) |
| Workplace - Secretary | Subjective | 20 | LLM scoring (0-10) |

### Visual Multimodal Benchmarks (5 types, 120 questions total)

| Benchmark | Type | Count | Scoring Method | Source |
|-----------|------|-------|----------------|--------|
| ChartQA | Chart understanding | 20 | Answer comparison | Industry-standard chart QA |
| TextVQA | Text-in-image recognition | 20 | Answer comparison | Text-in-image recognition |
| MathVista | Visual math reasoning | 20 | Answer comparison | Visual math reasoning |
| VQA | General visual QA | 30 | Answer comparison | General visual QA |
| MMMU | Multimodal understanding | 30 | Answer comparison | Multi-discipline multimodal |

> Multimodal benchmarks use OpenAI Vision-compatible interface (`image_url` + base64). Enabled only when `ENABLE_MULTIMODAL_TEST=True`. Images are auto-generated by `benchmarks/multimodal/generate_images.py`.

---

## Multimodal Routing Principles

The system supports automatic detection and routing of visual multimodal tasks, **distinguishing vision recognition from image generation** by capability type.

### Capability Type Grouping

The `multimodal` substructure in `model_benchmarks.json` is grouped by capability type:

| Capability Type | Description | Domains |
|----------------|-------------|---------|
| `vision_recognition` | Vision recognition (understanding existing images) | chart_qa / text_vqa / math_vista / vqa / mmmu |
| `image_generation` | Image generation (creating new images) | t2i |

> Extensible to `audio_recognition`, `audio_generation`, etc. in the future.

### How It Works

1. **Task Detection**: Router determines task type by priority
   - **Image generation** (highest priority): text hits generation keywords (draw a / generate image / t2i, etc.)
   - **Vision recognition**: contains image_url or visual keywords (image/chart/screenshot/OCR, etc.)
   - **Text-only**: none of the above
2. **Branch Routing**:
   - **Text-only**: Uses original KNN routing, only references `benchmarks` substructure
   - **Vision recognition**: Only selects from models with `multimodal.vision_recognition`
   - **Image generation**: Only selects from models with `multimodal.image_generation`
3. **Message Pass-through**: Multimodal messages (with `image_url`) are passed through to remote models as-is

### Detection Rules

- **Image generation task**: text hits generation keywords (draw a / generate image / t2i / etc.), highest priority
- **Vision recognition task**: any of the following
  - messages contain `content` as a list with `type=image_url` item
  - text contains visual keywords (image/chart/screenshot/OCR/visual, etc.)
  - text contains base64 image data URL (`data:image/`)
- **Text-only task**: none of the above

> For enabling multimodal testing, see [Module-specific Testing - Enabling Multimodal Testing](#enabling-multimodal-testing) above. For multimodal request examples, see [API Endpoints](#api-endpoints) below.

---

## API Endpoints

The routing service provides the following HTTP endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/home` | GET | **Home page** (choose test/route) |
| `/test` | GET | **Model evaluation UI** (Web UI) |
| `/v1/chat/completions` | POST | Chat completion (core endpoint, OpenAI-compatible) |
| `/v1/route` | POST | Query routing result only (no remote model call) |
| `/v1/models` | GET | List available models |
| `/v1/models/@mention` | GET | Frontend @mention model list |
| `/dashboard` | GET | **Visual tuning dashboard** (Web UI) |
| `/api/test/config` | GET | Get test default config (for UI prefill) |
| `/api/test/run` | POST | Start test subprocess (supports `num_samples_map` per-bank sampling) |
| `/api/test/status` | GET | Get test status and incremental logs |
| `/api/test/stop` | POST | Stop test subprocess |
| `/api/test/export` | POST | Export test results as XLSX (browser download or copy to path) |
| `/api/test/benchmarks` | GET | Get per-bank metadata (for per-bank sampling UI) |
| `/api/route/config` | GET / POST | Get / update routing service config (test_mode, remote_endpoints, model_routes, verify_ssl, request_timeout), takes effect immediately |
| `/api/route/discover` | POST | Trigger remote model auto-discovery (calls `{url}/models` on each endpoint) |
| `/api/route/status` | GET | Get current routing service status (mode, registered models, discovered model count) |
| `/api/route/models` | GET | Get model list and capability types (test mode reads from test results, normal mode detects from remote `/v1/models`) |
| `/api/route/forced-model` | GET / POST | Get / set forced routing model (checkbox: all requests route to this model, priority below `@model-name`) |
| `/api/params` | GET / POST | Get / update routing parameters |
| `/api/params/meta` | GET | Get parameter metadata (ranges, descriptions) |
| `/api/params/preset` | GET / POST | Get / apply preset |
| `/api/params/reset` | POST | Reset to default preset |
| `/api/docs/markdown` | GET | Get project docs (Markdown) |
| `/api/docs/readme` | GET | Get README (Markdown) |
| `/api/docs/docx/download` | GET | Download project docs (DOCX) |
| `/health` | GET | Health check |

### Examples

```bash
# Routing analysis (no remote model call)
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Solve 3x+5=20 for x"}'

# Chat completion (auto-routing + remote call)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Solve 3x+5=20 for x"}]}'
```

**Specify a model (bypass routing):** Use `@model-name` at the start of the message
```
@model-name Write a quicksort algorithm
```

**Multimodal request example (vision recognition):** Pass image_url in messages content; the router auto-detects it as a multimodal task:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the content of this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
      ]
    }]
  }'
```

> For multimodal task detection rules, see [Multimodal Routing Principles](#multimodal-routing-principles) above.

---

## Configuration Guide

### Key Configuration Files

| File | Purpose |
|------|---------|
| `autotest/config.py` | API keys, model names, benchmark paths |
| `router/config.py` | Routing strategy, remote model server URLs, WhoEngine config |
| `router/routing_params.py` | **Centralized parameter management** (34 params + 3 presets + change log) |
| `router/static/dashboard.html` | **Visual tuning dashboard frontend** |

### WhoEngine Configuration Example

```python
WHOENGINE_CONFIG = {
    "embedder": "BAAI/bge-large-zh-v1.5",  # Embedding model (1024-dim, matches whoengine.pt checkpoint)
    "routing_strategy": "knn_prior",       # Recommended strategy
    "knn_k": 20,                           # KNN neighbors
    "knn_prior_alpha": 0.7,                # Prior mixing coefficient (KNN-biased)
    "knn_sim_temp": 10.0,                  # Similarity temperature
}
```

> ⚠️ The `embedder` must match the model used when training `whoengine.pt` (default: `bge-large-zh-v1.5`, 1024-dim → 3072-dim multi-pooled features). Using `bge-small-zh-v1.5` (512-dim) will cause a dimension mismatch error.

### Routing Model Server Configuration

The routing service supports two ways to connect to remote models:

| Method | Description | Recommended |
|--------|-------------|-------------|
| `remote_endpoints` | Provide URL + API Key; the program calls `GET {url}/models` to auto-detect running models | ⭐ Recommended |
| `model_routes` | Manually fill in model name → API URL mappings one by one | Advanced / backward-compatible |

Both methods can be used simultaneously. `get_model_url()` lookup order: `model_routes` first → then `remote_endpoints` discovery cache.

Configuration can be modified in `router/config.py` or via the Web UI (`/dashboard` → Normal Mode → Remote Endpoints), taking effect immediately.

---

## FAQ

**Q: What if routing accuracy is low?**
A: Ensure `routing_strategy` is set to `knn_prior`, delete `whoengine.pt`, and restart the service to retrain.

**Q: How do I enable multimodal visual testing?**
A: Set `ENABLE_MULTIMODAL_TEST = True` (recognition) / `ENABLE_T2I_TEST = True` (generation) in `autotest/config.py`. Ensure the model supports the corresponding interface (Vision / images/generations), then run `python run.py test`. Results are written to `multimodal.vision_recognition` and `multimodal.image_generation` groups grouped by capability type.

**Q: How are multimodal tasks routed?**
A: The router determines task type by priority: image generation (generation keywords) → vision recognition (image/visual keywords) → text-only. Image generation tasks only select from models with `multimodal.image_generation`; vision recognition tasks only select from models with `multimodal.vision_recognition`; text-only tasks use the original KNN routing, unaffected by multimodal results.

**Q: How do I add a new Domain?**
1. Prepare training question files and place them in `benchmarks/`
2. Add the path in `router/config.py` under `benchmark_files`
3. Add keywords for the new domain in `whoengine.py` under `DOMAIN_KEYWORDS_PRIOR`
4. Delete `whoengine.pt` and restart the service

**Q: Which embedding models are supported?**
A: All sentence-transformers models are supported; `BAAI/bge-large-zh-v1.5` is recommended. The first run automatically caches to `models/sentence_transformers/`, no internet required afterward.

**Q: Is GPU acceleration supported?**
A: Yes, WhoEngine auto-detects GPU; embedding model and KNN computations all run on GPU, with inference latency ~10ms. To install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## License

[MIT](LICENSE)
