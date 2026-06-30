"""
============================================================================
模型自动测试 - 主控脚本
============================================================================
流程:
  1. 读取 config.py 中的配置
  2. 解析各题集文件（不含答案的题目 + 正确答案）
  3. 逐题调用待测模型 API，获取模型答案
  4. 调用评分逻辑，比对/打分
  5. 导出 xlsx 结果
  6. 更新 model_benchmarks.json
"""
import os
import sys
import time
import argparse

# 将当前目录加入 sys.path，确保可以 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from parser import parse_benchmark
from model_api import (
    ask_model,
    ask_model_multimodal,
    ask_model_t2i,
    score_answer,
    _warm_up,
    probe_vision_capability,
    probe_t2i_capability,
    PROMPT_BUILDER_MAP,
    EXTRACTOR_MAP,
)
from utils import export_all, export_efficiency
from benchmarks_json import (
    build_benchmark_summary,
    update_benchmarks_json,
    CAPABILITY_STATUS_SUPPORTED,
    CAPABILITY_STATUS_UNSUPPORTED,
    CAPABILITY_STATUS_NOT_TESTED,
    CAPABILITY_TYPES,
)


# =========================
# 可选测试模块定义
# =========================
# 用户可通过 --modules 参数选择测试哪些模块
# 模块名 → 中文说明
TEST_MODULES = {
    "text":               "通用语言能力（纯文本题集）",
    "vision_recognition": "视觉识图（理解已有图片）",
    "image_generation":   "视觉生图（生成新图片）",
    "efficiency":         "效率测试（吞吐/延迟/并发）",
}

# 默认测试全部模块
DEFAULT_MODULES = ["text", "vision_recognition", "image_generation", "efficiency"]


def _get_filepath(bench_key, fname):
    """获取题集完整路径。如果是绝对路径则直接返回，否则拼接项目根目录。"""
    if os.path.isabs(fname):
        return fname
    return os.path.join(cfg.PROJECT_ROOT, fname)


def run_benchmark(benchmark_name, benchmark_display, filepath,
                  test_api_key, test_base_url, test_model,
                  judge_api_key, judge_base_url, judge_model,
                  num_samples=None, request_interval=1.0, timeout=600):
    """
    对一个题集执行完整评测流程，返回结果字典。
    """
    print(f"\n{'='*60}")
    print(f"  题集: {benchmark_name}")
    print(f"  文件: {filepath}")
    print(f"{'='*60}")

    # 1. 解析
    questions = parse_benchmark(benchmark_name, filepath)
    if not questions:
        print("  [警告] 未解析到任何题目，跳过")
        return None
    print(f"  解析完成: 共 {len(questions)} 题, 类型: {questions[0]['type']}")

    # 采样
    if num_samples and num_samples < len(questions):
        questions = questions[:num_samples]
        print(f"  采样: 取前 {num_samples} 题")

    # 2. 逐题评测
    results = []
    is_subjective = questions[0]["type"] in ("subjective", "long_context")

    for i, q in enumerate(questions):
        qid = q["id"]
        qtype = q["type"]
        print(f"\n  [{i+1}/{len(questions)}] Q{qid} ({qtype})", end=" ", flush=True)

        # 构建提示词
        prompt_builder = PROMPT_BUILDER_MAP.get(qtype)
        if not prompt_builder:
            print(f"[错误] 不支持的题目类型: {qtype}")
            continue
        prompt = prompt_builder(q)

        # 调用待测模型
        raw_output = ask_model(prompt, test_api_key, test_base_url, test_model, timeout)
        display_output = raw_output[:120] + "..." if len(raw_output) > 120 else raw_output
        print(f"-> 模型输出: {display_output}")

        # 提取答案
        extractor = EXTRACTOR_MAP.get(qtype)
        if extractor:
            extracted = extractor(raw_output)
        else:
            extracted = raw_output

        # 评分
        score, comment = score_answer(q, raw_output, judge_api_key, judge_base_url, judge_model, timeout)

        result = {
            "qid": qid,
            "model_output": raw_output,
            "extracted_answer": extracted,
            "correct_answer": q.get("correct_answer", ""),
            "score": score,
            "comment": comment,
        }
        results.append(result)

        if is_subjective:
            print(f"    得分: {score}/10 | 评语: {comment}")
        else:
            print(f"    提取答案: {extracted} | 正确答案: {q.get('correct_answer', '')} | 得分: {score}")

        # 请求间隔，避免限流
        if i < len(questions) - 1:
            time.sleep(request_interval)

    # 3. 统计
    total = len(results)
    if is_subjective:
        total_score = sum(r["score"] for r in results)
        avg_score = total_score / total if total > 0 else 0
        acc = avg_score / 10 if total > 0 else 0  # 统一为正确率（0~1）
        print(f"\n  --- {benchmark_name} 汇总 ---")
        print(f"  总题数: {total}  总分: {total_score}  均分: {avg_score:.2f}/10  正确率: {acc:.2%}")
    else:
        correct = sum(1 for r in results if r["score"] == 1)
        acc = correct / total if total > 0 else 0
        print(f"\n  --- {benchmark_name} 汇总 ---")
        print(f"  总题数: {total}  正确: {correct}  准确率: {acc:.2%}")

    return {
        "benchmark_name": benchmark_name,
        "benchmark_display": benchmark_display,
        "questions": questions,
        "results": results,
    }


def run_multimodal_benchmark(benchmark_name, benchmark_display, filepath,
                             test_api_key, test_base_url, test_model,
                             judge_api_key, judge_base_url, judge_model,
                             num_samples=None, request_interval=1.0, timeout=600):
    """
    对一个多模态题集执行完整评测流程，返回结果字典。
    与 run_benchmark 类似，但调用 ask_model_multimodal 发送图片+文本。
    """
    print(f"\n{'='*60}")
    print(f"  [多模态] 题集: {benchmark_name}")
    print(f"  文件: {filepath}")
    print(f"{'='*60}")

    questions = parse_benchmark(benchmark_name, filepath)
    if not questions:
        print("  [警告] 未解析到任何题目，跳过")
        return None
    print(f"  解析完成: 共 {len(questions)} 题, 类型: {questions[0]['type']}")

    if num_samples and num_samples < len(questions):
        questions = questions[:num_samples]
        print(f"  采样: 取前 {num_samples} 题")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        qtype = q["type"]
        image_path = q.get("image_path", "")
        print(f"\n  [{i+1}/{len(questions)}] Q{qid} ({qtype})", end=" ", flush=True)

        if not image_path or not os.path.exists(image_path):
            print(f"[跳过] 图片不存在: {image_path}")
            results.append({
                "qid": qid, "model_output": "", "extracted_answer": "",
                "correct_answer": q.get("correct_answer", ""),
                "score": 0, "comment": "图片文件缺失",
            })
            continue

        prompt_builder = PROMPT_BUILDER_MAP.get(qtype)
        if not prompt_builder:
            print(f"[错误] 不支持的题目类型: {qtype}")
            continue
        prompt = prompt_builder(q)

        raw_output = ask_model_multimodal(
            prompt, image_path, test_api_key, test_base_url, test_model, timeout
        )
        display_output = raw_output[:120] + "..." if len(raw_output) > 120 else raw_output
        print(f"-> 模型输出: {display_output}")

        extractor = EXTRACTOR_MAP.get(qtype)
        extracted = extractor(raw_output) if extractor else raw_output

        score, comment = score_answer(q, raw_output, judge_api_key, judge_base_url, judge_model, timeout)

        result = {
            "qid": qid,
            "model_output": raw_output,
            "extracted_answer": extracted,
            "correct_answer": q.get("correct_answer", ""),
            "score": score,
            "comment": comment,
        }
        results.append(result)
        print(f"    提取答案: {extracted} | 正确答案: {q.get('correct_answer', '')} | 得分: {score}")

        if i < len(questions) - 1:
            time.sleep(request_interval)

    total = len(results)
    correct = sum(1 for r in results if r["score"] == 1)
    acc = correct / total if total > 0 else 0
    print(f"\n  --- {benchmark_name} 汇总 ---")
    print(f"  总题数: {total}  正确: {correct}  准确率: {acc:.2%}")

    return {
        "benchmark_name": benchmark_name,
        "benchmark_display": benchmark_display,
        "questions": questions,
        "results": results,
    }


def run_t2i_benchmark(benchmark_name, benchmark_display, filepath,
                      test_api_key, test_base_url, test_model,
                      judge_api_key, judge_base_url, judge_model,
                      num_samples=None, request_interval=1.0, timeout=600,
                      image_size="1024x1024"):
    """
    对一个文生图题集执行完整评测流程，返回结果字典。
    与 run_benchmark 类似，但调用 ask_model_t2i 生成图片，再用打分模型评分。

    评分流程：
      1. 调用待测模型的 /v1/images/generations 接口生成图片
      2. 将生成的图片（base64 或 URL）发送给打分模型
      3. 打分模型根据提示词和评分维度给出 0-10 分
    """
    print(f"\n{'='*60}")
    print(f"  [文生图] 题集: {benchmark_name}")
    print(f"  文件: {filepath}")
    print(f"{'='*60}")

    questions = parse_benchmark(benchmark_name, filepath)
    if not questions:
        print("  [警告] 未解析到任何题目，跳过")
        return None
    print(f"  解析完成: 共 {len(questions)} 题, 类型: {questions[0]['type']}")

    if num_samples and num_samples < len(questions):
        questions = questions[:num_samples]
        print(f"  采样: 取前 {num_samples} 题")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        qtype = q["type"]
        prompt_text = q["question_text"]
        print(f"\n  [{i+1}/{len(questions)}] Q{qid} ({qtype})", end=" ", flush=True)
        print(f"\n    提示词: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")

        # 调用文生图模型生成图片
        image_data, raw_resp = ask_model_t2i(
            prompt_text, test_api_key, test_base_url, test_model,
            timeout=timeout, size=image_size,
        )

        if not image_data:
            print("    [跳过] 文生图模型未返回图片")
            results.append({
                "qid": qid, "model_output": "", "extracted_answer": "",
                "correct_answer": q.get("correct_answer", ""),
                "score": 0, "comment": "模型未生成图片",
            })
            if i < len(questions) - 1:
                time.sleep(request_interval)
            continue

        # 显示生成结果标识（base64 截断或 URL）
        display = image_data[:60] + "..." if len(image_data) > 60 else image_data
        print(f"    生成图片: {display}")

        # 调用打分模型评分（model_output 即图片标识）
        score, comment = score_answer(
            q, image_data, judge_api_key, judge_base_url, judge_model, timeout,
        )

        result = {
            "qid": qid,
            "model_output": image_data,
            "extracted_answer": image_data,
            "correct_answer": q.get("correct_answer", ""),
            "score": score,
            "comment": comment,
        }
        results.append(result)
        print(f"    得分: {score}/10  理由: {comment[:80]}")

        if i < len(questions) - 1:
            time.sleep(request_interval)

    total = len(results)
    score_sum = sum(r["score"] for r in results)
    avg = score_sum / total if total > 0 else 0
    print(f"\n  --- {benchmark_name} 汇总 ---")
    print(f"  总题数: {total}  总分: {score_sum:.1f}  均分: {avg:.2f}/10")

    return {
        "benchmark_name": benchmark_name,
        "benchmark_display": benchmark_display,
        "questions": questions,
        "results": results,
    }


def _get_output_subdir(base_dir, model_name):
    """
    在 results 目录下为模型创建子文件夹，若已有同名则自动加 _1, _2...
    返回完整子文件夹路径。
    """
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    candidate = os.path.join(base_dir, safe_name)
    if not os.path.exists(candidate):
        return candidate

    idx = 1
    while True:
        candidate = os.path.join(base_dir, f"{safe_name}_{idx}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _parse_args():
    """解析命令行参数，支持指定模型和选择测试模块。"""
    parser = argparse.ArgumentParser(
        description="模型自动测试程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试全部模块（默认）
  python run.py test

  # 指定模型名（覆盖 config.py 中的 TEST_MODEL_NAME）
  python run.py test --model qwen36-35b-a3b

  # 仅测试视觉识图模块（增量更新，不覆盖已有其他模块结果）
  python run.py test --model qwen36-35b-a3b --modules vision_recognition

  # 仅测试文生图模块
  python run.py test --model qwen36-35b-a3b --modules image_generation

  # 测试纯文本 + 效率（跳过多模态）
  python run.py test --modules text,efficiency

  # 同时指定 API 地址和密钥
  python run.py test --model my-model --base-url https://api.example.com/v1/chat/completions --api-key sk-xxx

  # 指定测试题数（采样）
  python run.py test --num-samples 5
""",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="待测模型名（覆盖 config.py 的 TEST_MODEL_NAME）")
    parser.add_argument("--api-key", type=str, default=None,
                        help="待测模型 API 密钥（覆盖 config.py 的 TEST_API_KEY）")
    parser.add_argument("--base-url", type=str, default=None,
                        help="待测模型 API 地址（覆盖 config.py 的 TEST_BASE_URL）")
    parser.add_argument("--modules", type=str, default=None,
                        help=("选择测试模块，逗号分隔。可选: "
                              "text(纯文本), vision_recognition(视觉识图), "
                              "image_generation(视觉生图), efficiency(效率)。"
                              "默认全部测试"))
    parser.add_argument("--num-samples", type=int, default=None,
                        help="每题集最多测试题数（采样），不指定则测试全部")
    parser.add_argument("--num-samples-map", type=str, default=None,
                        help="分题库采样题数（JSON 字符串，如 '{\"mmlu\":5,\"gsm8k\":3}'），"
                             "优先级高于 --num-samples，未指定的题库用默认值")
    parser.add_argument("--skip-probe", action="store_true",
                        help="跳过能力探测，直接执行测试（若 API 明确支持可加速）")
    return parser.parse_args()


def _resolve_modules(modules_arg):
    """解析 --modules 参数，返回实际要测试的模块列表。"""
    if not modules_arg:
        return list(DEFAULT_MODULES)
    parts = [m.strip() for m in modules_arg.split(",") if m.strip()]
    invalid = [p for p in parts if p not in TEST_MODULES]
    if invalid:
        print(f"  [错误] 未知的模块名: {invalid}")
        print(f"  可选模块: {list(TEST_MODULES.keys())}")
        sys.exit(1)
    return parts


def main():
    args = _parse_args()

    # ========== 应用命令行覆盖 ==========
    test_model = args.model or cfg.TEST_MODEL_NAME
    test_api_key = args.api_key or cfg.TEST_API_KEY
    test_base_url = args.base_url or cfg.TEST_BASE_URL
    modules = _resolve_modules(args.modules)
    num_samples_override = args.num_samples

    # 分题库采样题数（优先级最高）：JSON 字符串解析为 dict
    num_samples_map = {}
    if args.num_samples_map:
        try:
            import json as _json
            num_samples_map = _json.loads(args.num_samples_map)
            if not isinstance(num_samples_map, dict):
                print("  [警告] --num-samples-map 解析结果非字典，已忽略")
                num_samples_map = {}
            else:
                # 转换值为 int
                num_samples_map = {k: int(v) for k, v in num_samples_map.items() if v}
                print(f"  分题库采样配置: {num_samples_map}")
        except Exception as e:
            print(f"  [警告] --num-samples-map 解析失败: {e}，已忽略")
            num_samples_map = {}

    print("=" * 60)
    print("  模型自动测试程序")
    print(f"  待测模型: {test_model}  @  {test_base_url}")
    print(f"  打分模型: {cfg.JUDGE_MODEL_NAME}  @  {cfg.JUDGE_BASE_URL}")
    print(f"  项目根目录: {cfg.PROJECT_ROOT}")
    print(f"  测试模块: {', '.join(modules)}")
    for m in modules:
        print(f"    - {m}: {TEST_MODULES.get(m, m)}")
    print("=" * 60)

    # ========== 预热：建立 API 连接，避免第一题超长等待 ==========
    print("\n  预热 API 连接...")
    _warm_up(test_api_key, test_base_url, test_model, timeout=30)
    if cfg.JUDGE_BASE_URL != test_base_url or cfg.JUDGE_MODEL_NAME != test_model:
        _warm_up(cfg.JUDGE_API_KEY, cfg.JUDGE_BASE_URL, cfg.JUDGE_MODEL_NAME, timeout=30)
    print("  预热完成，开始评测\n")

    # ========== 能力探测：明确判断多模态能力是否可用 ==========
    # 探测结果会写入 capability_status，路由器据此区分"未测试"与"不支持"
    # 避免出现"程序认为不具备多模态能力但实际是有的"误判
    capability_status = {}
    for cap in CAPABILITY_TYPES:
        capability_status[cap] = CAPABILITY_STATUS_NOT_TESTED

    if not args.skip_probe:
        if "vision_recognition" in modules:
            print(f"\n{'='*60}")
            print("  能力探测：视觉识图（Vision 输入）")
            print(f"{'='*60}")
            probe = probe_vision_capability(test_api_key, test_base_url, test_model)
            if probe is True:
                capability_status["vision_recognition"] = CAPABILITY_STATUS_SUPPORTED
            elif probe is False:
                capability_status["vision_recognition"] = CAPABILITY_STATUS_UNSUPPORTED
                print("  → 判定模型不支持视觉识图，跳过该模块测试")
            else:
                # None：无法确定，保留 not_tested，但仍尝试测试（避免误判）
                print("  → 无法确定是否支持，仍尝试执行测试（避免误判）")

        if "image_generation" in modules:
            print(f"\n{'='*60}")
            print("  能力探测：视觉生图（/v1/images/generations）")
            print(f"{'='*60}")
            probe = probe_t2i_capability(test_api_key, test_base_url, test_model)
            if probe is True:
                capability_status["image_generation"] = CAPABILITY_STATUS_SUPPORTED
            elif probe is False:
                capability_status["image_generation"] = CAPABILITY_STATUS_UNSUPPORTED
                print("  → 判定模型不支持文生图，跳过该模块测试")
            else:
                print("  → 无法确定是否支持，仍尝试执行测试（避免误判）")

    # ========== 确定输出子文件夹 ==========
    output_dir = _get_output_subdir(cfg.OUTPUT_DIR, test_model)
    print(f"\n  结果输出: {output_dir}\n")

    all_results = []
    multimodal_results = []
    t2i_results = []
    eff_result = None

    # 各题集题目数量限制（按文件名规定的数字取前 N 题）
    BENCH_NUM_SAMPLES = {
        "mmlu": 30,
        "gsm8k": 10,
        "hellaswag": 20,
        "workplace_pm": 20,
        "workplace_secretary": 20,
        "bbh_semantic": 10,
        "bbh_math": 10,
        "longbench": 10,
    }

    # 多模态题集默认题数（按文件名数字）
    MULTIMODAL_BENCH_NUM_SAMPLES = {
        "chartqa": 20,
        "textvqa": 20,
        "mathvista": 20,
        "vqa": 30,
        "mmmu": 30,
    }

    # 文生图题集默认题数
    T2I_BENCH_NUM_SAMPLES = {
        "t2i": 15,
    }

    # 分题库采样题数解析函数：优先级 map > override > default
    def _resolve_num_samples(bench_key, default_value):
        """优先级：num_samples_map[bench_key] > num_samples_override > default_value"""
        if bench_key in num_samples_map:
            return num_samples_map[bench_key]
        if num_samples_override:
            return num_samples_override
        return default_value

    # ========== 模块1: 纯文本题集 ==========
    if "text" in modules:
        for bench_key, fname in cfg.BENCHMARK_FILES.items():
            filepath = _get_filepath(bench_key, fname)
            bench_display = cfg.BENCHMARK_NAMES.get(bench_key, bench_key)

            if not os.path.exists(filepath):
                print(f"\n  [警告] 题集文件不存在，跳过: {filepath}")
                continue

            bench_result = run_benchmark(
                benchmark_name=bench_key,
                benchmark_display=bench_display,
                filepath=filepath,
                test_api_key=test_api_key,
                test_base_url=test_base_url,
                test_model=test_model,
                judge_api_key=cfg.JUDGE_API_KEY,
                judge_base_url=cfg.JUDGE_BASE_URL,
                judge_model=cfg.JUDGE_MODEL_NAME,
                num_samples=_resolve_num_samples(bench_key, BENCH_NUM_SAMPLES.get(bench_key, cfg.NUM_SAMPLES)),
                request_interval=cfg.REQUEST_INTERVAL,
                timeout=cfg.REQUEST_TIMEOUT,
            )
            if bench_result:
                all_results.append(bench_result)
    else:
        print("\n  [跳过] 纯文本模块未选择")

    # ========== 模块2: 视觉识图题集 ==========
    if "vision_recognition" in modules:
        if capability_status.get("vision_recognition") == CAPABILITY_STATUS_UNSUPPORTED:
            print("\n  [跳过] 视觉识图模块：能力探测判定为不支持")
        else:
            print(f"\n{'='*60}")
            print("  开始多模态视觉题集测试（视觉识图）")
            print(f"{'='*60}")
            for bench_key, fname in cfg.MULTIMODAL_BENCHMARK_FILES.items():
                filepath = os.path.join(cfg.PROJECT_ROOT, fname)
                bench_display = cfg.MULTIMODAL_BENCHMARK_NAMES.get(bench_key, bench_key)

                if not os.path.exists(filepath):
                    print(f"\n  [警告] 多模态题集文件不存在，跳过: {filepath}")
                    continue

                bench_result = run_multimodal_benchmark(
                    benchmark_name=bench_key,
                    benchmark_display=bench_display,
                    filepath=filepath,
                    test_api_key=test_api_key,
                    test_base_url=test_base_url,
                    test_model=test_model,
                    judge_api_key=cfg.JUDGE_API_KEY,
                    judge_base_url=cfg.JUDGE_BASE_URL,
                    judge_model=cfg.JUDGE_MODEL_NAME,
                    num_samples=_resolve_num_samples(bench_key, MULTIMODAL_BENCH_NUM_SAMPLES.get(bench_key, cfg.NUM_SAMPLES)),
                    request_interval=cfg.REQUEST_INTERVAL,
                    timeout=cfg.REQUEST_TIMEOUT,
                )
                if bench_result:
                    multimodal_results.append(bench_result)
                    all_results.append(bench_result)
                    # 测试成功产出结果 → 标记为 supported
                    capability_status["vision_recognition"] = CAPABILITY_STATUS_SUPPORTED
    else:
        print("\n  [跳过] 视觉识图模块未选择")

    # ========== 模块3: 文生图题集 ==========
    if "image_generation" in modules:
        if capability_status.get("image_generation") == CAPABILITY_STATUS_UNSUPPORTED:
            print("\n  [跳过] 文生图模块：能力探测判定为不支持")
        else:
            print(f"\n{'='*60}")
            print("  开始文生图（text-to-image）题集测试")
            print(f"{'='*60}")
            for bench_key, fname in cfg.T2I_BENCHMARK_FILES.items():
                filepath = os.path.join(cfg.PROJECT_ROOT, fname)
                bench_display = cfg.T2I_BENCHMARK_NAMES.get(bench_key, bench_key)

                if not os.path.exists(filepath):
                    print(f"\n  [警告] 文生图题集文件不存在，跳过: {filepath}")
                    continue

                bench_result = run_t2i_benchmark(
                    benchmark_name=bench_key,
                    benchmark_display=bench_display,
                    filepath=filepath,
                    test_api_key=test_api_key,
                    test_base_url=test_base_url,
                    test_model=test_model,
                    judge_api_key=cfg.JUDGE_API_KEY,
                    judge_base_url=cfg.JUDGE_BASE_URL,
                    judge_model=cfg.JUDGE_MODEL_NAME,
                    num_samples=_resolve_num_samples(bench_key, T2I_BENCH_NUM_SAMPLES.get(bench_key, cfg.NUM_SAMPLES)),
                    request_interval=cfg.REQUEST_INTERVAL,
                    timeout=cfg.REQUEST_TIMEOUT,
                    image_size=getattr(cfg, 'T2I_IMAGE_SIZE', '1024x1024'),
                )
                if bench_result:
                    t2i_results.append(bench_result)
                    all_results.append(bench_result)
                    capability_status["image_generation"] = CAPABILITY_STATUS_SUPPORTED
    else:
        print("\n  [跳过] 文生图模块未选择")

    # ========== 模块4: 效率测试 ==========
    if "efficiency" in modules and getattr(cfg, 'ENABLE_EFFICIENCY_TEST', False):
        from benchmark_efficiency import run_efficiency_benchmark
        eff_result = run_efficiency_benchmark(
            api_key=test_api_key,
            base_url=test_base_url,
            model_name=test_model,
            single_rounds=getattr(cfg, 'EFFICIENCY_SINGLE_ROUNDS', 3),
            throughput_threshold=getattr(cfg, 'EFFICIENCY_THROUGHPUT_THRESHOLD', 0.65),
            timeout=getattr(cfg, 'EFFICIENCY_TIMEOUT', 120),
        )
        if eff_result:
            export_efficiency(eff_result, test_model, output_dir)
    else:
        print("\n  [跳过] 效率模块未选择或未启用")

    # 导出：每题集一个独立 xlsx，文件名含模型名+题库名
    print(f"\n{'='*60}")
    print("  导出结果...")
    export_all(all_results, test_model, output_dir)

    # 总汇总
    print(f"\n{'='*60}")
    print("  全部评测完成！总汇总：")
    print(f"{'='*60}")
    total_obj = 0
    correct_obj = 0
    total_sub = 0
    score_sub = 0
    for data in all_results:
        total = len(data["results"])
        is_sub = data["questions"][0]["type"] in ("subjective", "long_context") if data["questions"] else False
        if is_sub:
            tscore = sum(r["score"] for r in data["results"])
            avg = tscore / total if total > 0 else 0
            total_sub += total
            score_sub += tscore
            print(f"  {data['benchmark_display']:25s} : {total:3d}题  均分: {avg:.2f}/10")
        else:
            correct = sum(1 for r in data["results"] if r["score"] == 1)
            acc = correct / total if total > 0 else 0
            total_obj += total
            correct_obj += correct
            print(f"  {data['benchmark_display']:25s} : {correct:3d}/{total:3d}  ({acc:.2%})")

    if total_obj > 0:
        print(f"  {'客观题总计':25s} : {correct_obj:3d}/{total_obj:3d}  ({correct_obj/total_obj:.2%})")
    if total_sub > 0:
        print(f"  {'主观题总计':25s} : {total_sub:3d}题  均分: {score_sub/total_sub:.2f}/10")

    # 能力状态汇总
    print(f"\n  能力状态:")
    for cap, status in capability_status.items():
        print(f"    - {cap}: {status}")

    # 更新 model_benchmarks.json（增量合并，不覆盖未测模块的已有结果）
    json_path = getattr(cfg, 'BENCHMARKS_JSON_PATH', None)
    if json_path:
        print(f"\n{'='*60}")
        print("  更新 model_benchmarks.json（增量合并）...")
        summary = build_benchmark_summary(
            all_results, eff_result, multimodal_results, t2i_results,
            capability_status=capability_status,
        )
        update_benchmarks_json(json_path, test_model, summary)


if __name__ == "__main__":
    main()
