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

# 将当前目录加入 sys.path，确保可以 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from parser import parse_benchmark
from model_api import (
    ask_model,
    score_answer,
    _warm_up,
    PROMPT_BUILDER_MAP,
    EXTRACTOR_MAP,
)
from utils import export_all, export_efficiency
from benchmarks_json import build_benchmark_summary, update_benchmarks_json


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
        print(f"\n  --- {benchmark_name} 汇总 ---")
        print(f"  总题数: {total}  总分: {total_score}  均分: {avg_score:.2f}/10")
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


def main():
    print("=" * 60)
    print("  模型自动测试程序")
    print(f"  待测模型: {cfg.TEST_MODEL_NAME}  @  {cfg.TEST_BASE_URL}")
    print(f"  打分模型: {cfg.JUDGE_MODEL_NAME}  @  {cfg.JUDGE_BASE_URL}")
    print(f"  项目根目录: {cfg.PROJECT_ROOT}")
    print("=" * 60)

    # ========== 预热：建立 API 连接，避免第一题超长等待 ==========
    print("\n  预热 API 连接...")
    _warm_up(cfg.TEST_API_KEY, cfg.TEST_BASE_URL, cfg.TEST_MODEL_NAME, timeout=30)
    if cfg.JUDGE_BASE_URL != cfg.TEST_BASE_URL or cfg.JUDGE_MODEL_NAME != cfg.TEST_MODEL_NAME:
        _warm_up(cfg.JUDGE_API_KEY, cfg.JUDGE_BASE_URL, cfg.JUDGE_MODEL_NAME, timeout=30)
    print("  预热完成，开始评测\n")

    # ========== 确定输出子文件夹 ==========
    output_dir = _get_output_subdir(cfg.OUTPUT_DIR, cfg.TEST_MODEL_NAME)
    print(f"  结果输出: {output_dir}\n")

    all_results = []
    eff_result = None

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
            test_api_key=cfg.TEST_API_KEY,
            test_base_url=cfg.TEST_BASE_URL,
            test_model=cfg.TEST_MODEL_NAME,
            judge_api_key=cfg.JUDGE_API_KEY,
            judge_base_url=cfg.JUDGE_BASE_URL,
            judge_model=cfg.JUDGE_MODEL_NAME,
            num_samples=cfg.NUM_SAMPLES,
            request_interval=cfg.REQUEST_INTERVAL,
            timeout=cfg.REQUEST_TIMEOUT,
        )
        if bench_result:
            all_results.append(bench_result)

    # ========== 效率测试 ==========
    if getattr(cfg, 'ENABLE_EFFICIENCY_TEST', False):
        from benchmark_efficiency import run_efficiency_benchmark
        eff_result = run_efficiency_benchmark(
            api_key=cfg.TEST_API_KEY,
            base_url=cfg.TEST_BASE_URL,
            model_name=cfg.TEST_MODEL_NAME,
            single_rounds=getattr(cfg, 'EFFICIENCY_SINGLE_ROUNDS', 3),
            throughput_threshold=getattr(cfg, 'EFFICIENCY_THROUGHPUT_THRESHOLD', 0.65),
            timeout=getattr(cfg, 'EFFICIENCY_TIMEOUT', 120),
        )
        if eff_result:
            export_efficiency(eff_result, cfg.TEST_MODEL_NAME, output_dir)

    # 导出：每题集一个独立 xlsx，文件名含模型名+题库名
    print(f"\n{'='*60}")
    print("  导出结果...")
    export_all(all_results, cfg.TEST_MODEL_NAME, output_dir)

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

    # 更新 model_benchmarks.json
    json_path = getattr(cfg, 'BENCHMARKS_JSON_PATH', None)
    if json_path:
        print(f"\n{'='*60}")
        print("  更新 model_benchmarks.json...")
        summary = build_benchmark_summary(all_results, eff_result)
        update_benchmarks_json(json_path, cfg.TEST_MODEL_NAME, summary)


if __name__ == "__main__":
    main()
