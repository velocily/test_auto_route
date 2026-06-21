"""
============================================================================
模型效率测试模块
============================================================================
测试指标:
  1. 单用户首Token延迟 (TTFT)
  2. 单用户平均Token吞吐 (tokens/sec)
  3. 并行访问稳定数量（并发上限）
"""
import re
import time
import json
import threading
import statistics
import requests

from model_api import _get_session


_EFFICIENCY_PROMPT = (
    "请详细解释机器学习和深度学习的区别与联系，"
    "包括定义、核心算法、应用场景等方面，字数在300字左右。"
)


def _estimate_tokens(text):
    if not text:
        return 0
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    non_cn = re.sub(r'[\u4e00-\u9fff]', '', text)
    words = len(non_cn.split())
    return int(cn_chars / 1.5 + words)


def _call_stream(api_key, base_url, model_name, prompt, timeout=120):
    """流式调用模型 API，使用持久化 Session 复用连接"""
    session = _get_session()
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": True,
        "max_tokens": 512,
    }

    start_time = time.perf_counter()
    first_token_time = None
    full_text = ""
    usage_tokens = None

    try:
        r = session.post(base_url, headers=headers, json=payload,
                         timeout=(15, timeout), stream=True)
        r.raise_for_status()

        for line in r.iter_lines():
            if not line:
                continue
            # 兼容 bytes 和 str
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta") if choices[0] else None
            if delta is None:
                continue
            content = delta.get("content")
            if content is None:
                continue

            if first_token_time is None and content:
                first_token_time = time.perf_counter()

            full_text += content

            if "usage" in chunk and chunk["usage"]:
                usage_tokens = chunk["usage"].get("completion_tokens")

        end_time = time.perf_counter()

        if usage_tokens is None:
            tokens = _estimate_tokens(full_text)
        else:
            tokens = usage_tokens

        total_time_ms = (end_time - start_time) * 1000
        if first_token_time:
            ttft_ms = (first_token_time - start_time) * 1000
            generation_time_s = end_time - first_token_time
        else:
            ttft_ms = total_time_ms
            generation_time_s = end_time - start_time if end_time > start_time else 0.001

        throughput = tokens / generation_time_s if generation_time_s > 0 else 0

        return {
            "success": True,
            "ttft_ms": ttft_ms,
            "total_time_ms": total_time_ms,
            "tokens": tokens,
            "text": full_text,
            "throughput": throughput,
            "error": "",
        }

    except Exception as e:
        return {
            "success": False,
            "ttft_ms": 0,
            "total_time_ms": 0,
            "tokens": 0,
            "text": "",
            "throughput": 0,
            "error": str(e),
        }


def _single_user_test(api_key, base_url, model_name, rounds=3, timeout=120):
    print(f"  [单用户测试] 进行 {rounds} 轮请求...")
    results = []
    for i in range(rounds):
        print(f"    第 {i+1}/{rounds} 轮...", end=" ", flush=True)
        r = _call_stream(api_key, base_url, model_name, _EFFICIENCY_PROMPT, timeout)
        if r["success"]:
            print(f"TTFT={r['ttft_ms']:.0f}ms, 吞吐={r['throughput']:.2f}tok/s, tokens={r['tokens']}")
            results.append(r)
        else:
            print(f"失败: {r['error']}")
        if i < rounds - 1:
            time.sleep(1)

    if not results:
        return {"success": False, "error": "所有单用户请求均失败"}

    avg_ttft = statistics.mean([r["ttft_ms"] for r in results])
    avg_throughput = statistics.mean([r["throughput"] for r in results])
    avg_tokens = statistics.mean([r["tokens"] for r in results])

    print(f"  [单用户结果] 平均TTFT={avg_ttft:.0f}ms, 平均吞吐={avg_throughput:.2f}tok/s, 平均tokens={avg_tokens:.0f}")

    return {
        "success": True,
        "avg_ttft_ms": avg_ttft,
        "avg_throughput": avg_throughput,
        "avg_tokens": avg_tokens,
        "raw_results": results,
    }


def _concurrent_test(api_key, base_url, model_name, concurrency, timeout=120):
    print(f"  [并发测试] 并发数={concurrency}...")
    results = [None] * concurrency
    threads = []

    def _worker(idx):
        results[idx] = _call_stream(api_key, base_url, model_name, _EFFICIENCY_PROMPT, timeout)

    start_all = time.perf_counter()
    for i in range(concurrency):
        t = threading.Thread(target=_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_time = time.perf_counter() - start_all

    success_results = [r for r in results if r["success"]]
    failed = concurrency - len(success_results)

    if not success_results:
        print(f"    全部失败")
        return {"success": False, "error": "所有并发请求均失败"}

    avg_ttft = statistics.mean([r["ttft_ms"] for r in success_results])
    avg_throughput = statistics.mean([r["throughput"] for r in success_results])
    avg_tokens = statistics.mean([r["tokens"] for r in success_results])

    print(f"    成功{len(success_results)}/{concurrency} (失败{failed}), "
          f"平均TTFT={avg_ttft:.0f}ms, 平均吞吐={avg_throughput:.2f}tok/s, "
          f"总耗时={total_time:.1f}s")

    return {
        "success": True,
        "concurrency": concurrency,
        "avg_ttft_ms": avg_ttft,
        "avg_throughput": avg_throughput,
        "avg_tokens": avg_tokens,
        "failed": failed,
        "raw_results": success_results,
    }


def run_efficiency_benchmark(api_key, base_url, model_name,
                             single_rounds=3,
                             throughput_threshold=0.65,
                             timeout=120):
    """
    效率基准测试。
    并发扫描策略：
      - 先用初始列表 [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32] 密集扫描
      - 之后每次 +10 持续递增，直到吞吐跌穿阈值 或 全部失败 或 达到上限
    """
    from config import EFFICIENCY_INITIAL_CONCURRENCY, EFFICIENCY_CONCURRENCY_STEP, EFFICIENCY_MAX_CONCURRENCY

    def _concurrency_generator():
        """动态生成并发数：先密集再大步，无上限（安全上限 200）"""
        for c in EFFICIENCY_INITIAL_CONCURRENCY:
            yield c
        last = EFFICIENCY_INITIAL_CONCURRENCY[-1] if EFFICIENCY_INITIAL_CONCURRENCY else 32
        max_c = EFFICIENCY_MAX_CONCURRENCY if EFFICIENCY_MAX_CONCURRENCY is not None else 200
        while True:
            last += EFFICIENCY_CONCURRENCY_STEP
            if last > max_c:
                break
            yield last

    print(f"\n{'='*60}")
    print("  题集: efficiency (模型效率测试)")
    print(f"{'='*60}")

    single_result = _single_user_test(api_key, base_url, model_name, single_rounds, timeout)
    if not single_result["success"]:
        print(f"  [错误] 单用户测试失败: {single_result.get('error')}")
        return {
            "benchmark_name": "efficiency",
            "benchmark_display": "模型效率测试",
            "single_user": single_result,
            "concurrency_sweep": [],
            "stable_concurrency": None,
            "stable_result": None,
        }

    baseline_throughput = single_result["avg_throughput"]
    threshold_throughput = baseline_throughput * throughput_threshold

    print(f"\n  基准吞吐: {baseline_throughput:.2f} tok/s")
    print(f"  阈值吞吐 (65%): {threshold_throughput:.2f} tok/s")

    sweep_results = []
    stable_concurrency = None
    stable_result = None

    for conc in _concurrency_generator():
        if conc == 1:
            r = {
                "success": True,
                "concurrency": 1,
                "avg_ttft_ms": single_result["avg_ttft_ms"],
                "avg_throughput": single_result["avg_throughput"],
                "avg_tokens": single_result["avg_tokens"],
                "failed": 0,
            }
        else:
            r = _concurrent_test(api_key, base_url, model_name, conc, timeout)

        if not r["success"]:
            print(f"    并发数 {conc} 全部失败，停止扫描")
            stable_concurrency = sweep_results[-1]["concurrency"] if sweep_results else 1
            stable_result = sweep_results[-1] if sweep_results else None
            break

        # 出现任何失败请求即视为达到上限
        if r.get("failed", 0) > 0:
            print(f"    并发数 {conc} 出现失败 ({r['failed']}/{conc})，以此为上限")
            # 当前并发已不稳定，取上一个全部成功的并发数
            stable_concurrency = sweep_results[-1]["concurrency"] if sweep_results else 1
            stable_result = sweep_results[-1] if sweep_results else None
            break

        sweep_results.append(r)

        if r["avg_throughput"] <= threshold_throughput:
            if stable_concurrency is None:
                if len(sweep_results) >= 2:
                    stable_concurrency = sweep_results[-2]["concurrency"]
                    stable_result = sweep_results[-2]
                else:
                    stable_concurrency = 1
                    stable_result = sweep_results[0]
                print(f"\n  >>> 发现吞吐低于65%阈值，稳定并发数 = {stable_concurrency}")
            else:
                # 已经找到稳定点，再取一个数据点确认后停止扫描
                break

    if stable_concurrency is None and sweep_results:
        stable_concurrency = sweep_results[-1]["concurrency"]
        stable_result = sweep_results[-1]
        print(f"\n  >>> 所有测试并发均未低于65%阈值，取最大成功并发数 = {stable_concurrency}")

    print(f"\n  --- efficiency 汇总 ---")
    print(f"  单用户平均TTFT: {single_result['avg_ttft_ms']:.0f} ms")
    print(f"  单用户平均吞吐: {single_result['avg_throughput']:.2f} tok/s")
    print(f"  稳定并发数: {stable_concurrency}")
    if stable_result:
        print(f"  该并发下平均TTFT: {stable_result['avg_ttft_ms']:.0f} ms")
        print(f"  该并发下平均吞吐: {stable_result['avg_throughput']:.2f} tok/s")

    return {
        "benchmark_name": "efficiency",
        "benchmark_display": "模型效率测试",
        "single_user": single_result,
        "concurrency_sweep": sweep_results,
        "stable_concurrency": stable_concurrency,
        "stable_result": stable_result,
    }
