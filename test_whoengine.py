# test_whoengine.py
# 对比测试：KNN 路由 vs 句级路由（average）vs Token 级路由（majority_voting）
# 纯嵌入路由，无关键词匹配

import sys
import os

# ====== 兼容中文输出（修复 Windows 命令行 GBK 乱码）======
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "router"))

import torch
from whoengine import get_router, whoengine_route

# 显示 GPU 信息
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("GPU: 无 (使用 CPU)")

router = get_router(force_retrain=True)
print(f"路由器已就绪，已训练 {len(router.domains)} 个 domains: {router.domains}")
print(f"默认策略: {router.knn_router.k if router.knn_router else 'N/A'}-NN (KNN路由)")
print("=" * 80)

TEST_CASES = [
    # (题目, 预期 domain)
    # ====== 原测试题 ======
    ("光合作用的主要产物是什么？", "mmlu"),
    ("第二次世界大战爆发于哪一年？", "mmlu"),
    ("What is the chemical formula of water?", "mmlu"),
    ("小明有 15 个苹果，给了小红 6 个，还剩几个？", "gsm8k"),
    ("3x + 5 = 20，求 x 的值", "gsm8k"),
    ("一个圆的半径是 7 厘米，周长是多少？", "gsm8k"),
    ("为什么冬天呼出的气是白色的？", "hellaswag"),
    ("如果你在森林中迷路，最应该避免做什么？", "hellaswag"),
    ("如何正确地切洋葱以避免流泪？", "hellaswag"),
    ("Write a Python function to implement binary search", "bbh_semantic"),
    ("用 Python 写一个函数判断字符串是否是回文", "bbh_semantic"),
    ("什么是 P vs NP 问题？", "bbh_semantic"),
    ("请阅读以下关于气候变化的 5000 字报告，并总结其主要发现", "longbench"),
    ("请对比以下两篇论文的实验设计、数据集和评估指标的异同", "longbench"),
    ("以下是多份客户反馈记录，请归纳出最常见的三个问题和建议", "longbench"),
    ("请为一款新 App 的开发制定详细的项目计划", "project_manager"),
    ("项目进度落后了 2 周，请分析可能的原因并提出赶工方案", "project_manager"),
    ("如何评估和管理项目中的技术风险？", "project_manager"),
    ("请帮我起草一封正式的商务邀请函", "secretary"),
    ("帮我安排下周的出差行程，包括机票、酒店和会议预约", "secretary"),
    ("请起草一份给全公司的放假通知", "secretary"),
    # ====== 混合边界测试 ======
    ("1. 数学：小明有5个苹果，吃了2个，又买了3个，现在有多少个？", "gsm8k"),
    ("2. 知识问答：光合作用发生在植物的哪个细胞器中？", "mmlu"),
    ("3. 常识推理：如果一个人忘记带钥匙出门，他可能会做什么？", "hellaswag"),
    ("4. 编程语义：给定一个SQL查询 SELECT * FROM users WHERE age > 18，解释其含义。", "bbh_semantic"),
    ("5. 数学推理：设a,b为正整数，且a+b=10，求ab的最大值。", "bbh_math"),
    ("6. 长文本：请总结以下文章的主要观点……（伴随约2000字长文", "longbench"),
    ("7. 项目管理：请制定一个软件开发项目的WBS工作分解结构。", "project_manager"),
    ("8. 秘书工作：请帮我起草一份会议纪要，会议主题是Q3季度复盘。", "secretary"),
    ("9. 混合边界测试：请用数学方法证明2+2=4，并解释这个证明的教育意义。", "gsm8k"),
    ("10. 混合边界测试：资本资产定价模型(CAPM)中，β系数反映的是什么风险？", "mmlu"),
]

# 原测试题数量（前21题）
ORIGINAL_COUNT = 21

print("\n" + "=" * 80)
print("对比测试：Majority Voting vs Average vs Ensemble vs KNN vs KNN+Prior vs Ensemble V2")
print("策略：纯嵌入路由，无关键词匹配 | 升级：多池化+温度标定+KNN近邻软投票+关键词先验")
print("=" * 80)

correct_mv = 0
correct_avg = 0
correct_ens = 0
correct_knn = 0
correct_knn_prior = 0
correct_ens_v2 = 0
total = len(TEST_CASES)
results_mv = []
results_avg = []
results_ens = []
results_knn = []
results_knn_prior = []
results_ens_v2 = []

for i, (query, expected) in enumerate(TEST_CASES):
    print(f"\n--- 题目 {i+1}: {query[:60]}...")
    print(f"    预期 domain: {expected}")
    
    # Token 级路由（默认 majority_voting）
    result_mv = whoengine_route(query, strategy="majority_voting")
    detail_mv = result_mv.get("routing_detail", {})
    print(f"    [Token级] 预测: {result_mv['route_domain']} (置信度 {result_mv['route_confidence']:.3f}) "
          f"| {detail_mv.get('total_tokens', '?')}tokens, "
          f"top-{detail_mv.get('top_k_tokens', '?')} "
          f"| 投票: {detail_mv.get('vote_counts', {})}")
    if result_mv['route_domain'] == expected:
        correct_mv += 1
        results_mv.append(True)
        print(f"    [Token级] ✓ 正确")
    else:
        results_mv.append(False)
        print(f"    [Token级] ✗ 错误")
    
    # 句级路由（average）
    result_avg = whoengine_route(query, strategy="average")
    detail_avg = result_avg.get("routing_detail", {})
    print(f"    [句级]   预测: {result_avg['route_domain']} (置信度 {result_avg['route_confidence']:.3f}) "
          f"| 耗时 {detail_avg.get('routing_latency_ms', '?')}ms")
    if result_avg['route_domain'] == expected:
        correct_avg += 1
        results_avg.append(True)
        print(f"    [句级]   ✓ 正确")
    else:
        results_avg.append(False)
        print(f"    [句级]   ✗ 错误")

    # 集成路由（ensemble）
    result_ens = whoengine_route(query, strategy="ensemble")
    detail_ens = result_ens.get("routing_detail", {})
    print(f"    [集成]   预测: {result_ens['route_domain']} (置信度 {result_ens['route_confidence']:.3f}) "
          f"| 句级→{detail_ens.get('sentence_pred', '?')}, "
          f"token→{detail_ens.get('token_pred', '?')} "
          f"| 耗时 {detail_ens.get('routing_latency_ms', '?')}ms")
    if result_ens['route_domain'] == expected:
        correct_ens += 1
        results_ens.append(True)
        print(f"    [集成]   ✓ 正确")
    else:
        results_ens.append(False)
        print(f"    [集成]   ✗ 错误")

    # KNN 路由（新增）
    result_knn = whoengine_route(query, strategy="knn")
    detail_knn = result_knn.get("routing_detail", {})
    print(f"    [KNN]    预测: {result_knn['route_domain']} (置信度 {result_knn['route_confidence']:.3f}) "
          f"| k={detail_knn.get('knn_k', '?')} "
          f"| 耗时 {detail_knn.get('routing_latency_ms', '?')}ms")
    if result_knn['route_domain'] == expected:
        correct_knn += 1
        results_knn.append(True)
        print(f"    [KNN]    ✓ 正确")
    else:
        results_knn.append(False)
        print(f"    [KNN]    ✗ 错误")

    # KNN + 关键词先验路由（新增，推荐策略）
    result_knn_prior = whoengine_route(query, strategy="knn_prior")
    detail_knn_prior = result_knn_prior.get("routing_detail", {})
    hit_kw = detail_knn_prior.get("hit_keywords", {})
    hit_kw_str = ", ".join([f"{d}:{kws}" for d, kws in hit_kw.items()]) if hit_kw else "无"
    print(f"    [KNN+先验] 预测: {result_knn_prior['route_domain']} (置信度 {result_knn_prior['route_confidence']:.3f}) "
          f"| α={detail_knn_prior.get('prior_alpha', '?')} "
          f"| 命中: {hit_kw_str} "
          f"| 耗时 {detail_knn_prior.get('routing_latency_ms', '?')}ms")
    if result_knn_prior['route_domain'] == expected:
        correct_knn_prior += 1
        results_knn_prior.append(True)
        print(f"    [KNN+先验] ✓ 正确")
    else:
        results_knn_prior.append(False)
        print(f"    [KNN+先验] ✗ 错误")

    # 集成 V2 路由（新增：岭回归 + KNN）
    result_ens_v2 = whoengine_route(query, strategy="ensemble_v2")
    detail_ens_v2 = result_ens_v2.get("routing_detail", {})
    print(f"    [集成V2] 预测: {result_ens_v2['route_domain']} (置信度 {result_ens_v2['route_confidence']:.3f}) "
          f"| 岭回归→{detail_ens_v2.get('ridge_pred', '?')}, "
          f"KNN→{detail_ens_v2.get('knn_pred', '?')} "
          f"| 耗时 {detail_ens_v2.get('routing_latency_ms', '?')}ms")
    if result_ens_v2['route_domain'] == expected:
        correct_ens_v2 += 1
        results_ens_v2.append(True)
        print(f"    [集成V2] ✓ 正确")
    else:
        results_ens_v2.append(False)
        print(f"    [集成V2] ✗ 错误")

print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)
print(f"  Token级路由 (Majority Voting): {correct_mv}/{total} = {correct_mv/total*100:.1f}%")
print(f"  句级路由   (Average):         {correct_avg}/{total} = {correct_avg/total*100:.1f}%")
print(f"  集成路由   (Ensemble):        {correct_ens}/{total} = {correct_ens/total*100:.1f}%")
print(f"  KNN路由    (KNN k=20):        {correct_knn}/{total} = {correct_knn/total*100:.1f}%")
print(f"  KNN+先验   (KNN+Prior α=0.5): {correct_knn_prior}/{total} = {correct_knn_prior/total*100:.1f}%")
print(f"  集成V2路由 (Ridge+KNN):       {correct_ens_v2}/{total} = {correct_ens_v2/total*100:.1f}%")
best = max(correct_mv, correct_avg, correct_ens, correct_knn, correct_knn_prior, correct_ens_v2)
print(f"\n  最佳策略: ", end="")
if best == correct_knn_prior:
    print(f"KNN+Prior ({correct_knn_prior}/{total})")
elif best == correct_knn:
    print(f"KNN ({correct_knn}/{total})")
elif best == correct_ens_v2:
    print(f"Ensemble V2 ({correct_ens_v2}/{total})")
elif best == correct_ens:
    print(f"Ensemble ({correct_ens}/{total})")
elif best == correct_avg:
    print(f"Average ({correct_avg}/{total})")
else:
    print(f"Majority Voting ({correct_mv}/{total})")

# 分组统计
orig_mv = sum(1 for i in range(ORIGINAL_COUNT) if results_mv[i])
orig_avg = sum(1 for i in range(ORIGINAL_COUNT) if results_avg[i])
orig_ens = sum(1 for i in range(ORIGINAL_COUNT) if results_ens[i])
orig_knn = sum(1 for i in range(ORIGINAL_COUNT) if results_knn[i])
orig_knn_prior = sum(1 for i in range(ORIGINAL_COUNT) if results_knn_prior[i])
orig_ens_v2 = sum(1 for i in range(ORIGINAL_COUNT) if results_ens_v2[i])
boundary_mv = sum(1 for i in range(ORIGINAL_COUNT, total) if results_mv[i])
boundary_avg = sum(1 for i in range(ORIGINAL_COUNT, total) if results_avg[i])
boundary_ens = sum(1 for i in range(ORIGINAL_COUNT, total) if results_ens[i])
boundary_knn = sum(1 for i in range(ORIGINAL_COUNT, total) if results_knn[i])
boundary_knn_prior = sum(1 for i in range(ORIGINAL_COUNT, total) if results_knn_prior[i])
boundary_ens_v2 = sum(1 for i in range(ORIGINAL_COUNT, total) if results_ens_v2[i])
print(f"\n  原测试题 ({ORIGINAL_COUNT}题):")
print(f"    Token级:   {orig_mv}/{ORIGINAL_COUNT} = {orig_mv/ORIGINAL_COUNT*100:.1f}%")
print(f"    句级:     {orig_avg}/{ORIGINAL_COUNT} = {orig_avg/ORIGINAL_COUNT*100:.1f}%")
print(f"    集成:     {orig_ens}/{ORIGINAL_COUNT} = {orig_ens/ORIGINAL_COUNT*100:.1f}%")
print(f"    KNN:      {orig_knn}/{ORIGINAL_COUNT} = {orig_knn/ORIGINAL_COUNT*100:.1f}%")
print(f"    KNN+先验: {orig_knn_prior}/{ORIGINAL_COUNT} = {orig_knn_prior/ORIGINAL_COUNT*100:.1f}%")
print(f"    集成V2:   {orig_ens_v2}/{ORIGINAL_COUNT} = {orig_ens_v2/ORIGINAL_COUNT*100:.1f}%")
print(f"  边界测试 ({total - ORIGINAL_COUNT}题):")
print(f"    Token级:   {boundary_mv}/{total - ORIGINAL_COUNT} = {boundary_mv/(total - ORIGINAL_COUNT)*100:.1f}%")
print(f"    句级:     {boundary_avg}/{total - ORIGINAL_COUNT} = {boundary_avg/(total - ORIGINAL_COUNT)*100:.1f}%")
print(f"    集成:     {boundary_ens}/{total - ORIGINAL_COUNT} = {boundary_ens/(total - ORIGINAL_COUNT)*100:.1f}%")
print(f"    KNN:      {boundary_knn}/{total - ORIGINAL_COUNT} = {boundary_knn/(total - ORIGINAL_COUNT)*100:.1f}%")
print(f"    KNN+先验: {boundary_knn_prior}/{total - ORIGINAL_COUNT} = {boundary_knn_prior/(total - ORIGINAL_COUNT)*100:.1f}%")
print(f"    集成V2:   {boundary_ens_v2}/{total - ORIGINAL_COUNT} = {boundary_ens_v2/(total - ORIGINAL_COUNT)*100:.1f}%")
print("=" * 80)