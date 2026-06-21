"""对比测试：用 WhoEngine 跑原版 21 道测试题"""
import sys, os

# Windows 中文编码兼容性修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 自动检测项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'router'))
os.chdir(_PROJECT_ROOT)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 删除旧缓存
cache = 'whoengine.pt'
if os.path.exists(cache):
    os.remove(cache)

from whoengine import get_router, whoengine_route

router = get_router(force_retrain=True)
print(f'Domains: {router.domains}')

# 原版 IR3DE_测试题与结果.md 中的 21 道题
test_cases = [
    ('光合作用的主要产物是什么？', 'mmlu'),
    ('第二次世界大战爆发于哪一年？', 'mmlu'),
    ('What is the chemical formula of water?', 'mmlu'),
    ('小明有 15 个苹果，给了小红 6 个，还剩几个？', 'gsm8k'),
    ('3x + 5 = 20，求 x 的值', 'gsm8k'),
    ('一个圆的半径是 7 厘米，周长是多少？', 'gsm8k'),
    ('为什么冬天呼出的气是白色的？', 'hellaswag'),
    ('如果你在森林中迷路，最应该避免做什么？', 'hellaswag'),
    ('如何正确地切洋葱以避免流泪？', 'hellaswag'),
    ('Write a Python function to implement binary search', 'bbh_semantic'),
    ('用 Python 写一个函数判断字符串是否是回文', 'bbh_semantic'),
    ('什么是 P vs NP 问题？', 'bbh_semantic'),
    ('请阅读以下关于气候变化的 5000 字报告，并总结其主要发现', 'longbench'),
    ('请对比以下两篇论文的实验设计、数据集和评估指标的异同', 'longbench'),
    ('以下是多份客户反馈记录，请归纳出最常见的三个问题和建议', 'longbench'),
    ('请为一款新 App 的开发制定详细的项目计划', 'project_manager'),
    ('项目进度落后了 2 周，请分析可能的原因并提出赶工方案', 'project_manager'),
    ('如何评估和管理项目中的技术风险？', 'project_manager'),
    ('请帮我起草一封正式的商务邀请函', 'secretary'),
    ('帮我安排下周的出差行程，包括机票、酒店和会议预约', 'secretary'),
    ('请起草一份给全公司的放假通知', 'secretary'),
]

correct = 0
for i, (query, expected) in enumerate(test_cases, 1):
    result = whoengine_route(query, strategy='average')
    predicted = result['route_domain']
    ok = predicted == expected
    if ok:
        correct += 1
    mark = '✓' if ok else '✗'
    print(f'Q{i:2d}: pred={predicted:20s} expected={expected:20s} {mark} conf={result["route_confidence"]:.3f}')

print(f'\nTotal: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.1f}%')