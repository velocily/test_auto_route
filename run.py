# ============================================================
# test_auto_route - 自动测试 + 智能路由 一体化项目
# 入口脚本
# ============================================================

import sys
import os

# Windows 中文编码兼容性修复（解决 PowerShell 输出乱码）
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 优先使用本地 HF 缓存，避免联网检查更新导致启动失败/超时
# （如需更新模型，删除这两个环境变量或设置为 "0"）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "autotest"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "router"))


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python run.py test [选项]   -- 运行自动测试")
        print("  python run.py route        -- 启动路由服务")
        print()
        print("测试选项（常用）:")
        print("  --model NAME               指定待测模型名")
        print("  --modules a,b,c            选择测试模块: text,vision_recognition,image_generation,efficiency")
        print("  --num-samples N            每题集最多测试 N 题（采样）")
        print("  --api-key KEY              指定 API 密钥")
        print("  --base-url URL             指定 API 地址")
        print("  --skip-probe               跳过能力探测")
        print()
        print("示例:")
        print("  python run.py test --model qwen36-35b-a3b --modules vision_recognition")
        print("  python run.py test --modules text,efficiency --num-samples 5")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "test":
        # 移除 "test" 子命令，使 autotest/main.py 的 argparse 仅解析测试参数
        # （如 --model / --modules / --num-samples 等）
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from autotest.main import main as run_test
        run_test()

    elif command == "route":
        import uvicorn
        import threading
        import time
        import webbrowser
        import urllib.request
        from router.config import SERVICE_CONFIG

        host = SERVICE_CONFIG["host"]
        port = SERVICE_CONFIG["port"]
        # 使用 localhost 而非 0.0.0.0 作为浏览器访问地址
        dashboard_url = f"http://localhost:{port}/home"
        health_url = f"http://localhost:{port}/api/params"

        def _wait_and_open_browser():
            """轮询服务是否就绪，就绪后再打开浏览器（最多等待 120 秒）"""
            max_wait = 120  # 秒
            poll_interval = 1.0
            waited = 0.0
            while waited < max_wait:
                try:
                    # 尝试访问健康检查接口
                    req = urllib.request.Request(health_url, method="GET")
                    with urllib.request.urlopen(req, timeout=2):
                        # 服务已就绪，打开浏览器
                        webbrowser.open(dashboard_url)
                        print(f"\n[INFO] 服务已就绪，已自动打开可视化调参台: {dashboard_url}\n")
                        return
                except Exception:
                    pass
                time.sleep(poll_interval)
                waited += poll_interval
                # 每隔 10 秒打印一次等待提示
                if int(waited) % 10 == 0 and int(waited) > 0:
                    print(f"[INFO] 正在加载模型，等待服务就绪... ({int(waited)}s)", flush=True)
            print(f"\n[WARN] 等待超时（{max_wait}s），请手动访问: {dashboard_url}\n")

        threading.Thread(target=_wait_and_open_browser, daemon=True).start()

        print("=" * 60)
        print("智能路由服务启动中...")
        print(f"可视化调参台: {dashboard_url}（服务就绪后自动打开）")
        print(f"API 文档:      http://localhost:{port}/docs")
        print("=" * 60)

        uvicorn.run(
            "router.app:app",
            host=host,
            port=port,
            reload=False,
        )

    else:
        print(f"未知命令: {command}")
        print("可用命令: test, route")
        sys.exit(1)


if __name__ == "__main__":
    main()
