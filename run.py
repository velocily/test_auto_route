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

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "autotest"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "router"))


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python run.py test   -- 运行自动测试")
        print("  python run.py route  -- 启动路由服务")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "test":
        from autotest.main import main as run_test
        run_test()

    elif command == "route":
        import uvicorn
        from router.config import SERVICE_CONFIG
        uvicorn.run(
            "router.app:app",
            host=SERVICE_CONFIG["host"],
            port=SERVICE_CONFIG["port"],
            reload=False,
        )

    else:
        print(f"未知命令: {command}")
        print("可用命令: test, route")
        sys.exit(1)


if __name__ == "__main__":
    main()
