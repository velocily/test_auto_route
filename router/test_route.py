import sys
import traceback
sys.path.insert(0, r'd:\研究生\实习\epic智能科技有限公司工元智研lab\test_auto_route\router')

try:
    from router_engine import route_messages
    result = route_messages([{'role': 'user', 'content': 'hello'}])
    print("Result:", result)
    print("selected_model:", result.get("selected_model"))
except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()
