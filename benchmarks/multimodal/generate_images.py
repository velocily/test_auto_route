"""
============================================================================
多模态题库图片生成脚本
============================================================================
为多模态题库生成测试用图片。每张图片与题库中的 IMAGE: 引用一一对应。
使用 Pillow 绘制简单但内容明确的图表/图形/文字图片。

运行方式：
    cd benchmarks/multimodal
    python generate_images.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)

# 尝试加载字体，失败则用默认
# 注意：必须优先加载中文字体，否则中文会显示为方框（arial.ttf 不支持中文）
def _font(size=20):
    # 优先加载支持中文的字体（中英文均可正常显示）
    for path in [
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑（推荐，中英文显示效果好）
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑粗体
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/Deng.ttf",       # 等线
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驿微米黑
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux 英文兜底
        "C:/Windows/Fonts/arial.ttf",      # 英文兜底（不支持中文）
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

W, H = 400, 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
BLUE = (50, 100, 220)
GREEN = (50, 180, 80)
YELLOW = (240, 200, 50)
GRAY = (180, 180, 180)


def _new(color=WHITE):
    img = Image.new("RGB", (W, H), color)
    return img, ImageDraw.Draw(img)


def _save(img, name):
    img.save(os.path.join(OUT_DIR, name))
    print(f"  -> {name}")


# ===== ChartQA 图片 (chart_01 ~ chart_20) =====
def gen_charts():
    print("生成 ChartQA 图片...")
    # chart_01: 柱状图 - Q3最高
    img, d = _new()
    bars = [(60, 100, "Q1"), (120, 150, "Q2"), (180, 200, "Q3"), (90, 120, "Q4")]
    x = 60
    for h, top, label in bars:
        d.rectangle([x, 250 - h, x + 50, 250], fill=BLUE)
        d.text((x + 5, 260), label, fill=BLACK, font=_font(16))
        x += 70
    d.text((120, 20), "季度销售额", fill=BLACK, font=_font(20))
    _save(img, "chart_01.png")

    # chart_02: 折线图 - 增长率20%
    img, d = _new()
    points = [(60, 220), (130, 180), (200, 140), (270, 100), (340, 60)]
    d.line(points, fill=RED, width=3)
    for p in points:
        d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=RED)
    d.text((120, 20), "增长率 20%", fill=BLACK, font=_font(20))
    _save(img, "chart_02.png")

    # chart_03: 柱状图 - 产品B最低
    img, d = _new()
    bars = [(180, "A"), (50, "B"), (120, "C"), (100, "D")]
    x = 60
    for h, label in bars:
        d.rectangle([x, 250 - h, x + 50, 250], fill=GREEN)
        d.text((x + 5, 260), label, fill=BLACK, font=_font(16))
        x += 70
    d.text((100, 20), "产品线利润", fill=BLACK, font=_font(20))
    _save(img, "chart_03.png")

    # chart_04: 柱状图 - 增长10000
    img, d = _new()
    d.rectangle([60, 150, 110, 250], fill=BLUE)
    d.text((65, 260), "2022", fill=BLACK, font=_font(14))
    d.rectangle([130, 100, 180, 250], fill=BLUE)
    d.text((135, 260), "2023", fill=BLACK, font=_font(14))
    d.text((80, 20), "用户增长 +10000", fill=BLACK, font=_font(18))
    _save(img, "chart_04.png")

    # chart_05: 饼图 - 类别A最大
    img, d = _new()
    d.pieslice([80, 50, 320, 290], 0, 144, fill=RED)
    d.pieslice([80, 50, 320, 290], 144, 252, fill=BLUE)
    d.pieslice([80, 50, 320, 290], 252, 360, fill=GREEN)
    d.text((150, 20), "A占比最大", fill=BLACK, font=_font(18))
    _save(img, "chart_05.png")

    # chart_06: 折线图 - 4月最低
    img, d = _new()
    months = [(60, 150), (100, 220), (140, 180), (180, 100), (220, 120), (260, 90)]
    d.line(months, fill=BLUE, width=2)
    for p in months:
        d.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=BLUE)
    d.text((100, 20), "4月最低", fill=BLACK, font=_font(18))
    _save(img, "chart_06.png")

    # chart_07: 柱状图 - 第二高250
    img, d = _new()
    bars = [300, 250, 150, 100]
    x = 60
    for h in bars:
        d.rectangle([x, 250 - h, x + 50, 250], fill=BLUE)
        x += 70
    d.text((100, 20), "第二高=250", fill=BLACK, font=_font(18))
    _save(img, "chart_07.png")

    # chart_08: 两个柱 - 总和500
    img, d = _new()
    d.rectangle([80, 100, 140, 250], fill=BLUE)
    d.text((85, 260), "产品A", fill=BLACK, font=_font(14))
    d.rectangle([200, 50, 260, 250], fill=GREEN)
    d.text((205, 260), "产品B", fill=BLACK, font=_font(14))
    d.text((100, 20), "总计=500", fill=BLACK, font=_font(18))
    _save(img, "chart_08.png")

    # chart_09: 柱状图 - 华东最高
    img, d = _new()
    bars = [(100, "华北"), (200, "华东"), (120, "华南"), (80, "西部")]
    x = 50
    for h, label in bars:
        d.rectangle([x, 250 - h, x + 50, 250], fill=BLUE)
        d.text((x, 260), label, fill=BLACK, font=_font(14))
        x += 70
    d.text((100, 20), "华东增长率最高", fill=BLACK, font=_font(16))
    _save(img, "chart_09.png")

    # chart_10: 两组柱 - Q4比Q3增长10%
    img, d = _new()
    d.rectangle([60, 120, 110, 250], fill=BLUE)
    d.text((65, 260), "Q3", fill=BLACK, font=_font(14))
    d.rectangle([140, 100, 190, 250], fill=RED)
    d.text((145, 260), "Q4", fill=BLACK, font=_font(14))
    d.text((80, 20), "Q4增长10%", fill=BLACK, font=_font(18))
    _save(img, "chart_10.png")

    # chart_11: 显示25%利润率
    img, d = _new()
    d.rectangle([50, 100, 150, 250], fill=BLUE)
    d.text((60, 260), "收入", fill=BLACK, font=_font(14))
    d.rectangle([170, 175, 270, 250], fill=GREEN)
    d.text((180, 260), "利润25%", fill=BLACK, font=_font(14))
    _save(img, "chart_11.png")

    # chart_12: 折线图 - 2023最高
    img, d = _new()
    pts = [(60, 200), (120, 170), (180, 130), (240, 80)]
    d.line(pts, fill=RED, width=3)
    for p in pts:
        d.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=RED)
    labels = ["2020", "2021", "2022", "2023"]
    for i, p in enumerate(pts):
        d.text((p[0]-15, p[1]+10), labels[i], fill=BLACK, font=_font(12))
    _save(img, "chart_12.png")

    # chart_13: 堆叠柱状图 - 蓝色=收入
    img, d = _new()
    d.rectangle([80, 100, 160, 250], fill=BLUE)
    d.rectangle([80, 50, 160, 100], fill=GREEN)
    d.text((90, 20), "蓝=收入", fill=BLACK, font=_font(16))
    _save(img, "chart_13.png")

    # chart_14: 散点图 - 正相关
    img, d = _new()
    pts = [(80, 220), (120, 190), (160, 160), (200, 130), (240, 100), (280, 70)]
    for p in pts:
        d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=BLUE)
    d.line([(60, 230), (300, 60)], fill=GRAY, width=1)
    _save(img, "chart_14.png")

    # chart_15: 饼图 - 三类总100%
    img, d = _new()
    d.pieslice([80, 50, 320, 290], 0, 120, fill=RED)
    d.pieslice([80, 50, 320, 290], 120, 240, fill=BLUE)
    d.pieslice([80, 50, 320, 290], 240, 360, fill=GREEN)
    _save(img, "chart_15.png")

    # chart_16: 折线图 - 先升后降
    img, d = _new()
    pts = [(60, 200), (130, 80), (200, 100), (270, 180), (340, 220)]
    d.line(pts, fill=BLUE, width=3)
    for p in pts:
        d.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=BLUE)
    _save(img, "chart_16.png")

    # chart_17: 柱状图 - 差值200
    img, d = _new()
    bars = [250, 150, 100, 50]
    x = 60
    for h in bars:
        d.rectangle([x, 250 - h, x + 50, 250], fill=BLUE)
        x += 70
    d.text((80, 20), "差值=200", fill=BLACK, font=_font(18))
    _save(img, "chart_17.png")

    # chart_18: 饼图 - 财务部最小
    img, d = _new()
    d.pieslice([80, 50, 320, 290], 0, 90, fill=RED)
    d.pieslice([80, 50, 320, 290], 90, 180, fill=BLUE)
    d.pieslice([80, 50, 320, 290], 180, 330, fill=GREEN)
    d.pieslice([80, 50, 320, 290], 330, 360, fill=YELLOW)
    d.text((100, 20), "财务部最小", fill=BLACK, font=_font(16))
    _save(img, "chart_18.png")

    # chart_19: 柱状图 - 同比12.5%
    img, d = _new()
    d.rectangle([80, 150, 140, 250], fill=BLUE)
    d.text((85, 260), "去年", fill=BLACK, font=_font(14))
    d.rectangle([180, 100, 240, 250], fill=RED)
    d.text((185, 260), "今年", fill=BLACK, font=_font(14))
    d.text((100, 20), "同比增长12.5%", fill=BLACK, font=_font(16))
    _save(img, "chart_19.png")

    # chart_20: 折线图 - 晚上18-21最高
    img, d = _new()
    pts = [(60, 200), (120, 180), (180, 150), (240, 60), (300, 120)]
    d.line(pts, fill=RED, width=3)
    labels = ["6-9", "12-14", "15-17", "18-21", "0-3"]
    for i, p in enumerate(pts):
        d.text((p[0]-15, p[1]+10), labels[i], fill=BLACK, font=_font(10))
    _save(img, "chart_20.png")


# ===== TextVQA 图片 (text_01 ~ text_20) =====
def gen_texts():
    print("生成 TextVQA 图片...")
    texts = [
        ("text_01.png", "中山路", "路牌"),
        ("text_02.png", "¥256.80", "发票"),
        ("text_03.png", "营业时间\n10:00-22:00", "店铺"),
        ("text_04.png", "作者：李四", "书籍"),
        ("text_05.png", "生产日期：2023-06-10", "标签"),
        ("text_06.png", "活动日期：5月1日", "海报"),
        ("text_07.png", "电话：13800138000", "广告"),
        ("text_08.png", "佛跳墙 ¥298", "菜单"),
        ("text_09.png", "www.example.com", "网页"),
        ("text_10.png", "目的地：上海", "机票"),
        ("text_11.png", "info@company.com", "名片"),
        ("text_12.png", "用法用量：每日两次\n每次1片", "说明书"),
        ("text_13.png", "Wi-Fi密码：12345678", "提示牌"),
        ("text_14.png", "科技有限公司", "名片"),
        ("text_15.png", "6901234567890", "条形码"),
        ("text_16.png", "截止日期：2024年3月1日", "公告"),
        ("text_17.png", "房间号：202", "门牌"),
        ("text_18.png", "总人数：250", "表格"),
        ("text_19.png", "尾号：5678", "银行卡"),
        ("text_20.png", "保修期：2年", "说明书"),
    ]
    for fname, text, label in texts:
        img, d = _new()
        d.rectangle([20, 20, 380, 280], outline=BLACK, width=2)
        d.text((100, 120), text, fill=BLACK, font=_font(24))
        d.text((100, 250), f"[{label}]", fill=GRAY, font=_font(14))
        _save(img, fname)


# ===== MathVista 图片 (math_01 ~ math_20) =====
def gen_maths():
    print("生成 MathVista 图片...")
    # math_01: 三角形 底6高4 面积12
    img, d = _new()
    d.polygon([(200, 50), (80, 250), (320, 250)], outline=BLACK, width=2, fill=(200, 220, 255))
    d.text((180, 260), "底=6, 高=4", fill=BLACK, font=_font(16))
    _save(img, "math_01.png")

    # math_02: 圆 半径3
    img, d = _new()
    d.ellipse([100, 50, 300, 250], outline=BLACK, width=2, fill=(200, 220, 255))
    d.line([200, 150, 300, 150], fill=RED, width=2)
    d.text((220, 130), "r=3", fill=RED, font=_font(16))
    _save(img, "math_02.png")

    # math_03: 长方形 8x5
    img, d = _new()
    d.rectangle([80, 80, 320, 230], outline=BLACK, width=2, fill=(200, 220, 255))
    d.text((180, 250), "长=8 宽=5", fill=BLACK, font=_font(16))
    _save(img, "math_03.png")

    # math_04: 数轴 A=2
    img, d = _new()
    d.line([50, 150, 350, 150], fill=BLACK, width=2)
    for i, x in enumerate(range(50, 351, 50)):
        d.line([x, 145, x, 155], fill=BLACK, width=1)
        d.text((x - 5, 160), str(i - 3), fill=BLACK, font=_font(12))
    d.ellipse([195, 140, 205, 160], fill=RED)
    d.text((190, 120), "A", fill=RED, font=_font(16))
    _save(img, "math_04.png")

    # math_05: 梯形 上4下6高5
    img, d = _new()
    d.polygon([(120, 80), (280, 80), (320, 230), (80, 230)], outline=BLACK, width=2, fill=(200, 220, 255))
    d.text((160, 250), "上=4 下=6 高=5", fill=BLACK, font=_font(14))
    _save(img, "math_05.png")

    # math_06: 方程组 x=1,y=2
    img, d = _new()
    d.text((80, 100), "x + y = 3", fill=BLACK, font=_font(24))
    d.text((80, 150), "x - y = -1", fill=BLACK, font=_font(24))
    _save(img, "math_06.png")

    # math_07: 正方体 边长3
    img, d = _new()
    d.rectangle([100, 100, 250, 250], outline=BLACK, width=2, fill=(200, 220, 255))
    d.polygon([(100, 100), (150, 50), (300, 50), (250, 100)], outline=BLACK, fill=(220, 240, 255))
    d.polygon([(250, 100), (300, 50), (300, 200), (250, 250)], outline=BLACK, fill=(180, 200, 240))
    d.text((160, 270), "边长=3", fill=BLACK, font=_font(16))
    _save(img, "math_07.png")

    # math_08: y=x² 图像
    img, d = _new()
    d.line([200, 50, 200, 250], fill=GRAY, width=1)
    d.line([50, 150, 350, 150], fill=GRAY, width=1)
    pts = [(200, 150), (230, 120), (260, 90), (290, 60)]
    d.line(pts, fill=RED, width=3)
    pts2 = [(200, 150), (170, 120), (140, 90), (110, 60)]
    d.line(pts2, fill=RED, width=3)
    d.text((250, 60), "x=2", fill=BLACK, font=_font(14))
    _save(img, "math_08.png")

    # math_09: 概率树 P(A∩B)=0.12
    img, d = _new()
    d.line([100, 150, 180, 100], fill=BLACK, width=2)
    d.line([100, 150, 180, 200], fill=BLACK, width=2)
    d.line([180, 100, 280, 80], fill=BLACK, width=2)
    d.text((120, 110), "0.4", fill=BLACK, font=_font(14))
    d.text((210, 80), "0.3", fill=BLACK, font=_font(14))
    d.text((250, 70), "0.12", fill=RED, font=_font(16))
    _save(img, "math_09.png")

    # math_10: 扇形 90° r=4
    img, d = _new()
    d.pieslice([100, 50, 300, 250], 0, 90, fill=(200, 220, 255), outline=BLACK, width=2)
    d.text((140, 270), "90° r=4", fill=BLACK, font=_font(16))
    _save(img, "math_10.png")

    # math_11: 直角三角形 3-4-5
    img, d = _new()
    d.polygon([(80, 250), (280, 250), (80, 50)], outline=BLACK, width=2, fill=(200, 220, 255))
    d.text((150, 260), "3", fill=BLACK, font=_font(14))
    d.text((60, 150), "4", fill=BLACK, font=_font(14))
    d.text((180, 140), "5", fill=RED, font=_font(16))
    _save(img, "math_11.png")

    # math_12: 柱状图 平均20
    img, d = _new()
    bars = [10, 20, 30, 20]
    x = 60
    for h in bars:
        d.rectangle([x, 250 - h * 5, x + 50, 250], fill=BLUE)
        x += 70
    d.text((100, 20), "平均=20", fill=BLACK, font=_font(18))
    _save(img, "math_12.png")

    # math_13: 长方体 2x3x4 表面积52
    img, d = _new()
    d.rectangle([100, 100, 220, 220], outline=BLACK, width=2, fill=(200, 220, 255))
    d.polygon([(100, 100), (140, 60), (260, 60), (220, 100)], outline=BLACK, fill=(220, 240, 255))
    d.polygon([(220, 100), (260, 60), (260, 180), (220, 220)], outline=BLACK, fill=(180, 200, 240))
    d.text((130, 240), "2×3×4", fill=BLACK, font=_font(16))
    _save(img, "math_13.png")

    # math_14: 坐标系 第二象限(-1,1)
    img, d = _new()
    d.line([200, 50, 200, 250], fill=GRAY, width=1)
    d.line([50, 150, 350, 150], fill=GRAY, width=1)
    d.ellipse([140, 80, 160, 100], fill=RED)
    d.text((130, 60), "(-1,1)", fill=RED, font=_font(14))
    d.text((120, 20), "第二象限", fill=BLACK, font=_font(16))
    _save(img, "math_14.png")

    # math_15: 75% = 3/4
    img, d = _new()
    d.rectangle([50, 50, 350, 250], outline=BLACK, width=2)
    d.rectangle([50, 50, 275, 250], fill=BLUE)
    d.text((150, 130), "75%", fill=WHITE, font=_font(28))
    _save(img, "math_15.png")

    # math_16: 平行四边形 底10高6
    img, d = _new()
    d.polygon([(80, 200), (180, 80), (320, 80), (220, 200)], outline=BLACK, width=2, fill=(200, 220, 255))
    d.text((150, 220), "底=10 高=6", fill=BLACK, font=_font(16))
    _save(img, "math_16.png")

    # math_17: 数列 2,4,8,16,32
    img, d = _new()
    d.text((60, 120), "2, 4, 8, 16, ?", fill=BLACK, font=_font(28))
    _save(img, "math_17.png")

    # math_18: 圆锥 r=3 h=4
    img, d = _new()
    d.polygon([(200, 50), (100, 250), (300, 250)], outline=BLACK, width=2, fill=(200, 220, 255))
    d.ellipse([100, 230, 300, 270], outline=BLACK, width=1)
    d.text((150, 280), "r=3 h=4", fill=BLACK, font=_font(16))
    _save(img, "math_18.png")

    # math_19: 韦恩图 A∪B=7
    img, d = _new()
    d.ellipse([80, 80, 220, 220], outline=BLUE, width=2, fill=(200, 220, 255))
    d.ellipse([180, 80, 320, 220], outline=RED, width=2, fill=(255, 200, 200))
    d.text((110, 130), "A=4", fill=BLUE, font=_font(16))
    d.text((250, 130), "B=4", fill=RED, font=_font(16))
    d.text((185, 130), "1", fill=BLACK, font=_font(14))
    d.text((150, 20), "A∪B=7", fill=BLACK, font=_font(18))
    _save(img, "math_19.png")

    # math_20: 折线图 中位数15
    img, d = _new()
    pts = [(60, 200), (120, 150), (180, 100), (240, 150), (300, 200)]
    d.line(pts, fill=BLUE, width=3)
    for p in pts:
        d.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=BLUE)
    d.text((100, 20), "中位数=15", fill=BLACK, font=_font(18))
    _save(img, "math_20.png")


# ===== VQA 图片 (vqa_01 ~ vqa_30) =====
def gen_vqa():
    print("生成 VQA 图片...")
    # vqa_01: 3个红色圆形
    img, d = _new()
    for i in range(3):
        d.ellipse([60 + i * 100, 100, 120 + i * 100, 160], fill=RED)
    _save(img, "vqa_01.png")

    # vqa_02: 最大形状蓝色
    img, d = _new()
    d.rectangle([50, 50, 250, 250], fill=BLUE)
    d.ellipse([150, 100, 210, 160], fill=RED)
    _save(img, "vqa_02.png")

    # vqa_03: 4个三角形
    img, d = _new()
    for i in range(4):
        x = 50 + i * 80
        d.polygon([(x, 200), (x + 40, 100), (x + 80, 200)], fill=GREEN)
    _save(img, "vqa_03.png")

    # vqa_04: 红色正方形
    img, d = _new()
    d.rectangle([100, 50, 300, 250], fill=RED)
    _save(img, "vqa_04.png")

    # vqa_05: 6个图形
    img, d = _new()
    d.ellipse([50, 50, 110, 110], fill=RED)
    d.rectangle([150, 50, 210, 110], fill=BLUE)
    d.polygon([(250, 110), (290, 50), (330, 110)], fill=GREEN)
    d.ellipse([50, 150, 110, 210], fill=YELLOW)
    d.rectangle([150, 150, 210, 210], fill=GREEN)
    d.polygon([(250, 210), (290, 150), (330, 210)], fill=RED)
    _save(img, "vqa_05.png")

    # vqa_06: 三角形在最上面
    img, d = _new()
    d.polygon([(150, 30), (200, 100), (100, 100)], fill=RED)
    d.rectangle([100, 110, 200, 200], fill=BLUE)
    d.ellipse([100, 210, 200, 280], fill=GREEN)
    _save(img, "vqa_06.png")

    # vqa_07: 蓝色圆形
    img, d = _new()
    d.ellipse([100, 50, 300, 250], fill=BLUE)
    _save(img, "vqa_07.png")

    # vqa_08: 2个绿色图形
    img, d = _new()
    d.ellipse([50, 50, 150, 150], fill=GREEN)
    d.rectangle([200, 50, 300, 150], fill=GREEN)
    d.ellipse([50, 180, 150, 280], fill=RED)
    _save(img, "vqa_08.png")

    # vqa_09: 最大数字9
    img, d = _new()
    d.text((60, 100), "3 5 9 7", fill=BLACK, font=_font(36))
    _save(img, "vqa_09.png")

    # vqa_10: 箭头向右
    img, d = _new()
    d.line([50, 150, 300, 150], fill=BLACK, width=5)
    d.polygon([(300, 130), (350, 150), (300, 170)], fill=BLACK)
    _save(img, "vqa_10.png")

    # vqa_11: 4条线段
    img, d = _new()
    d.line([50, 50, 350, 50], fill=BLACK, width=3)
    d.line([50, 150, 350, 150], fill=BLACK, width=3)
    d.line([50, 250, 350, 250], fill=BLACK, width=3)
    d.line([50, 50, 50, 250], fill=BLACK, width=3)
    _save(img, "vqa_11.png")

    # vqa_12: 1/4阴影
    img, d = _new()
    d.rectangle([50, 50, 350, 250], outline=BLACK, width=2)
    d.rectangle([50, 50, 200, 150], fill=GRAY)
    _save(img, "vqa_12.png")

    # vqa_13: 长方形面积最大
    img, d = _new()
    d.ellipse([50, 50, 120, 120], fill=BLUE)
    d.rectangle([150, 50, 220, 120], fill=RED)
    d.polygon([(250, 120), (300, 50), (350, 120)], fill=GREEN)
    d.rectangle([50, 150, 350, 250], fill=YELLOW)
    _save(img, "vqa_13.png")

    # vqa_14: 轴对称
    img, d = _new()
    d.line([200, 50, 200, 250], fill=GRAY, width=1)
    d.polygon([(100, 150), (200, 50), (200, 250)], fill=BLUE)
    d.polygon([(300, 150), (200, 50), (200, 250)], fill=BLUE)
    _save(img, "vqa_14.png")

    # vqa_15: 4个直角
    img, d = _new()
    d.rectangle([50, 50, 350, 250], outline=BLACK, width=2)
    _save(img, "vqa_15.png")

    # vqa_16: 数列 2,4,6,8,10
    img, d = _new()
    d.text((60, 120), "2 4 6 8 ?", fill=BLACK, font=_font(36))
    _save(img, "vqa_16.png")

    # vqa_17: 旋转变换
    img, d = _new()
    d.rectangle([50, 50, 150, 150], outline=BLUE, width=2)
    d.rectangle([200, 80, 330, 210], outline=RED, width=2)
    d.arc([150, 100, 250, 200], 0, 90, fill=GREEN, width=2)
    _save(img, "vqa_17.png")

    # vqa_18: 全等三角形
    img, d = _new()
    d.polygon([(50, 250), (150, 250), (50, 100)], fill=BLUE)
    d.polygon([(200, 250), (300, 250), (200, 100)], fill=BLUE)
    _save(img, "vqa_18.png")

    # vqa_19: 7个黑格
    img, d = _new()
    for i in range(5):
        d.rectangle([50 + i * 60, 50, 110 + i * 60, 110], outline=BLACK, fill=WHITE)
    blacks = [0, 1, 3, 4, 5, 6, 7]
    grid = [50, 110, 170, 230, 290]
    for r in range(3):
        for c in range(5):
            idx = r * 5 + c
            x = grid[c]
            y = 50 + r * 60
            d.rectangle([x, y, x + 60, y + 60], outline=BLACK, fill=(50, 50, 50) if idx in blacks else WHITE)
    _save(img, "vqa_19.png")

    # vqa_20: 时钟 3:30
    img, d = _new()
    d.ellipse([50, 50, 350, 350], outline=BLACK, width=3)
    cx, cy = 200, 200
    d.line([cx, cy, cx, cy - 80], fill=BLACK, width=4)  # 时针指3
    d.line([cx, cy, cx + 60, cy], fill=BLACK, width=3)  # 分针指6(30分)
    _save(img, "vqa_20.png")

    # vqa_21: 3种颜色
    img, d = _new()
    d.ellipse([50, 100, 150, 200], fill=RED)
    d.ellipse([150, 100, 250, 200], fill=BLUE)
    d.ellipse([250, 100, 350, 200], fill=GREEN)
    _save(img, "vqa_21.png")

    # vqa_22: 金属质感
    img, d = _new((200, 200, 210))
    d.rectangle([100, 50, 300, 250], fill=(180, 180, 190))
    d.rectangle([100, 50, 300, 60], fill=(220, 220, 230))
    d.text((150, 130), "METAL", fill=(100, 100, 110), font=_font(24))
    _save(img, "vqa_22.png")

    # vqa_23: 冬季场景
    img, d = _new((230, 240, 255))
    d.ellipse([250, 30, 330, 110], fill=(255, 240, 100))  # 太阳
    for i in range(5):
        d.ellipse([20 + i * 70, 200, 60 + i * 70, 240], fill=WHITE)
    d.text((150, 130), "WINTER", fill=BLUE, font=_font(24))
    _save(img, "vqa_23.png")

    # vqa_24: 4个交点
    img, d = _new()
    d.line([50, 50, 350, 250], fill=BLACK, width=2)
    d.line([50, 250, 350, 50], fill=BLACK, width=2)
    d.line([50, 150, 350, 150], fill=BLACK, width=2)
    d.line([200, 50, 200, 250], fill=BLACK, width=2)
    _save(img, "vqa_24.png")

    # vqa_25: 晴天
    img, d = _new((135, 206, 235))
    d.ellipse([120, 50, 280, 210], fill=(255, 230, 50))
    for i in range(8):
        import math
        angle = i * 45
        x1 = 200 + 100 * math.cos(math.radians(angle))
        y1 = 130 + 100 * math.sin(math.radians(angle))
        x2 = 200 + 130 * math.cos(math.radians(angle))
        y2 = 130 + 130 * math.sin(math.radians(angle))
        d.line([x1, y1, x2, y2], fill=(255, 230, 50), width=3)
    _save(img, "vqa_25.png")

    # vqa_26: 最大圆半径4
    img, d = _new()
    d.ellipse([50, 50, 250, 250], outline=BLUE, width=3)
    d.ellipse([100, 100, 200, 200], outline=RED, width=2)
    d.text((100, 260), "r=4", fill=BLACK, font=_font(16))
    _save(img, "vqa_26.png")

    # vqa_27: 圆柱
    img, d = _new()
    d.ellipse([100, 50, 300, 100], fill=(200, 220, 255), outline=BLACK)
    d.rectangle([100, 75, 300, 250], fill=(200, 220, 255), outline=BLACK)
    d.ellipse([100, 225, 300, 275], fill=(180, 200, 240), outline=BLACK)
    _save(img, "vqa_27.png")

    # vqa_28: 阴影三角形
    img, d = _new()
    d.rectangle([50, 50, 350, 250], outline=BLACK, width=2)
    d.polygon([(100, 200), (200, 100), (300, 200)], fill=GRAY)
    _save(img, "vqa_28.png")

    # vqa_29: 4层
    img, d = _new()
    for i in range(4):
        y0 = 250 - i * 50
        y1 = y0 - 45
        d.rectangle([100, y1, 300, y0], fill=BLUE if i % 2 == 0 else GREEN)
    _save(img, "vqa_29.png")

    # vqa_30: 偶数个物体
    img, d = _new()
    for i in range(6):
        d.ellipse([50 + i * 55, 120, 100 + i * 55, 170], fill=RED)
    _save(img, "vqa_30.png")


# ===== MMMU 图片 (mmmu_01 ~ mmmu_30) =====
def gen_mmmu():
    print("生成 MMMU 图片...")
    # mmmu_01: 电阻符号
    img, d = _new()
    d.line([50, 150, 120, 150], fill=BLACK, width=2)
    d.rectangle([120, 130, 280, 170], fill=WHITE, outline=BLACK, width=2)
    d.line([280, 150, 350, 150], fill=BLACK, width=2)
    d.text((170, 100), "R", fill=BLACK, font=_font(20))
    _save(img, "mmmu_01.png")

    # mmmu_02: 植物细胞
    img, d = _new()
    d.ellipse([50, 50, 350, 250], outline=BLACK, width=2, fill=(220, 255, 220))
    d.ellipse([170, 120, 230, 180], fill=BLUE)
    d.text((190, 140), "①", fill=WHITE, font=_font(16))
    _save(img, "mmmu_02.png")

    # mmmu_03: 化学方程式
    img, d = _new()
    d.text((60, 120), "2H₂ + O₂ → 2H₂O", fill=BLACK, font=_font(24))
    _save(img, "mmmu_03.png")

    # mmmu_04: 力学示意图
    img, d = _new()
    d.rectangle([150, 150, 250, 250], fill=GRAY)
    d.line([200, 150, 200, 50], fill=RED, width=3)
    d.polygon([(190, 60), (200, 40), (210, 60)], fill=RED)
    d.text((220, 80), "F", fill=RED, font=_font(20))
    _save(img, "mmmu_04.png")

    # mmmu_05: DNA双螺旋
    img, d = _new()
    for y in range(50, 250, 20):
        offset = 50 * (1 if (y // 20) % 2 == 0 else -1)
        d.line([200 - offset, y, 200 + offset, y + 10], fill=BLUE, width=2)
    _save(img, "mmmu_05.png")

    # mmmu_06: 地形图
    img, d = _new((200, 180, 150))
    for r in range(50, 0, -10):
        d.ellipse([200 - r, 150 - r, 200 + r, 150 + r], outline=(150, 100, 50))
    _save(img, "mmmu_06.png")

    # mmmu_07: 供需曲线
    img, d = _new()
    d.line([50, 250, 350, 50], fill=BLUE, width=2)
    d.line([50, 50, 350, 250], fill=RED, width=2)
    d.text((60, 60), "S", fill=BLUE, font=_font(18))
    d.text((330, 60), "D", fill=RED, font=_font(18))
    _save(img, "mmmu_07.png")

    # mmmu_08: 艺术作品
    img, d = _new((240, 220, 180))
    d.ellipse([120, 80, 280, 240], fill=(220, 180, 140))
    d.text((130, 20), "文艺复兴风格", fill=BLACK, font=_font(16))
    _save(img, "mmmu_08.png")

    # mmmu_09: 电流表
    img, d = _new()
    d.ellipse([100, 50, 300, 250], outline=BLACK, width=3, fill=WHITE)
    d.arc([120, 70, 280, 230], 180, 270, fill=RED, width=3)
    d.text((180, 130), "A", fill=BLACK, font=_font(28))
    _save(img, "mmmu_09.png")

    # mmmu_10: 生物分类
    img, d = _new()
    d.text((100, 50), "动物界", fill=BLACK, font=_font(20))
    d.line([200, 80, 200, 120], fill=BLACK, width=2)
    d.text((100, 130), "脊索动物门", fill=BLACK, font=_font(16))
    d.line([200, 150, 200, 180], fill=BLACK, width=2)
    d.text((100, 190), "哺乳纲", fill=BLACK, font=_font(16))
    d.line([200, 210, 200, 240], fill=BLACK, width=2)
    d.text((100, 250), "人类", fill=RED, font=_font(16))
    _save(img, "mmmu_10.png")

    # mmmu_11: 几何证明 角A=60°
    img, d = _new()
    d.polygon([(100, 250), (300, 250), (200, 80)], outline=BLACK, width=2, fill=(220, 240, 255))
    d.text((180, 270), "A", fill=RED, font=_font(18))
    d.text((100, 20), "∠A=60°", fill=BLACK, font=_font(18))
    _save(img, "mmmu_11.png")

    # mmmu_12: 甲烷分子
    img, d = _new()
    d.ellipse([180, 130, 220, 170], fill=BLACK)
    for angle in [0, 90, 180, 270]:
        import math
        x = 200 + 60 * math.cos(math.radians(angle))
        y = 150 + 60 * math.sin(math.radians(angle))
        d.ellipse([x-10, y-10, x+10, y+10], fill=WHITE, outline=BLACK, width=2)
        d.line([200, 150, x, y], fill=BLACK, width=2)
    _save(img, "mmmu_12.png")

    # mmmu_13: 历史时间线
    img, d = _new()
    d.line([50, 150, 350, 150], fill=BLACK, width=2)
    d.ellipse([80, 140, 100, 160], fill=RED)
    d.text((70, 170), "15世纪", fill=BLACK, font=_font(12))
    d.ellipse([200, 140, 220, 160], fill=BLUE)
    d.text((190, 170), "16世纪", fill=BLACK, font=_font(12))
    d.ellipse([320, 140, 340, 160], fill=GREEN)
    d.text((310, 170), "17世纪", fill=BLACK, font=_font(12))
    _save(img, "mmmu_13.png")

    # mmmu_14: 循环结构流程图
    img, d = _new()
    d.rectangle([120, 50, 280, 100], fill=(200, 220, 255), outline=BLACK)
    d.text((150, 65), "开始", fill=BLACK, font=_font(16))
    d.polygon([(200, 100), (250, 150), (200, 200), (150, 150)], fill=(255, 230, 200), outline=BLACK)
    d.text((175, 140), "条件?", fill=BLACK, font=_font(14))
    d.line([200, 200, 200, 230], fill=BLACK, width=2)
    d.line([200, 230, 120, 230], fill=BLACK, width=2)
    d.line([120, 230, 120, 75], fill=BLACK, width=2)
    d.line([120, 75, 120, 75], fill=BLACK, width=2)
    _save(img, "mmmu_14.png")

    # mmmu_15: 板块张裂
    img, d = _new()
    d.polygon([50, 100, 180, 100, 180, 250, 50, 250], fill=(200, 180, 140))
    d.polygon([220, 100, 350, 100, 350, 250, 220, 250], fill=(200, 180, 140))
    d.line([180, 100, 220, 80], fill=RED, width=3)
    d.line([180, 250, 220, 270], fill=RED, width=3)
    d.text((150, 50), "←张裂→", fill=RED, font=_font(18))
    _save(img, "mmmu_15.png")

    # mmmu_16: 饼图
    img, d = _new()
    d.pieslice([80, 50, 320, 290], 0, 120, fill=RED)
    d.pieslice([80, 50, 320, 290], 120, 240, fill=BLUE)
    d.pieslice([80, 50, 320, 290], 240, 360, fill=GREEN)
    _save(img, "mmmu_16.png")

    # mmmu_17: 光的折射
    img, d = _new()
    d.line([50, 50, 200, 150], fill=RED, width=2)
    d.line([200, 150, 300, 200], fill=RED, width=2)
    d.line([50, 150, 350, 150], fill=BLACK, width=1)
    d.arc([180, 130, 220, 170], 0, 90, fill=BLUE, width=2)
    d.text((220, 160), "30°", fill=BLUE, font=_font(14))
    _save(img, "mmmu_17.png")

    # mmmu_18: 化学装置 向上排空气法
    img, d = _new()
    d.rectangle([80, 100, 120, 250], fill=(200, 220, 255), outline=BLACK)
    d.line([100, 100, 100, 50], fill=BLACK, width=2)
    d.line([100, 50, 250, 50], fill=BLACK, width=2)
    d.line([250, 50, 250, 200], fill=BLACK, width=2)
    d.rectangle([220, 200, 280, 250], fill=(220, 240, 255), outline=BLACK)
    d.text((100, 20), "↑向上排空气", fill=BLACK, font=_font(14))
    _save(img, "mmmu_18.png")

    # mmmu_19: 食物链
    img, d = _new()
    d.text((60, 120), "草", fill=GREEN, font=_font(20))
    d.text((140, 120), "→", fill=BLACK, font=_font(20))
    d.text((180, 120), "兔", fill=BLACK, font=_font(20))
    d.text((240, 120), "→", fill=BLACK, font=_font(20))
    d.text((280, 120), "鹰", fill=BLACK, font=_font(20))
    _save(img, "mmmu_19.png")

    # mmmu_20: 二次函数
    img, d = _new()
    d.line([200, 50, 200, 250], fill=GRAY, width=1)
    d.line([50, 150, 350, 150], fill=GRAY, width=1)
    pts = [(100, 50), (150, 100), (200, 150), (250, 100), (300, 50)]
    d.line(pts, fill=RED, width=3)
    d.text((250, 30), "y=x²", fill=BLACK, font=_font(16))
    _save(img, "mmmu_20.png")

    # mmmu_21: 并联电阻
    img, d = _new()
    d.line([50, 150, 100, 150], fill=BLACK, width=2)
    d.line([100, 150, 100, 100], fill=BLACK, width=2)
    d.rectangle([100, 80, 180, 120], fill=WHITE, outline=BLACK)
    d.text((120, 90), "R1=2", fill=BLACK, font=_font(14))
    d.line([180, 100, 250, 100], fill=BLACK, width=2)
    d.line([100, 150, 100, 200], fill=BLACK, width=2)
    d.rectangle([100, 180, 180, 220], fill=WHITE, outline=BLACK)
    d.text((120, 190), "R2=3", fill=BLACK, font=_font(14))
    d.line([180, 200, 250, 200], fill=BLACK, width=2)
    d.line([250, 100, 250, 200], fill=BLACK, width=2)
    d.line([250, 150, 350, 150], fill=BLACK, width=2)
    _save(img, "mmmu_21.png")

    # mmmu_22: 行星椭圆轨道
    img, d = _new()
    d.ellipse([50, 80, 350, 220], outline=BLUE, width=2)
    d.ellipse([190, 140, 210, 160], fill=YELLOW)
    d.ellipse([320, 140, 340, 160], fill=RED)
    _save(img, "mmmu_22.png")

    # mmmu_23: 人体呼吸系统
    img, d = _new()
    d.ellipse([150, 50, 250, 150], fill=(255, 200, 200), outline=BLACK)
    d.text((170, 90), "肺", fill=BLACK, font=_font(20))
    d.line([200, 150, 200, 250], fill=BLACK, width=3)
    _save(img, "mmmu_23.png")

    # mmmu_24: 计算机架构
    img, d = _new()
    d.rectangle([100, 50, 300, 100], fill=(200, 220, 255), outline=BLACK)
    d.text((130, 65), "CPU", fill=BLACK, font=_font(18))
    d.rectangle([100, 120, 180, 170], fill=(255, 230, 200), outline=BLACK)
    d.text((110, 135), "运算器", fill=BLACK, font=_font(12))
    d.rectangle([190, 120, 270, 170], fill=(255, 230, 200), outline=BLACK)
    d.text((200, 135), "控制器", fill=BLACK, font=_font(12))
    d.rectangle([100, 190, 180, 240], fill=(200, 255, 200), outline=BLACK)
    d.text((110, 205), "存储器", fill=BLACK, font=_font(12))
    _save(img, "mmmu_24.png")

    # mmmu_25: 元素周期表第1周期
    img, d = _new()
    d.rectangle([50, 100, 150, 200], fill=(255, 255, 200), outline=BLACK, width=2)
    d.text((75, 120), "H", fill=BLACK, font=_font(28))
    d.text((70, 170), "氢", fill=BLACK, font=_font(14))
    d.rectangle([200, 100, 300, 200], fill=(255, 200, 200), outline=BLACK, width=2)
    d.text((225, 120), "He", fill=BLACK, font=_font(28))
    d.text((220, 170), "氦", fill=BLACK, font=_font(14))
    _save(img, "mmmu_25.png")

    # mmmu_26: 力的符号
    img, d = _new()
    d.text((100, 100), "F = mg", fill=BLACK, font=_font(28))
    d.text((100, 150), "单位：牛顿(N)", fill=RED, font=_font(18))
    _save(img, "mmmu_26.png")

    # mmmu_27: 生态系统
    img, d = _new()
    d.rectangle([50, 200, 350, 250], fill=GREEN)
    d.text((60, 215), "草(生产者)", fill=WHITE, font=_font(14))
    d.ellipse([80, 100, 140, 160], fill=(200, 180, 140))
    d.text((85, 120), "兔", fill=BLACK, font=_font(14))
    d.ellipse([250, 80, 330, 160], fill=GRAY)
    d.text((275, 110), "狼", fill=WHITE, font=_font(14))
    _save(img, "mmmu_27.png")

    # mmmu_28: 圆周长公式
    img, d = _new()
    d.ellipse([100, 50, 300, 250], outline=BLUE, width=3)
    d.line([200, 150, 300, 150], fill=RED, width=2)
    d.text((220, 130), "r", fill=RED, font=_font(16))
    d.text((100, 270), "C = 2πr", fill=BLACK, font=_font(20))
    _save(img, "mmmu_28.png")

    # mmmu_29: 赤道纬度
    img, d = _new()
    d.ellipse([100, 50, 300, 250], outline=BLACK, width=2, fill=(135, 206, 235))
    d.line([100, 150, 300, 150], fill=RED, width=3)
    d.text((180, 160), "0°", fill=RED, font=_font(16))
    d.text((180, 20), "赤道", fill=BLACK, font=_font(16))
    _save(img, "mmmu_29.png")

    # mmmu_30: 功的符号
    img, d = _new()
    d.text((80, 100), "W = Fs", fill=BLACK, font=_font(28))
    d.text((80, 150), "W = 功", fill=RED, font=_font(18))
    _save(img, "mmmu_30.png")


if __name__ == "__main__":
    print("=" * 60)
    print("多模态题库图片生成脚本")
    print("=" * 60)
    gen_charts()
    gen_texts()
    gen_maths()
    gen_vqa()
    gen_mmmu()
    print("=" * 60)
    print(f"所有图片已生成到: {OUT_DIR}")
    print(f"共生成 {len(os.listdir(OUT_DIR))} 个文件")
    print("=" * 60)
