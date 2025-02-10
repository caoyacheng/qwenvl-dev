import cv2
import base64
import os
import time
import threading
import json
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化 OpenAI 客户端以连接 DashScope (Qwen-VL-Max)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Flask 应用初始化
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 禁用 ASCII 转义，确保返回的 JSON 包含可读的中文


# 将捕获的图像发送到 Qwen-VL-Max 进行分析
def analyze_image_with_qwenvl(image):
    if image is None:
        return {"error": "No image to analyze."}

    # 动态生成当前时间戳
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    _, img_buffer = cv2.imencode('.jpg', image)
    image_data = base64.b64encode(img_buffer).decode('utf-8')
    message_content = [
        {
            "type": "text",
            "text": f"""
你的任务是根据输入的图像生成一份详细的巡检报告。报告需要以结构化的方式呈现，包括以下内容：

1. **标题**: 巡检报告标题，例如“视频监控巡检报告”。
2. **时间**: {timestamp} （当前时间已自动填充）。
3. **检查内容**: 描述当前视频或图像中观察到的主要内容，包括环境、人物、物品等。
4. **发现问题**: 列出发现的异常或潜在问题，例如危险行为（抽烟、拿刀）、设备故障、环境隐患等。
   - 如果发现危险行为，请特别标注为：⚠️⚠️存在危险行为，危险行为是[具体描述]。
5. **建议措施**: 针对发现的问题，提出具体的改进建议或处理措施。
6. **备注**: 其他需要注意的信息或补充说明。

请严格按照以下格式输出：
----------------------------------------
**巡检报告**

- **标题**: [填写标题]
- **时间**: {timestamp}
- **检查内容**: [填写检查内容]
- **发现问题**: [填写发现的问题]
- **建议措施**: [填写建议措施]
- **备注**: [填写备注]
----------------------------------------

请根据输入的图像内容生成完整的巡检报告。
"""
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        }
    ]
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[{"role": "user", "content": message_content}]
    )
    response_text = completion.choices[0].message.content

    # 将报告解析为 JSON 格式
    report = {
        "title": "视频监控巡检报告",
        "timestamp": timestamp,
        "inspection_content": "未解析的检查内容",
        "issues_found": "未解析的发现问题",
        "suggestions": "未解析的建议措施",
        "remarks": "未解析的备注"
    }
    try:
        # 解析模型返回的内容
        lines = response_text.split("\n")
        for line in lines:
            if line.startswith("- **标题**:"):
                report["title"] = line.split(":")[1].strip()
            elif line.startswith("- **时间**:"):
                model_timestamp = line.split(":")[1].strip()
                # 验证模型返回的时间戳是否完整（长度为 19）
                if len(model_timestamp) == 19:  # 完整时间戳格式为 YYYY-MM-DD HH:MM:SS
                    report["timestamp"] = model_timestamp
            elif line.startswith("- **检查内容**:"):
                report["inspection_content"] = line.split(":")[1].strip()
            elif line.startswith("- **发现问题**:"):
                report["issues_found"] = line.split(":")[1].strip()
            elif line.startswith("- **建议措施**:"):
                report["suggestions"] = line.split(":")[1].strip()
            elif line.startswith("- **备注**:"):
                report["remarks"] = line.split(":")[1].strip()
    except Exception as e:
        print(f"Error parsing report: {e}")

    return report


# 后台线程：持续捕获图像并进行分析
def background_capture(cap):
    while True:
        time.sleep(2)  # 每隔 2 秒捕获一次图像
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        report = analyze_image_with_qwenvl(frame)
        save_report_to_json(report)


# 将巡检报告保存为 JSON 文件
# 将巡检报告保存为 JSON 文件
def save_report_to_json(report):
    filename = "inspection_reports.json"
    if not os.path.exists(filename):
        # 如果文件不存在，初始化为一个空数组
        with open(filename, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=4)

    # 读取现有数据，添加新报告，然后重新写入
    with open(filename, "r+", encoding="utf-8") as file:
        try:
            data = json.load(file)  # 读取现有数据
        except json.JSONDecodeError:
            data = []  # 如果文件为空或格式错误，初始化为空数组
        data.append(report)  # 添加新报告
        file.seek(0)  # 回到文件开头
        json.dump(data, file, ensure_ascii=False, indent=4)  # 写回文件
        file.truncate()  # 截断多余内容
    print(f"Report saved to {filename}")

# Flask API：获取最新的巡检报告
# Flask API：获取最新的巡检报告
@app.route('/get_latest_report', methods=['GET'])
def get_latest_report():
    try:
        with open("inspection_reports.json", "r", encoding="utf-8") as file:
            try:
                reports = json.load(file)  # 加载整个 JSON 数组
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in inspection_reports.json"}), 500

            if not reports:  # 如果数组为空
                return jsonify({"error": "No reports available"}), 404

            latest_report = reports[-1]  # 获取最后一份报告
        return jsonify(latest_report), 200
    except FileNotFoundError:
        return jsonify({"error": "inspection_reports.json not found"}), 500


@app.route('/get_all_reports', methods=['GET'])
def get_all_reports():
    try:
        with open("inspection_reports.json", "r", encoding="utf-8") as file:
            try:
                reports = json.load(file)  # 加载整个 JSON 数组
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in inspection_reports.json"}), 500

            if not reports:  # 如果数组为空
                return jsonify({"error": "No reports available"}), 404

        return jsonify(reports), 200  # 返回整个数组
    except FileNotFoundError:
        return jsonify({"error": "inspection_reports.json not found"}), 500
# Flask API：启动视频捕获和分析
@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture_thread
    if 'capture_thread' in globals() and capture_thread.is_alive():
        return jsonify({"message": "Capture already running"}), 200

    cap = cv2.VideoCapture(0)  # 打开摄像头
    if not cap.isOpened():
        return jsonify({"error": "Unable to access the camera"}), 500

    capture_thread = threading.Thread(target=background_capture, args=(cap,))
    capture_thread.daemon = True
    capture_thread.start()
    return jsonify({"message": "Capture started"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)