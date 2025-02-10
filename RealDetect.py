import cv2
import base64
import os
import time
import threading
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化 OpenAI 客户端以连接 DashScope (Qwen-VL-Max)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 将捕获的图像发送到 Qwen-VL-Max 进行分析
def analyze_image_with_qwenvl(image):
    if image is None:
        return "No image to analyze."

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
    return completion.choices[0].message.content


# 将响应保存到带有时间戳的文本文件中
def save_response_to_file(response):
    current_date = time.strftime("%Y%m%d")  # 当前日期，格式为 YYYYMMDD
    filename = f"response_{current_date}.txt"
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"{response}\n\n")
    print(f"Response saved to {filename}")


# 后台线程：持续捕获图像并进行分析
def background_capture(cap):
    while True:
        time.sleep(2)  # 每隔 2 秒捕获一次图像
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        response_content = analyze_image_with_qwenvl(frame)
        print("Qwen-VL-Max Response:", response_content)
        save_response_to_file(response_content)


# 主函数：显示实时视频流并启动后台分析
def main():
    cap = cv2.VideoCapture(0)  # 打开摄像头或指定视频文件
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # 启动后台线程进行图像捕获与分析
    capture_thread = threading.Thread(target=background_capture, args=(cap,))
    capture_thread.daemon = True
    capture_thread.start()

    # 显示实时视频流
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 500))  # 调整帧大小以便更好地显示
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # 按下 'q' 键退出
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()