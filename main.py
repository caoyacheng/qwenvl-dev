import cv2
import base64
import os
import time
import threading
from openai import OpenAI # 导入 OpenAI 库，用于 DashScope
from dotenv import load_dotenv


load_dotenv()


# ✅ Initialize the OpenAI client for DashScope (Qwen-VL-Max)
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 从环境变量中读取 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # DashScope API Base URL
)

# ✅ Function to send the captured image to Qwen-VL-Max for analysis
def analyze_image_with_qwenvl(image): # 函数名保持不变，因为主体逻辑是图像分析
    if image is None:
        return "No image to analyze."

    # Convert the captured image to base64 (图像预处理部分不变)
    _, img_buffer = cv2.imencode('.jpg', image)
    image_data = base64.b64encode(img_buffer).decode('utf-8')

    # Create the message with the image (使用 Qwen-VL-Max 的消息格式)
    message_content = [
        {"type": "text", "text": "你的任务是用中文描述视频的内容和检查危险行为。如果发现危险信息要特别记录下来，比如抽烟、拿刀。记录格式为：⚠️⚠️存在危险行为，危险行为是[]"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ]

    # Send the message to Qwen-VL-Max and get the response (使用 OpenAI client 调用)
    completion = client.chat.completions.create(
        model="qwen-vl-max", # 模型名称修改为 qwen-vl-max
        messages=[{"role": "user", "content": message_content}]
    )
    return completion.choices[0].message.content #  提取文本响应内容，可能需要根据实际返回结构调整

# ✅ Function to save response to a text file with timestamp (保存部分不变，文件名修改)
def save_response_to_file(response):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = "qwen_responses.txt" # 修改保存的文件名，更清晰

    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"{timestamp} - {response}\n\n")

    print(f"Response saved to {filename}")

# ✅ Function to continuously capture images and analyze them (后台捕获分析逻辑不变，打印信息修改)
def background_capture(cap):
    while True:
        time.sleep(2)  # Wait for 2 seconds before capturing the next image

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        print("发送图片进行分析...")
        response_content = analyze_image_with_qwenvl(frame)  # Analyze the image with Qwen-VL-Max (函数名没改，但内部实现变了)
        print("Qwen-VL-Max Response:", response_content)  # 修改打印信息，更清晰

        save_response_to_file(response_content)  # Save response to a file

# ✅ Main function to show live feed and start background analysis (主函数逻辑不变)
def main():
    cap = cv2.VideoCapture(0)  # Load video file or replace with 0 for webcam

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # 每 2 秒启动一个后台线程来捕获和分析图像
    capture_thread = threading.Thread(target=background_capture, args=(cap,))
    capture_thread.daemon = True  # Ensure the thread exits with the main program
    capture_thread.start()

    # 进入一个无限循环，不断读取视频帧，调整帧大小，并在名为 "Video Feed" 的窗口中显示视频画面
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))  # Resize frame for better display
        cv2.imshow("Video Feed", frame)

        # Exit on 'q' key press
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # 资源释放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()