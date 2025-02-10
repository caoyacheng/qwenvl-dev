import os
import time
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()


# 获取前一天的日期
def get_previous_date():
    today = datetime.now()
    previous_day = today - timedelta(days=1)
    return previous_day.strftime("%Y%m%d")


# 读取前一天的巡检记录文件
def read_inspection_records(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return []

    records = []
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read().strip()
        # 按分隔符分割每份报告
        reports = content.split("----------------------------------------")[1:]  # 跳过第一个空元素
        for report in reports:

            report = report.strip()
            if not report:
                continue

            record = {}
            lines = report.strip().split("\n")
            for line in lines:
                if line.startswith("- **标题**:"):
                    record["title"] = line.split(":")[1].strip()
                elif line.startswith("- **时间**:"):
                    record["timestamp"] = line.split(":")[1].strip()
                elif line.startswith("- **检查内容**:"):
                    record["inspection_content"] = line.split(":")[1].strip()
                elif line.startswith("- **发现问题**:"):
                    record["issues_found"] = line.split(":")[1].strip()
                elif line.startswith("- **建议措施**:"):
                    record["suggestions"] = line.split(":")[1].strip()
                elif line.startswith("- **备注**:"):
                    record["remarks"] = line.split(":")[1].strip()
            if record:
                records.append(record)
    return records


# 调用 DeepSeek API 生成总结
def generate_summary_with_deepseek(records):
    # 将所有记录整理成一段文本
    inspection_text = "\n".join(
        [
            f"- 时间: {record.get('timestamp', '无')}\n"
            f"  检查内容: {record.get('inspection_content', '无')}\n"
            f"  发现问题: {record.get('issues_found', '无')}\n"
            f"  建议措施: {record.get('suggestions', '无')}\n"
            f"  备注: {record.get('remarks', '无')}"
            for record in records
        ]
    )
    # 构造 DeepSeek API 请求
    api_key = os.getenv("DEEPSEEK_API_KEY")  # 确保在 .env 文件中配置 DEEPSEEK_API_KEY
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": f"""
请根据以下巡检记录生成一份总结报告：
{inspection_text}

总结报告应包括以下内容：
1. 巡检总次数。
2. 发现异常的数量及简要说明。
3. 如果未发现问题，请说明巡检正常。
"""
            }
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"]
            return summary
        else:
            print(f"Error calling DeepSeek API: {response.text}")
            return "未能生成总结内容，请检查 API 配置。"
    except Exception as e:
        print(f"Exception occurred while calling DeepSeek API: {e}")
        return "未能生成总结内容，请检查网络连接或 API 配置。"


# 生成 Markdown 文件
def generate_markdown(records, output_filename, inspection_date, summary):
    markdown_content = []

    # 添加标题
    markdown_content.append("# 每日巡检报告")
    markdown_content.append(f"**日期**: {inspection_date}")
    markdown_content.append("")

    # 添加总结内容
    markdown_content.append("## 巡检总结")
    markdown_content.append(summary)
    markdown_content.append("")

    # 添加巡检记录
    markdown_content.append("## 巡检记录")
    for i, record in enumerate(records, start=1):
        markdown_content.append(f"### 报告 {i}")
        markdown_content.append(f"- **标题**: {record.get('title', '无')}")
        markdown_content.append(f"- **时间**: {record.get('timestamp', '无')}")
        markdown_content.append(f"- **检查内容**: {record.get('inspection_content', '无')}")
        markdown_content.append(f"- **发现问题**: {record.get('issues_found', '无')}")
        markdown_content.append(f"- **建议措施**: {record.get('suggestions', '无')}")
        markdown_content.append(f"- **备注**: {record.get('remarks', '无')}")
        markdown_content.append("")

    # 写入 Markdown 文件
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("\n".join(markdown_content))
    print(f"Markdown report saved to {output_filename}")


# 主函数
def main():
    # 获取前一天的日期
    previous_date = get_previous_date()
    input_filename = f"response_{previous_date}.txt"
    output_filename = f"DailyReport_{previous_date}.md"

    # 读取巡检记录
    records = read_inspection_records(input_filename)
    if not records:
        print("No records found. Exiting...")
        return

    # 调用 DeepSeek API 生成总结
    summary = generate_summary_with_deepseek(records)

    # 生成 Markdown 报告
    generate_markdown(records, output_filename, previous_date, summary)


if __name__ == "__main__":
    main()