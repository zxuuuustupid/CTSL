import os
import time
from modelscope.hub.api import HubApi
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()
YOUR_ACCESS_TOKEN = os.getenv("YOUR_ACCESS_TOKEN")

if not YOUR_ACCESS_TOKEN:
    raise ValueError("YOUR_ACCESS_TOKEN not found in environment variables")

# 初始化 API
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

repo_id = "Jusin0305/mcid"
local_data_dir = r"F:\Project\mid\S-MID\data"

print(f"🚀 开始上传数据到魔搭社区（带自动续传功能）...")

max_retries = 20  # 最大重试次数
retry_delay = 5   # 失败后等待 5 秒再次尝试

for i in range(max_retries):
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_data_dir,
            repo_type="dataset",
            commit_message=f"Upload batch retry {i}",
        )
        print("✅ 【全部完成】所有数据已成功上传！")
        break  # 成功后退出循环
    except Exception as e:
        print(f"⚠️ 第 {i+1} 次上传中断（原因：网络抖动）。")
        print(f"错误详情: {e}")
        if i < max_retries - 1:
            print(f"等待 {retry_delay} 秒后自动尝试续传...")
            time.sleep(retry_delay)
        else:
            print("❌ 重试次数过多，请检查网络环境或关闭代理/VPN。")
