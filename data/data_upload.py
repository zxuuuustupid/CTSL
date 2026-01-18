import os
from modelscope.msdatasets import MsDataset
from modelscope.hub.api import HubApi

# 1. 填入你的魔搭 SDK Token
YOUR_ACCESS_TOKEN = "这里填入你的魔搭Token"

# 2. 初始化 API
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

# 3. 配置路径
repo_id = "Jusin0305/mcid"  # 填入你刚才在官网创建的数据集ID
local_data_dir = r"F:\Project\mid\S-MID\data\gearbox" # 你的本地几十GB数据根目录

print(f"🚀 开始上传数据到魔搭社区...")

# 4. 执行上传
# upload_folder 会自动递归上传子文件夹，并处理大文件分片
try:
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_data_dir,
        repo_type="dataset",
        commit_message="Upload gearbox dataset (tens of GBs)",
    )
    print("✅ 全部数据上传完成！")
except Exception as e:
    print(f"❌ 上传失败，你可以再次运行脚本进行续传。错误信息：\n{e}")
