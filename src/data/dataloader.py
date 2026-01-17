import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DirectNpyDataset(Dataset):
    def __init__(self, file_list):
        """
        Args:
            file_list (list): 包含 {'path': str, 'label': int} 的列表
        """
        self.all_data = []
        self.all_labels = []

        # 1. 遍历所有文件，暴力加载
        for item in file_list:
            filepath = item['path']
            label = item['label']
            
            try:
                # 假设每个 npy 都是 (N, C, L)
                data_numpy = np.load(filepath)
                
                # 强校验：必须是 3 维
                if data_numpy.ndim != 3:
                    raise ValueError(f"Shape Error: {filepath} shape is {data_numpy.shape}, expected (N, C, L)")
                
                # 统一转 float32 (PyTorch需要)
                if data_numpy.dtype != np.float32:
                    data_numpy = data_numpy.astype(np.float32)

                self.all_data.append(data_numpy)
                
                # 生成对应的标签数组
                # data_numpy.shape[0] 是样本个数 N
                current_n = data_numpy.shape[0]
                self.all_labels.append(np.full(current_n, label, dtype=np.int64))
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        # 2. 拼接 (Concatenate)
        # 把所有文件的 (N1, C, L), (N2, C, L)... 拼成一个大数组 (Total_N, C, L)
        if len(self.all_data) > 0:
            self.data_tensor = torch.from_numpy(np.concatenate(self.all_data, axis=0))
            self.label_tensor = torch.from_numpy(np.concatenate(self.all_labels, axis=0))
            
            # 打印一下，让你知道具体数字是多少
            print(f"Dataset Loaded. Total Shape: {self.data_tensor.shape}") 
            # 例如输出: (10000, 1, 2048) -> N=10000, C=1, L=2048
        else:
            raise RuntimeError("No valid data loaded!")

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.label_tensor[idx]


class DataManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 标签映射，随时可在外部修改
        self.label_map = {
            'normal': 0, 
            'inner': 1, 
            'outer': 2, 
            'ball': 3
        }

    def get_dataloader(self, domain_folder, batch_size=32, shuffle=True):
        """
        Args:
            domain_folder (str): 根目录下的子文件夹名
        """
        target_path = os.path.join(self.root_dir, domain_folder)
        
        # 1. 扫描文件
        file_list = []
        for root, _, files in os.walk(target_path):
            for f in files:
                if f.endswith('.npy'):
                    full_path = os.path.join(root, f)
                    
                    # 匹配标签
                    matched_label = None
                    path_lower = full_path.lower()
                    for key, val in self.label_map.items():
                        if key in path_lower:
                            matched_label = val
                            break
                    
                    if matched_label is not None:
                        file_list.append({'path': full_path, 'label': matched_label})
        
        if not file_list:
            raise ValueError(f"No matched .npy files in {target_path}")

        # 2. 构造 Dataset (内部自动拼接)
        dataset = DirectNpyDataset(file_list)
        
        # 3. 返回 Loader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)