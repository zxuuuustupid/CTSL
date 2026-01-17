import os
import json
import torch
import numpy as np
import pandas as pd
import datetime
# from datetime import datetime
import yaml
from sklearn.metrics import confusion_matrix, accuracy_score

class MetricRecorder:
    def __init__(self, save_dir, experiment_name, class_names=None):
        """
        Args:
            save_dir (str): 根日志目录
            experiment_name (str): 实验名称
            class_names (list): 类别名称列表 (e.g. ['G0', 'G3', 'G7', 'G8'])
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
        self.base_dir = os.path.join('log', experiment_name, timestamp)
        
        self.class_names = class_names
        # 缓冲区: { 'WC1': {'preds': [], 'targets': []}, ... }
        self.buffer = {} 
        
        # 确保目录存在
        os.makedirs(self.base_dir, exist_ok=True)


    def save_config(self, config):
        """保存配置文件到当前时间戳目录下"""
        yaml_path = os.path.join(self.base_dir, 'config.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
            
            
    def reset(self):
        """每轮评估前重置缓冲区"""
        self.buffer = {}

    def update(self, condition_name, preds, targets):
        """累积 Batch 数据"""
        if condition_name not in self.buffer:
            self.buffer[condition_name] = {'preds': [], 'targets': []}
            
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        self.buffer[condition_name]['preds'].append(preds)
        self.buffer[condition_name]['targets'].append(targets)

    def calculate_and_save(self, epoch):
        """
        计算指标并保存 CSV 和 JSON (覆盖旧文件)
        """
        summary_data = []
        
        # 遍历每个工况 (WC1, WC2...)
        for wc, data in self.buffer.items():
            y_pred = np.concatenate(data['preds'])
            y_true = np.concatenate(data['targets'])
            
            # 计算混淆矩阵 & 总准确率
            cm = confusion_matrix(y_true, y_pred)
            total_acc = accuracy_score(y_true, y_pred) * 100
            
            # 计算各故障类别的单独准确率
            # 加上 1e-6 防止除以0
            per_class_acc = (cm.diagonal() / (cm.sum(axis=1) + 1e-6)) * 100
            
            # 确定类别标签
            if self.class_names is None:
                labels = [f"Class {i}" for i in range(len(cm))]
            else:
                labels = self.class_names

            # === 1. 保存详细 JSON (含混淆矩阵数据) ===
            details = {
                "Epoch": epoch,
                "Condition": wc,
                "Total_Accuracy": float(total_acc),
                "Per_Class_Accuracy": {label: float(acc) for label, acc in zip(labels, per_class_acc)},
                "Confusion_Matrix": cm.tolist() # 只存数据，不存图
            }
            
            json_path = os.path.join(self.base_dir, f"best_metrics_{wc}.json")
            with open(json_path, 'w') as f:
                json.dump(details, f, indent=4)
            
            # === 2. 收集汇总数据用于 CSV ===
            row = {
                "Condition": wc,
                "Total Acc": f"{total_acc:.2f}%"
            }
            for label, acc in zip(labels, per_class_acc):
                row[f"{label} Acc"] = f"{acc:.2f}%"
            summary_data.append(row)

        # === 3. 保存汇总表格 CSV ===
        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values(by="Condition")
            
            csv_path = os.path.join(self.base_dir, "best_summary_report.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            print(f"[Metrics] 最佳结果数据已更新: {csv_path}")