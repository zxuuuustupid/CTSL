"""
Teacher 模型训练脚本
使用编码器 (Encoder) + 分类器 (Classifier) 进行故障诊断模型训练
在单一工况下训练，在所有工况下测试
"""

import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import get_dataloader
from src.models.encoder import MechanicEncoder
from src.models.classifier import MechanicClassifier


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config):
    """获取计算设备"""
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"使用设备: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")
    return device


def build_model(config, device):
    """构建编码器和分类器"""
    encoder_cfg = config['model']['encoder']
    classifier_cfg = config['model']['classifier']
    
    encoder = MechanicEncoder(
        input_channels=encoder_cfg['input_channels'],
        base_filters=encoder_cfg['base_filters'],
        output_feature_dim=encoder_cfg['output_feature_dim']
    ).to(device)
    
    classifier = MechanicClassifier(
        feature_dim=classifier_cfg['feature_dim'],
        num_classes=classifier_cfg['num_classes'],
        dropout_rate=classifier_cfg['dropout_rate']
    ).to(device)
    
    return encoder, classifier


def build_optimizer(encoder, classifier, config):
    """构建优化器"""
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(
        params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    return optimizer


def build_scheduler(optimizer, config):
    """构建学习率调度器"""
    scheduler_cfg = config['training']['scheduler']
    scheduler_type = scheduler_cfg['type']
    
    if scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg['step_size'],
            gamma=scheduler_cfg['gamma']
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg['T_max']
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    return scheduler


def train_one_epoch(encoder, classifier, train_loader, criterion, optimizer, device, config):
    """训练一个epoch"""
    encoder.train()
    classifier.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    log_interval = config['output']['log_interval']
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        features = encoder(data)
        outputs = classifier(features)
        
        # 计算损失
        loss = criterion(outputs, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(encoder, classifier, test_loader, criterion, device, wc_name=""):
    """在测试集上评估模型"""
    encoder.eval()
    classifier.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            features = encoder(data)
            outputs = classifier(features)
            
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate_all_wcs(encoder, classifier, config, criterion, device):
    """在所有工况上评估模型"""
    results = {}
    data_cfg = config['data']
    batch_size = config['training']['batch_size']
    
    print("\n" + "=" * 60)
    print("在所有工况上评估:")
    print("=" * 60)
    
    total_acc = 0.0
    for wc in data_cfg['test_wcs']:
        test_path = os.path.join(data_cfg['root_dir'], wc, 'test')
        test_loader = get_dataloader(test_path, batch_size=batch_size, shuffle=False)
        
        loss, acc = evaluate(encoder, classifier, test_loader, criterion, device, wc)
        results[wc] = {'loss': loss, 'accuracy': acc}
        total_acc += acc
        
        print(f"  {wc}: Loss={loss:.4f}, Acc={acc:.2f}%")
    
    avg_acc = total_acc / len(data_cfg['test_wcs'])
    print(f"\n  平均准确率: {avg_acc:.2f}%")
    print("=" * 60)
    
    return results, avg_acc


def save_checkpoint(encoder, classifier, optimizer, epoch, best_acc, save_path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")


def main(config_path):
    """主训练函数"""
    # 加载配置
    config = load_config(config_path)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 创建输出目录
    save_dir = os.path.join(config['output']['save_dir'],config['data']['dataset_name'])
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存一份配置文件副本
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = os.path.join(save_dir, f"config_{timestamp}.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    # 获取设备
    device = get_device(config)
    
    # 构建模型
    print("\n构建模型...")
    encoder, classifier = build_model(config, device)
    
    # 打印模型参数量
    encoder_params = sum(p.numel() for p in encoder.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"Encoder 参数量: {encoder_params:,}")
    print(f"Classifier 参数量: {classifier_params:,}")
    print(f"总参数量: {encoder_params + classifier_params:,}")
    
    # 构建优化器和调度器
    optimizer = build_optimizer(encoder, classifier, config)
    scheduler = build_scheduler(optimizer, config)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 加载训练数据
    data_cfg = config['data']
    train_path = os.path.join(data_cfg['root_dir'], data_cfg['train_wc'], 'train')
    batch_size = config['training']['batch_size']
    train_loader = get_dataloader(train_path, batch_size=batch_size, shuffle=True)
    
    # 训练配置
    epochs = config['training']['epochs']
    early_stopping_cfg = config['training']['early_stopping']
    
    # 早停相关变量
    best_avg_acc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print(f"开始训练 - 训练工况: {data_cfg['train_wc']}")
    print(f"训练轮数: {epochs}, 批次大小: {batch_size}")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}] - LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            encoder, classifier, train_loader, criterion, optimizer, device, config
        )
        print(f"训练集 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 在所有工况上评估
        results, avg_acc = evaluate_all_wcs(encoder, classifier, config, criterion, device)
        
        # 保存最佳模型
        if config['output']['save_best'] and avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_model_path = os.path.join(save_dir, "best_model.pth")
            save_checkpoint(encoder, classifier, optimizer, epoch, best_avg_acc, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if early_stopping_cfg['enabled']:
            if patience_counter >= early_stopping_cfg['patience']:
                print(f"\n早停触发！已经 {patience_counter} 个epoch没有提升。")
                break
    
    # 保存最后一个模型
    if config['output']['save_last']:
        last_model_path = os.path.join(save_dir, "last_model.pth")
        save_checkpoint(encoder, classifier, optimizer, epoch, avg_acc, last_model_path)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳平均准确率: {best_avg_acc:.2f}%")
    print(f"模型保存目录: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher 模型训练脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/teacher.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    
    main(args.config)
