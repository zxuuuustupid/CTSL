"""
类别缺失对学生模型泛化性能影响的消融实验

实验设计：
- 阶梯式探索不同程度的类别缺失 (missing_num = 0, 1, 2, ...)
- 每个Target工况可以缺失部分类别，但所有Target工况的类别并集必须覆盖全部类别
- 研究类别不平衡/缺失对CTSL框架泛化能力的影响

Usage:
    python experiments/students_unbalance/train_imbalance_study.py --config configs/mcid_PU_train_1_meta_2_3.yaml
    python experiments/students_unbalance/train_imbalance_study.py --config configs/mcid_PU_train_1_meta_2_3.yaml --missing_nums 0 1 2 3
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
from torch.autograd import grad
from copy import deepcopy
from itertools import cycle

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import ConcatDataset, DataLoader, Subset
from src.data.dataloader import NpyDataset, get_dataloader
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier
from src.utils.metrics import MetricRecorder


# ============================================================================
# 工具函数
# ============================================================================

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_infinite_loader(loader):
    """将DataLoader转换为无限循环生成器"""
    while True:
        for batch in loader:
            yield batch


# ============================================================================
# 类别缺失方案生成器
# ============================================================================

class ClassMaskingScheme:
    """
    类别缺失方案生成器

    支持两种模式：
    1. 完备模式 (allow_incomplete=False): 所有Target工况的类别并集必须覆盖全部类别
    2. 极限模式 (allow_incomplete=True): 允许部分类别在所有Target中都缺失，
       用于探究学生未见过但教师见过的类别的泛化能力
    """

    def __init__(self, num_classes: int, target_wcs: list, missing_num: int,
                 seed: int = 42, allow_incomplete: bool = False):
        """
        Args:
            num_classes: 总类别数
            target_wcs: 目标工况列表 (如 ['WC2', 'WC3'])
            missing_num: 每个工况缺失的类别数量
            seed: 随机种子
            allow_incomplete: 是否允许不完全覆盖（极限模式）
        """
        self.num_classes = num_classes
        self.target_wcs = target_wcs
        self.missing_num = missing_num
        self.all_classes = set(range(num_classes))
        self.seed = seed
        self.allow_incomplete = allow_incomplete

        # 生成的方案: {wc: [保留的类别列表]}
        self.retained_classes = {}
        # 学生完全未见过的类别（仅教师见过）
        self.unseen_classes = set()

    def generate(self, max_attempts: int = 1000) -> dict:
        """
        生成满足约束的类别保留方案

        Returns:
            dict: {工况名: 保留类别列表}
        """
        rng = random.Random(self.seed)
        best_scheme = None
        best_coverage = 0

        for attempt in range(max_attempts):
            scheme = {}

            for wc in self.target_wcs:
                all_cls = list(self.all_classes)
                rng.shuffle(all_cls)

                # 移除 missing_num 个类别
                if self.missing_num > 0 and self.missing_num < self.num_classes:
                    removed = all_cls[:self.missing_num]
                    retained = sorted(all_cls[self.missing_num:])
                else:
                    retained = sorted(all_cls)

                scheme[wc] = retained

            # 检查完备性：所有工况的类别并集是否覆盖全部类别
            union_classes = set()
            for wc, classes in scheme.items():
                union_classes.update(classes)

            coverage = len(union_classes)

            # 记录最佳方案（覆盖类别最多的）
            if coverage > best_coverage:
                best_coverage = coverage
                best_scheme = scheme

            if union_classes == self.all_classes:
                self.retained_classes = scheme
                self.unseen_classes = set()
                return scheme

            # 不满足，换一个种子重试
            rng = random.Random(self.seed + attempt + 1)

        # 如果允许不完全覆盖（极限模式），使用最佳方案
        if self.allow_incomplete and best_scheme is not None:
            self.retained_classes = best_scheme
            union_classes = set()
            for wc, classes in best_scheme.items():
                union_classes.update(classes)
            self.unseen_classes = self.all_classes - union_classes
            return best_scheme

        raise RuntimeError(
            f"无法在 {max_attempts} 次尝试内生成满足约束的类别方案。\n"
            f"参数: num_classes={self.num_classes}, target_wcs={self.target_wcs}, missing_num={self.missing_num}\n"
            f"提示: 使用 allow_incomplete=True 启用极限模式，或减少 missing_num。"
        )

    def print_scheme(self):
        """打印当前方案"""
        print("\n" + "=" * 60)
        mode_str = "极限模式" if self.allow_incomplete else "完备模式"
        print(f"类别保留方案 (missing_num = {self.missing_num}, {mode_str})")
        print("=" * 60)

        union_classes = set()
        for wc, classes in self.retained_classes.items():
            union_classes.update(classes)
            print(f"  {wc}: {classes} (保留 {len(classes)}/{self.num_classes} 类)")

        print(f"  ────────────────────────────────")
        print(f"  学生可见类别 (Union): {sorted(union_classes)}")

        if union_classes == self.all_classes:
            print(f"  Status: ✓ 完备性检查通过 (覆盖全部 {self.num_classes} 类)")
        else:
            missing = self.all_classes - union_classes
            print(f"  Status: ⚠ 极限模式 - 学生未见类别: {sorted(missing)}")
            print(f"  说明: 这些类别学生从未在训练中见过，但教师已学习过")

        print("=" * 60 + "\n")

    def is_extreme_mode(self) -> bool:
        """判断当前是否为极限模式（存在学生未见类别）"""
        return len(self.unseen_classes) > 0

    def get_unseen_classes(self) -> set:
        """获取学生未见过的类别"""
        return self.unseen_classes


# ============================================================================
# 数据加载（支持类别过滤）
# ============================================================================

def filter_dataset_by_classes(dataset: NpyDataset, retained_classes: list) -> Subset:
    """
    根据保留的类别列表过滤数据集

    Args:
        dataset: 原始NpyDataset
        retained_classes: 保留的类别列表

    Returns:
        Subset: 只包含指定类别的数据子集
    """
    retained_set = set(retained_classes)

    # 获取满足条件的样本索引
    indices = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label in retained_set:
            indices.append(idx)

    return Subset(dataset, indices)


def load_data_split_with_masking(config, class_scheme: dict):
    """
    根据配置文件和类别保留方案加载数据

    Args:
        config: 配置字典
        class_scheme: 类别保留方案 {wc: [保留类别列表]}

    Returns:
        source_iter: 源域数据迭代器 (完整类别)
        target_iters: 目标域数据迭代器字典 (按方案过滤)
    """
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    # === 1. 加载 Source 数据 (完整，不过滤) ===
    source_wcs = data_cfg['source_wc']
    if isinstance(source_wcs, str):
        source_wcs = [source_wcs]

    source_datasets = []
    for wc in source_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            raise FileNotFoundError(f"源工况路径不存在: {path}")
        source_datasets.append(NpyDataset(path))

    combined_source = ConcatDataset(source_datasets)
    source_loader = DataLoader(combined_source, batch_size=batch_size, shuffle=True, pin_memory=True)
    source_iter = get_infinite_loader(source_loader)

    # === 2. 加载 Target 数据 (按方案过滤类别) ===
    target_iters = {}
    target_wcs = data_cfg['target_wcs']

    for wc in target_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            print(f"警告: 目标工况路径不存在，跳过: {path}")
            continue

        # 加载完整数据集
        full_dataset = NpyDataset(path)

        # 根据方案过滤类别
        if wc in class_scheme:
            retained_classes = class_scheme[wc]
            filtered_dataset = filter_dataset_by_classes(full_dataset, retained_classes)
            print(f"  {wc}: {len(full_dataset)} -> {len(filtered_dataset)} 样本 (保留类别: {retained_classes})")
        else:
            filtered_dataset = full_dataset
            print(f"  {wc}: {len(full_dataset)} 样本 (全部保留)")

        loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        target_iters[wc] = get_infinite_loader(loader)

    if len(target_iters) < 2:
        print("警告: target_wcs 数量少于2，元学习无法进行 Support/Query 划分！")

    return source_iter, target_iters


# ============================================================================
# Teacher 模型加载
# ============================================================================

def load_teacher(config, device):
    """加载冻结的教师模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    decoder = MechanicDecoder(cfg['feature_dim'], cfg['input_channels'], cfg['base_filters']).to(device)
    classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)

    source_list = config['data']['source_wc']
    nums = sorted(["".join(filter(str.isdigit, wc)) for wc in source_list], key=int)
    wc_tag = "_".join(nums)
    filename = f"train_{wc_tag}_best_model.pth"
    ckpt_path = os.path.join(
        config['teacher']['checkpoint'],
        config['data']['dataset_name'],
        filename
    )

    if os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        print(f"  Teacher 模型已加载: {ckpt_path}")
    else:
        raise FileNotFoundError(f"教师模型文件未找到: {ckpt_path}")

    encoder.eval()
    classifier.eval()
    decoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    return encoder, classifier, decoder


# ============================================================================
# 损失函数与元学习
# ============================================================================

def compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_s, y_s, x_t, y_t, config):
    """计算通用损失函数 (L_AC + L_CC + L_LC)"""
    loss_cfg = config['loss']

    # 1. 学生前向传播
    feat_s = encoder_s(x_s)
    logits = classifier(feat_s)

    # 2. 教师前向传播
    with torch.no_grad():
        feat_t = encoder_t(x_t).detach()

    # L_AC: 基于类别原型的对齐
    l_ac = torch.tensor(0.0, device=x_s.device)
    valid_classes = 0

    classes_s = torch.unique(y_s)
    classes_t = torch.unique(y_t)
    common_classes = [c for c in classes_s if c in classes_t]

    for c in common_classes:
        proto_s = feat_s[y_s == c].mean(dim=0)
        proto_t = feat_t[y_t == c].mean(dim=0)
        l_ac += nn.MSELoss()(proto_s, proto_t)
        valid_classes += 1

    if valid_classes > 0:
        l_ac = l_ac / valid_classes

    # L_CC: 循环一致性
    x_recon = decoder_t(feat_s, target_length=x_s.shape[-1])
    with torch.no_grad():
        feat_cycle = encoder_t(x_recon).detach()

    if feat_cycle.shape[0] == feat_t.shape[0]:
        l_cc = nn.MSELoss()(feat_cycle, feat_t)
    else:
        l_cc = torch.tensor(0.0, device=feat_cycle.device)

    # L_LC: 标签一致性
    l_lc = nn.CrossEntropyLoss()(logits, y_s)

    total = loss_cfg['lambda_ac'] * l_ac + loss_cfg['lambda_cc'] * l_cc + loss_cfg['lambda_lc'] * l_lc
    return total, {'ac': l_ac.item(), 'cc': l_cc.item(), 'lc': l_lc.item()}


def inner_update(encoder, loss, inner_lr, first_order=True):
    """内循环更新"""
    encoder_prime = deepcopy(encoder)

    grads = grad(loss,
                 encoder.parameters(),
                 create_graph=not first_order,
                 retain_graph=True,
                 allow_unused=True)

    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g

    return encoder_prime


def meta_train_step(source_iter, target_iters, encoder_s, classifier, encoder_t, decoder_t, config, device):
    """元学习训练步骤"""
    meta_cfg = config['meta']
    wc_list = list(target_iters.keys())

    random.shuffle(wc_list)
    query_wc = wc_list[-1]
    support_wcs = wc_list[:-1]

    # Support 阶段
    total_sup_loss = 0
    for wc in support_wcs:
        x_s, y_s = next(target_iters[wc])
        x_s, y_s = x_s.to(device), y_s.to(device)

        x_t, y_t = next(source_iter)
        x_t, y_t = x_t.to(device), y_t.to(device)

        l_step, _ = compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_s, y_s, x_t, y_t, config)
        total_sup_loss += l_step

    if len(support_wcs) > 0:
        total_sup_loss = total_sup_loss / len(support_wcs)
    else:
        total_sup_loss = torch.tensor(0.0, device=device)

    # Inner Loop
    if isinstance(total_sup_loss, torch.Tensor) and total_sup_loss.item() != 0:
        encoder_prime = inner_update(encoder_s, total_sup_loss, meta_cfg['inner_lr'], meta_cfg['first_order'])
    else:
        encoder_prime = encoder_s

    # Query 阶段
    x_q_s, y_q_s = next(target_iters[query_wc])
    x_q_s, y_q_s = x_q_s.to(device), y_q_s.to(device)

    x_q_t, y_q_t = next(source_iter)
    x_q_t, y_q_t = x_q_t.to(device), y_q_t.to(device)

    l_qry, metrics = compute_loss(encoder_prime, classifier, encoder_t, decoder_t, x_q_s, y_q_s, x_q_t, y_q_t, config)

    l_total = total_sup_loss + meta_cfg['beta'] * l_qry

    return l_total, total_sup_loss.item() if isinstance(total_sup_loss, torch.Tensor) else 0, l_qry.item()


# ============================================================================
# 评估函数
# ============================================================================

def evaluate(encoder, classifier, config, device, recorder=None, unseen_classes=None):
    """
    在测试集上评估泛化能力

    Args:
        encoder: 学生编码器
        classifier: 分类器
        config: 配置
        device: 设备
        recorder: 指标记录器
        unseen_classes: 学生未见过的类别集合（用于分析极限模式）
    """
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    results = {}
    total_acc = 0

    # 极限模式下的详细统计
    seen_correct, seen_total = 0, 0
    unseen_correct, unseen_total = 0, 0

    encoder.eval()
    classifier.eval()

    if recorder:
        recorder.reset()

    for wc in data_cfg['test_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'test')
        loader = get_dataloader(path, batch_size, shuffle=False)

        correct, total = 0, 0
        wc_seen_correct, wc_seen_total = 0, 0
        wc_unseen_correct, wc_unseen_total = 0, 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                feat = encoder(x)
                pred = classifier(feat).argmax(1)

                if recorder:
                    recorder.update(wc, pred, y)

                correct += (pred == y).sum().item()
                total += y.size(0)

                # 极限模式下分别统计见过/未见过类别的准确率
                if unseen_classes is not None and len(unseen_classes) > 0:
                    for i in range(y.size(0)):
                        label = y[i].item()
                        is_correct = (pred[i] == y[i]).item()

                        if label in unseen_classes:
                            wc_unseen_total += 1
                            wc_unseen_correct += is_correct
                        else:
                            wc_seen_total += 1
                            wc_seen_correct += is_correct

        acc = 100. * correct / total if total > 0 else 0
        results[wc] = acc
        total_acc += acc

        # 更新全局统计
        seen_correct += wc_seen_correct
        seen_total += wc_seen_total
        unseen_correct += wc_unseen_correct
        unseen_total += wc_unseen_total

        # 打印详细结果
        if unseen_classes is not None and len(unseen_classes) > 0:
            seen_acc = 100. * wc_seen_correct / wc_seen_total if wc_seen_total > 0 else 0
            unseen_acc = 100. * wc_unseen_correct / wc_unseen_total if wc_unseen_total > 0 else 0
            print(f"    {wc}: {acc:.2f}% (见过:{seen_acc:.2f}%, 未见:{unseen_acc:.2f}%)")
        else:
            print(f"    {wc}: {acc:.2f}%")

    avg_acc = total_acc / len(data_cfg['test_wcs'])

    # 打印汇总
    if unseen_classes is not None and len(unseen_classes) > 0:
        global_seen_acc = 100. * seen_correct / seen_total if seen_total > 0 else 0
        global_unseen_acc = 100. * unseen_correct / unseen_total if unseen_total > 0 else 0
        print(f"    ────────────────────────────────")
        print(f"    平均: {avg_acc:.2f}%")
        print(f"    ├─ 学生见过的类别: {global_seen_acc:.2f}% ({seen_correct}/{seen_total})")
        print(f"    └─ 学生未见的类别: {global_unseen_acc:.2f}% ({unseen_correct}/{unseen_total}) [仅教师见过]")

        return avg_acc, {'seen_acc': global_seen_acc, 'unseen_acc': global_unseen_acc,
                         'seen_total': seen_total, 'unseen_total': unseen_total}
    else:
        print(f"    平均: {avg_acc:.2f}%")
        return avg_acc


# ============================================================================
# 单次实验运行
# ============================================================================

def run_single_experiment(config, missing_num: int, exp_seed: int, device, base_output_dir: str,
                          allow_incomplete: bool = False):
    """
    运行单次类别缺失实验

    Args:
        config: 配置字典（会被修改）
        missing_num: 缺失类别数
        exp_seed: 实验随机种子
        device: 训练设备
        base_output_dir: 基础输出目录
        allow_incomplete: 是否允许极限模式（学生未见部分类别）
    """
    print("\n" + "=" * 70)
    mode_str = "极限模式" if allow_incomplete else "完备模式"
    print(f"开始实验: missing_num = {missing_num}, seed = {exp_seed}, {mode_str}")
    print("=" * 70)

    set_seed(exp_seed)

    # 获取参数
    num_classes = config['model']['num_classes']
    target_wcs = config['data']['target_wcs']

    # 计算理论上能覆盖的最大类别数
    # 每个工况保留 (num_classes - missing_num) 个类别
    # n个工况理论上最多覆盖 min(n * (num_classes - missing_num), num_classes) 个类别
    retained_per_wc = max(0, num_classes - missing_num)
    theoretical_max_coverage = min(len(target_wcs) * retained_per_wc, num_classes)

    # 判断是否需要极限模式
    need_extreme = theoretical_max_coverage < num_classes
    actual_allow_incomplete = allow_incomplete or need_extreme

    if need_extreme and not allow_incomplete:
        print(f"  注意: missing_num={missing_num} 过大，无法覆盖所有类别，自动启用极限模式")
        print(f"  理论最大覆盖: {theoretical_max_coverage}/{num_classes} 类")

    # 生成类别保留方案
    scheme_generator = ClassMaskingScheme(
        num_classes=num_classes,
        target_wcs=target_wcs,
        missing_num=missing_num,
        seed=exp_seed,
        allow_incomplete=actual_allow_incomplete
    )

    try:
        class_scheme = scheme_generator.generate()
    except RuntimeError as e:
        print(f"跳过 missing_num={missing_num}: {e}")
        return None

    scheme_generator.print_scheme()

    # 获取未见类别（极限模式专用）
    unseen_classes = scheme_generator.get_unseen_classes()

    # 修改输出路径（隔离实验结果）
    exp_output_dir = os.path.join(
        base_output_dir,
        config['data']['dataset_name'],
        f"missing_{missing_num}_classes",
        f"seed_{exp_seed}"
    )
    config['output']['save_dir'] = exp_output_dir
    os.makedirs(exp_output_dir, exist_ok=True)

    # 加载 Teacher
    print("\n[Step 1] 加载 Teacher 模型...")
    encoder_t, classifier_t, decoder_t = load_teacher(config, device)

    # 构建 Student
    print("\n[Step 2] 构建 Student 模型...")
    encoder_s = MechanicEncoder(
        input_channels=config['model']['input_channels'],
        base_filters=config['model']['base_filters'],
        output_feature_dim=config['model']['feature_dim']
    ).to(device)
    encoder_s.load_state_dict(encoder_t.state_dict())  # 用 Teacher 初始化

    classifier = classifier_t
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # 加载数据（带类别过滤）
    print("\n[Step 3] 加载数据（应用类别过滤）...")
    source_iter, target_iters = load_data_split_with_masking(config, class_scheme)

    # 初始化 MetricRecorder
    log_dir = os.path.join("log", "imbalance_study", config['data']['dataset_name'],
                           f"missing_{missing_num}", f"seed_{exp_seed}")
    os.makedirs(log_dir, exist_ok=True)

    metric_recorder = MetricRecorder(
        save_dir=log_dir,
        config=config,
        class_names=[str(i) for i in range(num_classes)],
    )

    # 保存实验配置
    exp_config = deepcopy(config)
    exp_config['experiment'] = {
        'type': 'imbalance_study',
        'missing_num': missing_num,
        'seed': exp_seed,
        'class_scheme': {k: list(v) for k, v in class_scheme.items()},
        'is_extreme_mode': scheme_generator.is_extreme_mode(),
        'unseen_classes': list(unseen_classes) if unseen_classes else []
    }
    metric_recorder.save_config(exp_config)

    # 优化器
    optimizer = optim.Adam(
        encoder_s.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # 训练循环
    iterations_per_epoch = config['training'].get('iterations_per_epoch', 100)
    total_epochs = 10  # 固定10个epoch
    best_acc = 0
    best_detailed_metrics = None

    print(f"\n[Step 4] 开始训练 (epochs={total_epochs})...")

    for epoch in range(1, total_epochs + 1):
        encoder_s.train()
        total_loss, total_sup, total_qry = 0, 0, 0

        for _ in range(iterations_per_epoch):
            optimizer.zero_grad()

            loss, l_sup, l_qry = meta_train_step(
                source_iter, target_iters,
                encoder_s, classifier, encoder_t, decoder_t,
                config, device
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sup += l_sup
            total_qry += l_qry

        # 日志
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: Loss={total_loss/iterations_per_epoch:.4f}, "
                  f"Sup={total_sup/iterations_per_epoch:.4f}, Qry={total_qry/iterations_per_epoch:.4f}")

        # 评估
        if epoch % 2 == 0 or epoch == total_epochs:
            print(f"  Epoch {epoch} 评估:")
            eval_result = evaluate(encoder_s, classifier, config, device, metric_recorder, unseen_classes)

            # 处理返回值（极限模式返回元组）
            if isinstance(eval_result, tuple):
                avg_acc, detailed_metrics = eval_result
            else:
                avg_acc = eval_result
                detailed_metrics = None

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_detailed_metrics = detailed_metrics

                # 保存模型
                src_list = config['data']['source_wc']
                src_nums = sorted(["".join(filter(str.isdigit, x)) for x in src_list], key=int)
                src_tag = "_".join(src_nums)

                tgt_list = config['data']['target_wcs']
                tgt_nums = sorted(["".join(filter(str.isdigit, x)) for x in tgt_list], key=int)
                tgt_tag = "_".join(tgt_nums)

                file_name = f"mcid_train_{src_tag}_meta_{tgt_tag}_missing_{missing_num}_best.pth"
                save_path = os.path.join(exp_output_dir, file_name)

                save_dict = {
                    'encoder_state_dict': encoder_s.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'best_acc': best_acc,
                    'missing_num': missing_num,
                    'class_scheme': class_scheme,
                    'seed': exp_seed,
                    'is_extreme_mode': scheme_generator.is_extreme_mode(),
                    'unseen_classes': list(unseen_classes) if unseen_classes else []
                }

                if detailed_metrics:
                    save_dict['detailed_metrics'] = detailed_metrics

                torch.save(save_dict, save_path)

                print(f"  ✓ 保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)")
                metric_recorder.calculate_and_save(epoch)

    # 打印最终结果
    print(f"\n实验完成: missing_num={missing_num}, best_acc={best_acc:.2f}%")
    if scheme_generator.is_extreme_mode():
        print(f"  [极限模式] 学生未见类别: {sorted(unseen_classes)}")
        if 'best_detailed_metrics' in dir() and best_detailed_metrics:
            print(f"  见过类别准确率: {best_detailed_metrics['seen_acc']:.2f}%")
            print(f"  未见类别准确率: {best_detailed_metrics['unseen_acc']:.2f}% (教师知识迁移效果)")

    result = {
        'missing_num': missing_num,
        'seed': exp_seed,
        'best_acc': best_acc,
        'class_scheme': class_scheme,
        'is_extreme_mode': scheme_generator.is_extreme_mode(),
        'unseen_classes': list(unseen_classes) if unseen_classes else []
    }

    if 'best_detailed_metrics' in dir() and best_detailed_metrics:
        result['detailed_metrics'] = best_detailed_metrics

    return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="类别缺失对模型泛化性能影响的消融实验")
    parser.add_argument("--config", type=str, default="configs/mcid_PU_train_1_meta_2_4.yaml",
                        help="配置文件路径")
    parser.add_argument("--missing_nums", type=int, nargs='+', default=None,
                        help="要测试的缺失类别数量列表 (默认: 0到num_classes-1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_dir", type=str, default="experiments/students_unbalance/results",
                        help="实验输出目录")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU设备ID")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="允许极限模式（学生未见部分类别，探究教师知识迁移）")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 自动生成缺失数量列表（如果未指定）
    num_classes = config['model']['num_classes']
    if args.missing_nums is None:
        # 默认测试从0到 num_classes-1 的所有缺失情况
        args.missing_nums = list(range(num_classes))

    # 设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 计算极限模式的临界点
    num_target_wcs = len(config['data']['target_wcs'])
    # 当每个工况保留的类别数 * 工况数 < 总类别数时，进入极限模式
    # retained_per_wc = num_classes - missing_num
    # 极限点: num_target_wcs * (num_classes - missing_num) < num_classes
    # => missing_num > num_classes * (1 - 1/num_target_wcs)
    extreme_threshold = int(num_classes * (1 - 1/num_target_wcs)) + 1

    # 打印实验信息
    print("\n" + "=" * 70)
    print("类别缺失消融实验 (Imbalance Study)")
    print("=" * 70)
    print(f"配置文件: {args.config}")
    print(f"数据集: {config['data']['dataset_name']}")
    print(f"总类别数: {num_classes}")
    print(f"源域: {config['data']['source_wc']}")
    print(f"目标域: {config['data']['target_wcs']} ({num_target_wcs}个工况)")
    print(f"测试域: {config['data']['test_wcs']}")
    print(f"缺失数量: {args.missing_nums}")
    print(f"随机种子: {args.seed}")
    print(f"输出目录: {args.output_dir}")
    print(f"允许极限模式: {args.allow_incomplete}")
    print(f"────────────────────────────────────")
    print(f"极限模式临界点: missing_num >= {extreme_threshold}")
    print(f"  (当 missing_num >= {extreme_threshold} 时，学生将无法见到所有类别)")
    print("=" * 70)

    # 运行实验：循环测试不同缺失数量
    results = []

    for missing_num in args.missing_nums:
        exp_config = deepcopy(config)

        result = run_single_experiment(
            config=exp_config,
            missing_num=missing_num,
            exp_seed=args.seed,
            device=device,
            base_output_dir=args.output_dir,
            allow_incomplete=args.allow_incomplete
        )

        if result:
            results.append(result)

    # 打印汇总结果
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    print(f"{'缺失数':<8} {'模式':<10} {'总准确率':<12} {'见过类别':<12} {'未见类别':<12} {'未见类别列表'}")
    print("-" * 80)

    for r in results:
        mode = "极限" if r.get('is_extreme_mode', False) else "完备"
        unseen_list = r.get('unseen_classes', [])

        if 'detailed_metrics' in r and r['detailed_metrics']:
            dm = r['detailed_metrics']
            print(f"{r['missing_num']:<8} {mode:<10} {r['best_acc']:.2f}%{'':<6} "
                  f"{dm['seen_acc']:.2f}%{'':<6} {dm['unseen_acc']:.2f}%{'':<6} {unseen_list}")
        else:
            print(f"{r['missing_num']:<8} {mode:<10} {r['best_acc']:.2f}%{'':<6} "
                  f"{'N/A':<12} {'N/A':<12} {unseen_list}")

    print("=" * 70)

    # 分析教师知识迁移效果
    extreme_results = [r for r in results if r.get('is_extreme_mode', False)]
    if extreme_results:
        print("\n[极限模式分析] 教师知识迁移到学生未见类别的效果:")
        for r in extreme_results:
            if 'detailed_metrics' in r:
                dm = r['detailed_metrics']
                transfer_gap = dm['seen_acc'] - dm['unseen_acc']
                print(f"  missing_num={r['missing_num']}: "
                      f"迁移差距 = {transfer_gap:.2f}% (见过-未见)")

    # 保存汇总结果
    summary_path = os.path.join(args.output_dir, config['data']['dataset_name'], "summary.yaml")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    summary = {
        'config_file': args.config,
        'dataset': config['data']['dataset_name'],
        'num_classes': num_classes,
        'num_target_wcs': num_target_wcs,
        'extreme_threshold': extreme_threshold,
        'seed': args.seed,
        'allow_incomplete': args.allow_incomplete,
        'results': [
            {
                'missing_num': r['missing_num'],
                'best_acc': r['best_acc'],
                'is_extreme_mode': r.get('is_extreme_mode', False),
                'unseen_classes': r.get('unseen_classes', []),
                'class_scheme': {k: list(v) for k, v in r['class_scheme'].items()},
                'detailed_metrics': r.get('detailed_metrics', None)
            }
            for r in results
        ]
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

    print(f"\n汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
