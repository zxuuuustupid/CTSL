# MCID: Meta-learning Cross-condition Invariant Diagnosis

这是一个使用元学习进行跨工况故障诊断的项目。

## 项目结构

```
.
├── configs/              # 配置文件
│   ├── teacher.yaml
│   └── mcid.yaml
├── data/                 # 数据目录 (需要自行准备)
│   └── gearbox/
│       ├── WC1/
│       ├── ...
│       └── WC9/
├── output/               # 模型输出目录
├── scripts/              # 训练脚本
│   ├── train_teacher.py  # 训练教师模型
│   └── train_main.py     # MCID 元学习训练
└── src/                  # 源代码
    ├── data/
    ├── models/
    └── core/
```

## 使用方法

### 1. 训练教师模型

首先，在单一工况（如 WC1）上训练一个教师模型。

```bash
python scripts/train_teacher.py --config configs/teacher.yaml
```

### 2. 运行 MCID 元学习

使用预训练好的教师模型，进行跨工况元学习。

```bash
python scripts/train_main.py --config configs/mcid.yaml
```

模型和结果将保存在 `ckpts/` 目录下。