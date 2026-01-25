# MCID: Meta-learning Cross-condition Invariant Diagnosis

## Overview

This repository provides the official implementation of **MCID**, a meta-learning framework designed for cross-condition fault diagnosis in rotating machinery. The proposed method addresses the challenge of domain shift across varying operating conditions by leveraging a teacher-student paradigm combined with episodic meta-learning.

## Repository Structure

```
.
├── configs/      # Configuration files for teacher and MCID training
├── data/         # Dataset directory (user-prepared, organized by working conditions)
├── ckpts/        # Saved model checkpoints and training artifacts
├── scripts/      # Training and evaluation scripts
├── src/          # Core source code (data loaders, models, training logic)
└── README.md
```

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Train the Teacher Model

Train a teacher model on a single source working condition (e.g., WC1):

```bash
python scripts/train_teacher.py --config configs/teacher_<dataset>_train_<wc>.yaml
```

### Step 2: Meta-learning with MCID

Perform cross-condition meta-learning using the pre-trained teacher:

```bash
python scripts/train_main.py --config configs/mcid_<dataset>_train_<source>_meta_<targets>.yaml
```

Trained models and experiment logs will be saved in the `ckpts/` and `log/` directories.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{mcid2026,
  title={MCID: Meta-learning Cross-condition Invariant Diagnosis},
  author={Zhixu Duan and Zuoyi Chen},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026}
}
```

## License

This project is released for academic research purposes.
