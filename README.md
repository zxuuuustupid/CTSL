# MCID: Meta-learning Cross-condition Invariant Diagnosis

This is a project that leverages meta-learning for cross-condition fault diagnosis.

## Project Structure

```
.
├── configs/              # Configuration files
│   ├── teacher.yaml
│   └── mcid.yaml
├── data/                 # Data directory (to be prepared by the user)
│   └── gearbox/
│       ├── WC1/
│       ├── ...
│       └── WC9/
├── output/               # Model output directory
├── scripts/              # Training scripts
│   ├── train_teacher.py  # Train the teacher model
│   └── train_main.py     # MCID meta-learning training
└── src/                  # Source code
    ├── data/
    ├── models/
    └── core/
```

## Usage

### 1. Train the Teacher Model

First, train a teacher model on a single operating condition (e.g., WC1):

```bash
python scripts/train_teacher.py --config configs/teacher.yaml
```

### 2. Run MCID Meta-learning

Use the pre-trained teacher model to perform cross-condition meta-learning:

```bash
python scripts/train_main.py --config configs/mcid.yaml
```

Models and results will be saved in the `ckpts/` directory.