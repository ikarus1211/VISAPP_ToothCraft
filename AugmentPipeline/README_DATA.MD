
# ToothCraft Augmentation Pipeline

This directory contains the data augmentation pipeline for **ToothCraft**.

## Installation

### 1. Create a virtual environment
Conda is recommended, but any environment manager should work.
```bash
conda create -n toothcraft_augpip python=3.10
conda activate toothcraft_augpip
```

### 2. Install dependencies
install dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
## Run

Before generating data, set your preferred setting in [configuration files](./configs).

Then run the main with selected configs:
```bash
python main.py --config-path "configs/" --config-name "gen_ood_sep"
```
We recommend running on multiple CPU to speed up the process