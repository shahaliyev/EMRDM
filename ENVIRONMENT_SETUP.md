# EMRDM Environment Setup

This should overcome difficulties with setting up the environment as of Jun 13, 2026. Use Python 3.10:

```bash
conda create --name emrdm python=3.10 -y
conda activate emrdm
```

Install PyTorch:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 numpy==1.26.4
```

Optional CUDA 12.1 wheel index:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Do not install NATTEN before PyTorch. Then install NATTEN:

```bash
pip install natten==0.17.1+torch220cu121 -f https://whl.natten.org/old
```

The old `shi-labs.com` NATTEN wheel URL failed due to an expired certificate. Install Lightning and missing packaging support:

```bash
pip install setuptools
pip install pytorch-lightning==2.3.0
```

Install the remaining README dependencies:

```bash
pip install wandb==0.17.8 matplotlib==3.9.2 natsort==8.4.0 omegaconf==2.3.0 scipy==1.14.0 dctorch==0.1.2 rasterio==1.3.11

pip install pandas==2.2.3 opencv-python==4.10.0.84 lpips==0.1.4

pip install tifffile==2024.7.24 s2cloudless==1.7.2 albumentations==1.4.10 albucore==0.0.12
```

Do not start with `pip install -r requirements.txt`. The repository README also warns that the file contains a full environment dump with complex and redundant dependencies. Use it only to look up missing packages.

`flash_attn` was not required for the tested SEN12MS-CR run because `use_flash_attn2: False`. If using a config that enables FlashAttention, install the README-pinned version:

```bash
MAX_JOBS=4 pip install flash_attn==2.5.9.post1 --no-build-isolation
```

## Known Version Pins

| Package | Working Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.2.1+cu121 |
| torchvision | 0.17.1 |
| torchaudio | 2.2.1 |
| NumPy | 1.26.4 |
| NATTEN | 0.17.1+torch220cu121 |
| PyTorch Lightning | 2.3.0 |
| Albumentations | 1.4.10 |
| s2cloudless | 1.7.2 |

Keep NumPy below 2.x. NumPy 2.2.6 caused runtime error. Fix:

```bash
pip install numpy==1.26.4
```