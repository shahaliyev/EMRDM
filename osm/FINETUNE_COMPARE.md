# OSM Fine-Tuning and Comparison

This assumes:

- SEN12MS-CR is at `/home/temporaryuser3/Desktop/data/SEN12MSCR`
- OSM masks are at `/home/temporaryuser3/Desktop/osm/out`
- Original EMRDM checkpoint is `checkpoints/sentinel/last.ckpt`
- OSM config is `configs/example_training/sentinel_osm.yaml`

## 1. Convert the Pretrained Checkpoint

Run once:

```bash
python osm/convert_ckpt.py \
  --input checkpoints/sentinel/last.ckpt \
  --output checkpoints/sentinel_osm/last_osm.ckpt
```

Expected output includes:

```text
model.diffusion_model.patch_in.proj.weight: (128, 28) -> (128, 31)
model_ema.diffusion_modelpatch_inprojweight: (128, 28) -> (128, 31)
```

## 2. Start Fine-Tuning

```bash
conda activate emrdm
cd ~/Desktop/code/emrdm
```

Run fine-tuning:

```bash
python main.py \
  -b configs/example_training/sentinel_osm.yaml \
  -t True \
  --no-test True \
  model.params.ckpt_path=checkpoints/sentinel_osm/last_osm.ckpt \
  lightning.trainer.max_steps=5001 \
  lightning.trainer.max_epochs=-1 \
  lightning.trainer.limit_val_batches=0 \
  lightning.trainer.num_sanity_val_steps=0 \
  lightning.callbacks.image_logger.params.disabled=True \
  lightning.modelcheckpoint.params.every_n_train_steps=1000
```

For a shorter test, replace `5001` with `2001`. For a longer run, increase it.

`--no-test True` and `lightning.callbacks.image_logger.params.disabled=True` are important for OSM fine-tuning. Without them, the post-training test/image logger can try to save the 18-channel conditioning tensor as an image.

The fine-tuned checkpoint will be saved under:

```text
logs/<osm_training_run>/checkpoints/last.ckpt
```

## 3. Evaluate the Original Baseline

If you already have baseline metrics, skip this and reuse that `metrics.csv`.

```bash
python main.py \
  -b configs/example_training/sentinel_single_gpu.yaml \
  -t False \
  --predict True \
  model.params.ckpt_path=checkpoints/sentinel/last.ckpt
```

Output:

```text
logs/<baseline_eval_run>/metrics.csv
```

## 4. Evaluate the OSM Fine-Tuned Checkpoint

Replace `<osm_training_run>` with the actual fine-tuning log folder:

```bash
python main.py \
  -b configs/example_training/sentinel_osm.yaml \
  -t False \
  --predict True \
  model.params.ckpt_path=logs/<osm_training_run>/checkpoints/last.ckpt
```

Output:

```text
logs/<osm_eval_run>/metrics.csv
```

## 5. Compare Metrics

Replace the log folder names with the actual folders:

```bash
python osm/compare_checkpoints.py \
  --baseline logs/<baseline_eval_run>/metrics.csv \
  --candidate logs/<osm_eval_run>/metrics.csv \
  --osm-root /home/temporaryuser3/Desktop/osm/out \
  --output logs/osm_comparison.json
```

Read these subsets first:

- `osm_any`
- `road`
- `water`
- `building`

The `all` subset may change only slightly because many test patches can have empty OSM masks.

Fine-tuning creates new checkpoints in `logs/<osm_training_run>/checkpoints/`.
