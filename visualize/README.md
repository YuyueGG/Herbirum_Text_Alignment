# Visualisation Tools

This folder contains two lightweight visualisation scripts for inspecting trained herbarium specimen classification models.

Checkpoints are not included in the repository. Please provide the checkpoint path explicitly from the command line.

## Scripts

### `visualize_model_embedding.py`

This script extracts image features from one trained model and projects them to two dimensions for qualitative inspection of the learned feature space.

Example:

```bash
python visualize_model_embedding.py \
  --family alignment \
  --preset resnet44 \
  --checkpoint path/to/resnet44_best.pt \
  --data_jsonl examples/sample_jsonl/cyrtandra44_test.jsonl \
  --output_prefix outputs/embedding/resnet44_ours
```

### `visualize_model_gradcam.py`

This script generates a Grad-CAM overlay for one trained model and one specimen image.

Example:
```bash
python visualize_model_gradcam.py \
  --family ours \
  --preset resnet44 \
  --checkpoint path/to/resnet44_best.pt
```

The default example image is:
```text
examples/sample_images/1.png
```

## Notes

* alignment and ours refer to the proposed model.
* For reproducible figures, keep the same random seed, dimensionality reduction method, and class filtering settings.
* Example label mappings and JSONL files are provided in  `examples/sample_jsonl/`.
* UMAP requires the optional package `umap-learn`.