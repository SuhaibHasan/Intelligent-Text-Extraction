# CLAUDE.md — Intelligent Text Extraction (ANPR)

## Project Overview

This project implements an **Automatic Number Plate Recognition (ANPR)** OCR pipeline using deep learning. It trains a CNN + Bidirectional GRU neural network with Connectionist Temporal Classification (CTC) loss to recognize license plate text from images.

The entire pipeline lives in a single Jupyter notebook: `TextExtraction.ipynb`.

---

## Repository Structure

```
Intelligent-Text-Extraction/
├── README.md                        # User-facing documentation
├── CLAUDE.md                        # This file (AI assistant guidance)
├── TextExtraction.ipynb             # Main notebook: full training/eval pipeline
└── anpr_ocr/
    ├── meta.json                    # Dataset metadata (tags, classes)
    ├── anpr_ocr__train/             # Training + validation data
    │   ├── img/                     # 10,821 PNG/JPG plate images
    │   └── ann/                     # 10,821 JSON annotation files
    └── anpr_ocr__test/              # Test data
        ├── img/                     # 561 PNG/JPG plate images
        └── ann/                     # 561 JSON annotation files
```

**No separate Python modules or packages exist** — all logic is self-contained in the notebook.

---

## Architecture

### Neural Network: CNN + Bidirectional GRU + CTC

```
Input (128×64×1 grayscale)
  → Conv2D(16, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(16, 3×3, ReLU) → MaxPool(2×2)
  → Reshape (32 time steps × 256 features)
  → Dense(32, ReLU)
  → Bidirectional GRU(512)
  → Bidirectional GRU(512)
  → Dense(num_classes) → Softmax
  → CTC Loss (Lambda layer)
```

- Downsampling factor: 4 (two MaxPool(2×2) layers)
- Time steps after CNN: 32 (128 / 4)
- Output classes: 19 (18 characters + 1 CTC blank token)
- Max text length: 8 characters

### Character Set (18 characters)

```
Digits: 0 1 2 3 4 5 6 7 8 9
Letters: A B C E H K M O P T X Y
```

Only these characters appear in the dataset. Inputs containing other characters are invalid.

### Key Hyperparameters

| Parameter | Value | Location in Notebook |
|-----------|-------|----------------------|
| Image width | 128 | Cell 7 (`train()`) |
| Image height | 64 | Cell 4 (`TextImageGenerator`) |
| Batch size | 32 | Cell 5 |
| Max text length | 8 | Cell 5 |
| GRU units | 512 | Cell 7 |
| Dense size (pre-RNN) | 32 | Cell 7 (`time_dense_size`) |
| Learning rate | 0.02 | Cell 7 |
| Optimizer | SGD (momentum=0.9, decay=1e-6, clipnorm=5) | Cell 7 |

---

## Data Pipeline

### Annotation Format

Each `.json` file in `ann/` follows this schema:

```json
{
  "tags": ["train"],        // dataset split(s): "train", "val", or "test"
  "description": "A001CB06", // ground-truth plate text (target label)
  "objects": [],             // unused in this project
  "size": {
    "height": 34,
    "width": 152
  }
}
```

The `tags` field controls which split each sample belongs to. A sample can belong to multiple splits.

### `TextImageGenerator` Class (Cell 4)

Handles data loading and batch generation:

- `dirpath`: path to dataset directory (e.g., `anpr_ocr/anpr_ocr__train`)
- `tag`: one of `"train"`, `"val"`, `"test"` — filters samples
- `img_w`, `img_h`: resize dimensions (128, 64)
- `batch_size`: samples per batch
- `downsample_factor`: CNN temporal downsampling (4)
- `max_text_len`: max characters per plate (8)

Preprocessing per image:
1. Read as grayscale
2. Resize to `(img_w, img_h)`
3. Normalize to `[0, 1]`
4. Transpose for time-major format: `(img_w, img_h, 1)`

Batch output dict keys: `the_input`, `the_labels`, `input_length`, `label_length`.

---

## Notebook Cell Guide

| Cell | Purpose |
|------|---------|
| 0 | Version check: confirms TF 1.11.0, Keras 2.2.4 |
| 1 | Library imports (cv2, numpy, keras, matplotlib, skimage) |
| 2 | TensorFlow session initialization |
| 3 | Character set analysis — reads annotations, builds alphabet |
| 4 | `TextImageGenerator` class + helper functions (`text_to_labels`, `labels_to_text`, `is_valid_str`) |
| 5 | Instantiate train/val/test generators |
| 6 | Visualize a batch (inspect images + labels) |
| 7 | `ctc_lambda_func` + `train()` — defines and compiles the model |
| 8 | Execute training: `model = train(img_w=128, load=False)` |
| 9 | `decode_batch()` — greedy best-path CTC decoding |
| 10 | Test evaluation: run model on test set, display predictions vs ground truth |
| 11 | Empty placeholder |

Run cells **sequentially** from top to bottom. Each cell depends on state from previous cells.

---

## Development Workflows

### Running the Notebook

```bash
# Install dependencies first (see below)
jupyter notebook TextExtraction.ipynb
# Then: Kernel → Restart & Run All
```

### Installing Dependencies

No `requirements.txt` exists. Install manually:

```bash
pip install tensorflow==1.15 keras==2.2.4 numpy scipy matplotlib scikit-image opencv-python jupyter
```

Or with TensorFlow 2.x (requires code adjustments — see Known Issues):

```bash
pip install tensorflow numpy scipy matplotlib scikit-image opencv-python jupyter
```

### Modifying the Model

- **Change architecture**: Edit Cell 7 (`train()` function)
- **Change hyperparameters**: Edit the `train()` call in Cell 8 or update defaults in Cell 7
- **Change character set**: Edit Cell 3 (hard-coded alphabet) and update `text_to_labels`/`labels_to_text` in Cell 4

### Adding New Data

1. Place images in `anpr_ocr/anpr_ocr__train/img/` (PNG or JPG)
2. Create matching JSON files in `anpr_ocr/anpr_ocr__train/ann/`
3. Set `"tags": ["train"]` or `["val"]` in the JSON
4. Use only supported characters in `"description"`

---

## Known Issues & Limitations

### Path Compatibility

The README shows Windows-style backslash paths (`anpr_ocr\\anpr_ocr__train`). The notebook itself uses forward slashes. When modifying paths, use `pathlib.Path` or `os.path.join()` for cross-platform compatibility.

### Legacy Framework

The project targets **Keras 2.2.4** standalone with **TensorFlow 1.x backend**. If running on TensorFlow 2.x:

- Replace `import keras` with `from tensorflow import keras`
- Replace `K.ctc_batch_cost` → `tf.keras.backend.ctc_batch_cost`
- Replace `model.fit_generator()` → `model.fit()`
- TF session initialization (Cell 2) is not needed and should be removed

### No Model Persistence

The notebook does not save or load trained weights. Add model checkpointing to Cell 7 or 8:

```python
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('model_best.h5', save_best_only=True, monitor='val_loss')
```

### Greedy Decoding Only

`decode_batch()` uses greedy best-path decoding. For improved accuracy, consider beam search decoding via `tf.nn.ctc_beam_search_decoder`.

### No Metrics Calculation

The test evaluation in Cell 10 shows qualitative visualizations only. No CER (Character Error Rate) or accuracy metrics are computed automatically.

---

## Conventions for AI Assistants

### What to Preserve

- **Cell ordering**: The notebook is stateful; cells must execute sequentially
- **Keras functional API**: The model uses `Input → layer(prev)` chaining, not `Sequential`
- **CTC loss Lambda layer**: The loss is computed inside the model graph via a `Lambda` layer — this is intentional for Keras 1.x/2.x compatibility
- **Batch dict format**: Training inputs use named keys (`the_input`, `the_labels`, etc.) — Keras requires these exact names to match model input/output names

### What to Avoid

- Do not refactor the notebook into a Python module without a specific request — the notebook structure is intentional for educational use
- Do not upgrade framework versions without updating all compatibility shims
- Do not change the character set without verifying the dataset contains only those characters
- Do not add beam search or language model components without preserving the existing greedy path for comparison

### Code Style

- Inline comments above code blocks (existing style)
- Functional helpers (no classes except `TextImageGenerator`)
- NumPy operations preferred over Python loops
- Keep the pipeline linear: data load → model build → train → evaluate

---

## Dataset Statistics

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | ~8,600 | ~8,600 |
| Val | ~2,200 | ~2,200 |
| Test | 561 | 561 |
| **Total** | **11,382** | **11,382** |

(Train/Val split is determined by `"tags"` in annotation files, not by directory.)

---

## Git Workflow

- Main branch: `main`
- Development branches follow the pattern: `claude/<description>`
- No CI/CD pipeline exists yet
- Commits should be descriptive: what changed and why

---

## Future Improvement Areas

If extending this project, prioritize in this order:

1. **`requirements.txt`** — critical for reproducibility
2. **`.gitignore`** — exclude `__pycache__`, `.ipynb_checkpoints`, `*.h5`
3. **Model saving** — checkpoint callbacks + `model.save()`
4. **CER metric** — character error rate calculation in test evaluation
5. **TF 2.x migration** — update imports and remove TF session code
6. **Config file** — externalize hyperparameters from notebook cells
7. **CLI interface** — `train.py` and `predict.py` scripts
8. **Unit tests** — test `text_to_labels`, `labels_to_text`, annotation loading
