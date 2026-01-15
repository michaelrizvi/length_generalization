# Hessian Analysis Module

This module provides tools for computing and visualizing Hessian eigenvalue spectral density (ESD) analysis for models trained on algorithmic tasks.

## Structure

- **`utils.py`**: Core utility functions for loading models and computing Hessian metrics
  - `load_model_and_config()`: Load trained model from checkpoint
  - `create_hessian_dataloaders()`: Create dataloaders for Hessian computation
  - `compute_hessian_metrics()`: Compute top eigenvalues and eigenvalue spectral density
  - `BatchWrapper`, `ModelWrapper`, `HessianDataLoaderWrapper`: Helper classes for PyHessian compatibility

- **`plot_hessian_simple.py`**: Analyze a single model checkpoint
  - Generates plots for train, same-length test, and length-generalization test sets
  - Automatically extracts training length range from model path
  - Saves results to `hessian_results/simple/{model_name}/`

- **`plot_hessian_many_models.py`**: Compare multiple model checkpoints
  - Analyzes the training set only across multiple models
  - Useful for studying how training length affects Hessian eigenvalues
  - Saves all results to a single directory with overlay plots
  - Automatically extracts training length from each model path

- **`plot_hessian_many_datasets.py`**: Analyze a single model on multiple dataset ranges
  - Evaluates one model on datasets of increasing length
  - Useful for studying how dataset length affects the Hessian spectrum
  - Saves all results to a single directory with overlay plots
  - Perfect for analyzing model robustness across different input lengths

## Usage

### Single Model Analysis

```bash
python plot_hessian_simple.py \
    --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \
    --task unique_copy \
    --output_dir ./hessian_results/simple \
    --batch_size 200 \
    --max_samples 500 \
    --top_k 100
```

**Key arguments:**
- `--model_path`: Path to model checkpoint directory
- `--task`: Task name (bin_majority, unique_copy, sort, etc.)
- `--output_dir`: Base output directory (default: `./hessian_results/simple`)
- `--batch_size`: Batch size for Hessian computation (default: 200)
- `--max_samples`: Max samples per dataset for efficiency (default: 500)
- `--top_k`: Number of top eigenvalues to compute (default: 100)
- `--train_seed`, `--test_seed`, `--hessian_seed`: Random seeds for reproducibility

**Output:**
- `hessian_results/simple/{model_name}/esd_*.pdf`: Eigenvalue spectral density plots
- `hessian_results/simple/{model_name}/top_eigenvalues_*.pdf`: Top eigenvalue plots

### Multiple Models Analysis

```bash
python plot_hessian_many_models.py \
    --model_paths ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \
                  ./lm-out-new-unique_copy/2l1h64d_n10000_len1-50_run0 \
                  ./lm-out-new-unique_copy/2l1h64d_n10000_len1-100_run0 \
    --task unique_copy \
    --output_dir ./hessian_results/many_models \
    --batch_size 200 \
    --max_samples 500
```

**Key arguments:**
- `--model_paths`: Space-separated list of model checkpoint paths
- Other arguments same as `plot_hessian_simple.py`

**Output:**
- `hessian_results/many_models/esd_comparison.pdf`: Overlay ESD plot for all models
- `hessian_results/many_models/top_eigenvalues_comparison.pdf`: Overlay eigenvalue plot for all models

### Single Model, Multiple Datasets Analysis

```bash
python plot_hessian_many_datasets.py \
    --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \
    --task unique_copy \
    --dataset_ranges 1-10 1-20 1-50 1-100 \
    --output_dir ./hessian_results/many_datasets \
    --batch_size 200 \
    --max_samples 500
```

**Key arguments:**
- `--model_path`: Path to model checkpoint directory (trained on one length range)
- `--task`: Task name
- `--dataset_ranges`: Space-separated list of dataset ranges to evaluate (e.g., `1-10 1-20 1-50 1-100`)
  - Default: `1-10 1-20 1-50 1-100`
  - Format: `min-max` for each range
- `--output_dir`: Output directory (default: `./hessian_results/many_datasets`)
- `--batch_size`, `--max_samples`: Same as other scripts
- `--top_k`: Number of top eigenvalues to plot (default: 100)

**Output:**
- `hessian_results/many_datasets/{model_name}/esd_comparison.pdf`: Overlay ESD plot for all dataset ranges
- `hessian_results/many_datasets/{model_name}/top_eigenvalues_comparison.pdf`: Overlay eigenvalue plot

**Use Case:**
Study how model's Hessian spectrum changes when evaluated on datasets of different lengths. Useful for analyzing model robustness and generalization properties.

## Model Path Pattern

The scripts automatically extract training length ranges from model directory names using the pattern:

```
.../{config}_len{min}-{max}_run{N}
```

Examples:
- `2l1h64d_n10000_len1-10_run0` → train length 1-10
- `4l2h256d_n50000_len1-50_run0` → train length 1-50

## Output Directory Structure

### Single Model (`plot_hessian_simple.py`)
```
hessian_results/
└── simple/
    └── 2l1h64d_n10000_len1-10_run0/
        ├── esd_train.pdf
        ├── esd_same_length_gen.pdf
        ├── top_eigenvalues_train.pdf
        └── top_eigenvalues_same_length_gen.pdf
```

### Multiple Models (`plot_hessian_many_models.py`)
```
hessian_results/
└── many_models/
    ├── esd_comparison.pdf
    └── top_eigenvalues_comparison.pdf
```

### Single Model, Multiple Datasets (`plot_hessian_many_datasets.py`)
```
hessian_results/
└── many_datasets/
    └── 2l1h64d_n10000_len1-10_run0/
        ├── esd_comparison.pdf
        └── top_eigenvalues_comparison.pdf
```

## Datasets Analyzed

### plot_hessian_simple.py
- **train**: Training distribution (length range extracted from model path)
- **same_length_gen**: Test distribution with same length range (different seed)
- **length_gen**: Test distribution with doubled length range

### plot_hessian_many_models.py
- **train only**: Focuses on training set to study how training length affects Hessian

### plot_hessian_many_datasets.py
- **Custom dataset ranges**: Each specified range is treated as an independent evaluation dataset
- Completely independent from the model's original training/test ranges

## Implementation Details

### Model Support
- Standard GPT2 models
- NoPE models (no positional embeddings)
- Regularized models (with regularization coefficient)

### Hessian Computation
- Uses PyHessian for eigenvalue computation
- Computes top-k eigenvalues using power iteration (default: top 100)
- Computes eigenvalue spectral density using random probing with n_v=5
- Automatically handles double backward requirements (disables flash attention)
- **ESD Averaging**: For statistical validity, plots mean ESD across all n_v runs (not individual runs)

### Plotting
- Top k eigenvalues plot with markers and legend
- Eigenvalue spectral density (ESD) using log scale
- Color-coded overlay plots for easy comparison
- Configurable output format (PDF, dpi=300)

### Memory Management
- Explicit cleanup after processing each dataset/model
- Deletes dataloaders and models after use
- Forces GPU cache cleanup and Python garbage collection

## Dependencies

- torch
- transformers
- pyhessian (patched version in `../algorithmic/pyhessian_patched/`)
- matplotlib
- numpy

## Notes
````

- Training length range is **required** in the model directory name for automatic extraction
- All analysis uses reproducible random seeds (default: 42 for Hessian, 0/1 for data)
- For large models, consider reducing `--max_samples` and `--top_k` for faster computation
- Hessian computation requires sufficient GPU memory; adjust batch size if needed
