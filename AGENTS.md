# Repository Guidelines

## Project Structure & Module Organization

This repository implements an encoder-decoder Transformer in PyTorch. `models/model/` contains the top-level model, encoder, and decoder; `models/blocks/`, `models/layers/`, and `models/embedding/` hold reusable components. Training and dataset setup live in `train.py`, `data.py`, and `util/`. Hyperparameters and device selection are in `conf.py`. `image/` contains README diagrams, `paper/` contains the reference paper, and `saved/` contains example outputs. Runtime data and virtual environments are excluded by `.gitignore`.

## Build, Test, and Development Commands

There is no build configuration or automated test suite in the repository. Use Python with compatible PyTorch and legacy torchtext dependencies installed.

- `python train.py` — load Multi30k, initialize the model, and run training; the dataset may be downloaded on first use.
- `python -m compileall models util data.py conf.py train.py` — check Python files for syntax errors without starting training.

Adjust model and training settings in `conf.py` before a run. Training can take a long time and requires the dataset and substantial compute for the default configuration.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Keep model components in their corresponding `models/` subpackages and use descriptive names such as `MultiHeadAttention`. Preserve tensor shape comments where they clarify attention or sequence transformations. No formatter or linter is configured; keep changes focused and consistent with adjacent files.

## Testing Guidelines

No test directory, test framework, or coverage target is currently defined. For changes, run the compile check above. When modifying model behavior, also run a small forward-pass or training check where practical, and describe the command and result in the pull request. Avoid launching a full 1,000-epoch training run as a routine check.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries (for example, `remove .idea`, `fix typos`, and `Update scale_dot_product_attention.py`); keep commit subjects concise and specific. Pull requests should explain the problem and change, identify affected model or training behavior, link related issues when available, and include relevant command results. Add screenshots only when changing visual assets or plots.

## Data and Configuration

`data.py` invokes torchtext's legacy Multi30k loader for English-German data. Keep downloaded datasets, checkpoints, and generated results out of commits; `.gitignore` already excludes common local artifacts. Do not commit credentials or machine-specific paths.
