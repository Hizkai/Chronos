# Chronos: Learning Temporal Dynamics of Reasoning Chains for Test-Time Scaling

Official implementation of the paper *"Chronos: Learning Temporal Dynamics of Reasoning Chains for Test-Time Scaling"* (Findings of the 64th Annual Meeting of the Association for Computational Linguistics). 

## Overview

**Chronos** (Chronological Reasoning Scorer) is a lightweight, plug-and-play module that improves Test-Time Scaling (TTS) for large language models. Unlike majority voting or heuristic token-level scoring methods that treat all reasoning trajectories equally, Chronos models each trajectory as a **time series** of token-level probabilities and learns to capture temporal dynamics to estimate trajectory quality.

### Key Features

- 🕐 **Temporal Modeling**: Treats reasoning traces as chronological sequences rather than unordered token statistics, preserving sequential dependencies.
- 🏗️ **Multi-scale Convolutional Architecture**: Employs parallel convolutional filters with varying kernel lengths to capture both local fluctuations and global patterns within reasoning processes.
- ⚡ **Lightweight & Plug-and-Play**: Introduces negligible computational overhead and can be applied to any LLM without retraining the base model.

## Getting Started

### 0. Environment Setup

```bash
conda env create -f environment.yml
conda activate chronos
pip install git+https://github.com/hao-ai-lab/Dynasor.git
```

### 1. Data Sampling

```bash
mkdir sampled_data
bash sample.sh
```

### 2. Data Processing

#### 2.1 Process Data for Training Chronos

```bash
cd process_data
bash process_ranker_data.sh <path_to_sampled_data>
```

#### 2.2 Process Data for Evaluation

```bash
bash process_eval_data.sh <path_to_sampled_data>
```

### 3. Train Chronos

```bash
cd ../scorer
bash train_scorer.sh <input_path> <output_path>
```

The trained model checkpoint and configuration (including scaler parameters) will be saved to the specified output directory.

### 4. Evaluation

Run Chronos-weighted voting on evaluation data and print results:

```bash
cd ..
python evaluate.py --input_path <path_to_eval_data> --model_path ./model_path.json
python print_result.py --input_path <path_to_eval_data>
```

> **Note**: Update `model_path.json` with the actual paths to your trained Chronos scorer checkpoints. Multiple scorers can be specified for ensemble evaluation.

## 📚 Acknowledgments

This project builds upon the following outstanding open-source works:

- [**DeepConf**](https://github.com/facebookresearch/deepconf) — An efficient parallel thinking framework for enhanced reasoning.
- [**InceptionTime**](https://github.com/hfawaz/inceptiontime) — A convolution-based framework for time series classification.

We thank the authors for their contributions.

