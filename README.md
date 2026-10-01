# crazytesi

Project for evaluating a text classification model for detecting toxic and offensive content on HateXplain data, comparing the probabilities assigned by the Laya model with the distribution of human consensus.

## Objective

The repository contains a complete workflow for:

- downloading and preprocessing a dataset of comments with human annotations;
- building a balanced subset of toxic and non-toxic examples;
- evaluating the Laya model on the toxic detection task;
- analyzing performance metrics and agreement with human probabilities.

The project is aimed at measuring the actual quality of the model on an annotated dataset, not just generating isolated predictions.

## Repository structure

- `prepare_hatexplain.py`: downloads the original HateXplain dataset, extracts annotator labels, and creates a balanced sample of 2,000 examples.
- `run_laya_hatexplain.py`: runs classification with `laya.Router(default="english")` on the prepared subset and saves predictions in raw and tabular format.
- `analyze.py`: loads the results, computes accuracy, ROC-AUC, Brier score, log loss, ECE, and analyzes correlation with human probabilities.
- `data/`: original data and preprocessed dataset.
- `results/`: model outputs and analysis plots.

## Pipeline

1. Dataset preparation
   - download HateXplain;
   - convert labels to binary values (`toxic` = 1, `normal` = 0);
   - build a sample of 1,000 toxic examples and 1,000 non-toxic examples;
   - randomize and generate unique `scenario_id` values.

2. Model execution
   - send each text to `laya` with a question about toxic content;
   - store the toxicity probability and binary prediction;
   - save the raw model response in `results/hatexplain_laya_raw.jsonl`.

3. Metric analysis
   - compare `ground_truth` with `laya_probability`;
   - compute correctness, separability, and distance from expected probabilities;
   - compare against the average human probability.

## Quick start

Requirements:

- Python 3.10+
- packages: `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `laya`

Run the steps in order:

```bash
python prepare_hatexplain.py
python run_laya_hatexplain.py
python analyze.py
```

## Main outputs

Results are saved in:

- `data/hatexplain_2000.csv`
- `results/hatexplain_laya_results.csv`
- `results/hatexplain_laya_raw.jsonl`
- `results/analysis/hatexplain_beta_fit.png`

## Current results

The values below are from the latest execution of `python analyze.py` on the current dataset.

| Metric | Value | Interpretation |
| --- | ---: | --- |
| N | 2000 | Total number of evaluated examples |
| Accuracy | 0.7150 | 71.5% correct classifications |
| ROC AUC | 0.7827 | Good separation between classes |
| Brier score | 0.1921 | Predicted probabilities are reasonably close to the true outcome |
| Log loss | 0.5666 | Penalization of incorrect probabilities |
| ECE | 0.0372 | Good overall calibration |
| Pearson r | 0.5165 | Moderate positive correlation with human judgment |
| Spearman ρ | 0.5128 | Moderate positive rank correlation |

### Short interpretation

- The model shows useful ability to distinguish toxic from non-toxic text, with `ROC AUC = 0.7827`.
- Overall accuracy of `0.7150` is positive but still leaves room for improvement.
- Calibration is good (`ECE = 0.0372`), so the produced probabilities are not strongly misaligned with observations.
- Correlation with human probabilities is moderate (`r ≈ 0.52`), indicating the model follows human consensus but does not match it perfectly.

### Beta fit of probabilities

The probability distribution was compared against a separate Beta model for each class:

- Toxic: `alpha = 4.455884`, `beta = 3.362935`, `mean = 0.574569`
- Non-toxic: `alpha = 2.106193`, `beta = 3.568812`, `mean = 0.371926`

The fact that the toxic group has a higher mean confirms that the model assigns consistently higher probabilities to genuinely offensive texts.

## Considerations

This repo is a solid baseline for reproducible toxic detection analysis and evaluation. For now, the work focuses on:

- dataset construction;
- model execution;
- quantitative performance evaluation.

Possible next steps include:

- comparison with other models or prompts;
- per-class and per-sample analysis;
- evaluation on a larger or more realistic dataset;
- robustness testing for offensive language, sarcasm, and ambiguous contexts.

## Current status

The repository is in a working baseline phase: dataset prepared, predictions generated, analysis completed, and results stored in `results/`.

---

This README was updated based on the repository structure and the verified results from the execution of `python analyze.py`.
