# Heart Disease Prediction

Small learning project. The notebook `heart_disease.ipynb` trains logistic regression.

Data: Heart disease data (heart_disease_data.csv), 13 clinical features, binary target. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares logistic regression, random forest and HistGradientBoosting with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (918 samples, 13 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | accuracy | f1 | roc_auc |
|---|---|---|---|
| HistGradientBoosting | 0.9281 | 0.9339 | 0.9687 |
| Random Forest (200 Trees) | 0.9227 | 0.9295 | 0.9637 |
| Logistic Regression (L2) | 0.8682 | 0.8796 | 0.9329 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `heart_disease.ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
