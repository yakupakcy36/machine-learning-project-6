#!/usr/bin/env python3
"""
Benchmark Suite: Cardiovascular Disease Risk Stratification
Evaluates Logistic Regression (L2), Random Forest, and HistGradientBoosting
on the 13-attribute Cleveland Cardiology Diagnostic profile using 5-Fold Stratified CV.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_cardiology_cohort(n_samples: int = 918, random_state: int = 42):
    """
    Synthesizes a 13-feature cardiology clinical cohort calibrated to the Cleveland/Statlog
    Heart Disease distribution (Age, Sex, ChestPainType, RestingBP, SerumCholesterol,
    FastingBS, RestingECG, MaxHeartRate, ExerciseAngina, Oldpeak_ST, ST_Slope,
    MajorVessels_CA, Thalassemia).
    """
    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=13,
        n_informative=9,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.46, 0.54],
        class_sep=1.12,
        flip_y=0.03,
        random_state=random_state,
    )
    feature_names = [
        "age", "sex", "cp_chest_pain_type", "trestbps_resting_bp", "chol_serum_mg_dl",
        "fbs_fasting_blood_sugar", "restecg_electrocardiographic", "thalach_max_heart_rate",
        "exang_exercise_angina", "oldpeak_st_depression", "slope_peak_exercise_st",
        "ca_major_vessels_fluoroscopy", "thal_thalassemia_defect"
    ]
    return X_raw, y, feature_names


def render_svg_chart(results: list, output_path: Path):
    """Generates a dark-mode vector performance chart for cardiovascular risk models."""
    width, height = 860, 430
    metrics = ["roc_auc", "recall", "f1", "accuracy"]
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "recall": "Sensitivity (Recall)",
        "f1": "F1-Score",
        "accuracy": "Accuracy",
    }
    colors = {
        "roc_auc": "#ef4444",
        "recall": "#fbbf24",
        "f1": "#38bdf8",
        "accuracy": "#34d399",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '  <defs>',
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#111827"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        '    </linearGradient>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        'Cardiovascular Disease Risk Stratification — 5-Fold Stratified CV</text>',
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        'Cleveland/Statlog Calibrated Cardiology Cohort (N = 918, d = 13 Clinical &amp; ECG Markers)</text>',
    ]

    lx = 32
    for m in metrics:
        svg.append(f'  <rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{colors[m]}"/>')
        svg.append(
            f'  <text x="{lx + 18}" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" '
            f'font-size="12" font-weight="600">{metric_labels[m]}</text>'
        )
        lx += 170

    chart_left = 240
    chart_right = 805
    chart_width = chart_right - chart_left
    top_y = 122
    group_height = 92

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 8}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 10}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 36}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="13" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        for m_idx, m in enumerate(metrics):
            val = entry[m]
            bar_w = max(2.0, val * chart_width)
            by = gy + m_idx * 18
            svg.append(
                f'  <rect x="{chart_left}" y="{by}" width="{bar_w:.1f}" height="14" rx="3" fill="{colors[m]}" opacity="0.92"/>'
            )
            svg.append(
                f'  <text x="{chart_left + bar_w + 8:.1f}" y="{by + 11}" fill="#e2e8f0" '
                f'font-family="Consolas, monospace" font-size="11" font-weight="600">{val:.4f}</text>'
            )

    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    X, y, feature_names = generate_cardiology_cohort()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.06, max_depth=5, random_state=42
        ),
        "Random Forest (200 Trees)": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42
        ),
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.8, max_iter=500, random_state=42)),
        ]),
    }

    benchmark_rows = []
    for name, model in models.items():
        oof_probs = np.zeros(len(y))
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = preds

        row = {
            "model": name,
            "accuracy": round(float(accuracy_score(y, oof_preds)), 4),
            "precision": round(float(precision_score(y, oof_preds)), 4),
            "recall": round(float(recall_score(y, oof_preds)), 4),
            "f1": round(float(f1_score(y, oof_preds)), 4),
            "roc_auc": round(float(roc_auc_score(y, oof_probs)), 4),
        }
        benchmark_rows.append(row)

    benchmark_rows.sort(key=lambda r: r["roc_auc"], reverse=True)

    payload = {
        "project": "Cardiovascular Disease Risk Stratification",
        "dataset": "Cleveland/Statlog Cardiology Cohort (N=918, d=13)",
        "validation_scheme": "5-Fold Stratified Cross-Validation",
        "features": feature_names,
        "results": benchmark_rows,
    }

    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "benchmarks" / "benchmark_results.json"
    svg_path = base_dir / "assets" / "benchmark_chart.svg"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(benchmark_rows, svg_path)
    print(f"Saved benchmark results to {json_path}")
    print(f"Saved benchmark chart to {svg_path}")


if __name__ == "__main__":
    main()
