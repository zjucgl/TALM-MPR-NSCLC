import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculate_dca(y_true, y_probs, thresholds=np.arange(0.01, 1.0, 0.01)):
    net_benefits = []
    n_total = len(y_true)
    for threshold in thresholds:
        preds = (y_probs >= threshold).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        net_benefit = (tp / n_total) - (fp / n_total) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)
    return thresholds, np.array(net_benefits)


def calculate_metrics_with_ci(y_true, y_probs, n_bootstraps=1000, alpha=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    metrics_dist = {"auc": [], "acc": [], "sens": [], "spec": [], "f1": [], "brier": []}

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    optimal_cutoff = thresholds[np.argmax(tpr - fpr)]

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_probs_boot = y_probs[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        preds_boot = (y_probs_boot >= optimal_cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_boot, preds_boot).ravel()

        metrics_dist["auc"].append(roc_auc_score(y_true_boot, y_probs_boot))
        metrics_dist["acc"].append((tp + tn) / n_samples)
        metrics_dist["sens"].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        metrics_dist["spec"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics_dist["f1"].append(f1_score(y_true_boot, preds_boot, zero_division=0))
        metrics_dist["brier"].append(brier_score_loss(y_true_boot, y_probs_boot))

    lower_p = (1.0 - alpha) / 2.0 * 100
    upper_p = (alpha + (1.0 - alpha) / 2.0) * 100

    results = {}
    for key, values in metrics_dist.items():
        if values:
            mean_val = np.mean(values)
            ci_lower = np.percentile(values, lower_p)
            ci_upper = np.percentile(values, upper_p)
            results[key] = f"{mean_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        else:
            results[key] = "N/A"

    return results, brier_score_loss(y_true, y_probs), optimal_cutoff


def format_axis(ax, is_square=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
    ax.tick_params(direction="out", length=6, width=1.2, labelsize=12)
    if is_square:
        ax.set_aspect("equal", adjustable="box")


def plot_performance(y_true, y_scores, brier_score, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 1.2

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    ax_roc = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_est = roc_auc_score(y_true, y_scores)
    ax_roc.plot(fpr, tpr, color="#D32F2F", lw=3.5, label=f"Model Performance (AUC = {auc_est:.3f})")
    ax_roc.plot([-0.05, 1.05], [-0.05, 1.05], color="#757575", lw=1.5, linestyle="--")
    ax_roc.set_xlim([-0.03, 1.03])
    ax_roc.set_ylim([-0.03, 1.03])
    ax_roc.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax_roc.set_title("(A) Receiver Operating Characteristic", loc="left", fontsize=16, fontweight="bold", pad=15)
    ax_roc.legend(loc="lower right", fontsize=13, frameon=False)
    format_axis(ax_roc, is_square=True)

    ax_pr = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    ax_pr.plot(recall, precision, color="#388E3C", lw=3.5, label=f"Model Performance (AP = {ap_score:.3f})")
    ax_pr.set_xlim([-0.03, 1.03])
    ax_pr.set_ylim([-0.03, 1.03])
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=14, fontweight="bold")
    ax_pr.set_ylabel("Precision (PPV)", fontsize=14, fontweight="bold")
    ax_pr.set_title("(B) Precision-Recall Curve", loc="left", fontsize=16, fontweight="bold", pad=15)
    ax_pr.legend(loc="lower left", fontsize=13, frameon=False)
    format_axis(ax_pr, is_square=True)

    ax_cal = axes[1, 0]
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
    ax_cal.plot(
        prob_pred,
        prob_true,
        marker="o",
        color="#7B1FA2",
        lw=2.5,
        markersize=8,
        label=f"Model Performance (Brier = {brier_score:.3f})",
    )
    ax_cal.plot([-0.05, 1.05], [-0.05, 1.05], linestyle="--", color="#757575", label="Perfect Calibration")
    ax_cal.set_xlim([-0.03, 1.03])
    ax_cal.set_ylim([-0.03, 1.03])
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=14, fontweight="bold")
    ax_cal.set_ylabel("Fraction of Positives", fontsize=14, fontweight="bold")
    ax_cal.set_title("(C) Calibration Curve", loc="left", fontsize=16, fontweight="bold", pad=15)
    ax_cal.legend(loc="lower right", fontsize=13, frameon=False)
    format_axis(ax_cal, is_square=True)

    ax_dca = axes[1, 1]
    thresholds, net_benefit = calculate_dca(y_true, y_scores)
    prevalence = np.sum(y_true == 1) / len(y_true)
    net_benefit_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    ax_dca.plot(thresholds, net_benefit_all, color="#9E9E9E", lw=2, linestyle="--", label="Treat All")
    ax_dca.plot(thresholds, np.zeros_like(thresholds), color="black", lw=2, label="Treat None")
    ax_dca.plot(thresholds, net_benefit, color="#F57C00", lw=3.5, label="Model Performance")
    ax_dca.set_xlim([-0.02, 0.62])
    ax_dca.set_ylim([-0.05, max(0.2, np.max(net_benefit) * 1.1)])
    ax_dca.set_xlabel("Threshold Probability", fontsize=14, fontweight="bold")
    ax_dca.set_ylabel("Net Benefit", fontsize=14, fontweight="bold")
    ax_dca.set_title("(D) Decision Curve Analysis", loc="left", fontsize=16, fontweight="bold", pad=15)
    ax_dca.legend(loc="upper right", fontsize=13, frameon=False)
    format_axis(ax_dca)

    plt.tight_layout(pad=4.0)
    pdf_path = output_dir / "Figure_3_Model_Performance.pdf"
    svg_path = output_dir / "Figure_3_Model_Performance.svg"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    return pdf_path, svg_path


def evaluate_from_csv(csv_path, output_dir, n_bootstraps=1000, seed=42, dca_reference_threshold=0.2):
    df = pd.read_csv(csv_path)
    y_true = df["true_label"].values
    y_scores = df["pred_prob"].values

    metrics, brier_score, cutoff = calculate_metrics_with_ci(
        y_true,
        y_scores,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    thresholds, net_benefit = calculate_dca(y_true, y_scores)
    threshold_idx = np.argmin(np.abs(thresholds - dca_reference_threshold))
    reference_net_benefit = net_benefit[threshold_idx]

    print("Full cohort metrics")
    print(f"Optimal cutoff (Youden index): {cutoff:.3f}")
    print(f"AUC: {metrics['auc']}")
    print(f"Accuracy: {metrics['acc']}")
    print(f"Sensitivity: {metrics['sens']}")
    print(f"Specificity: {metrics['spec']}")
    print(f"F1: {metrics['f1']}")
    print(f"Brier score: {brier_score:.3f}")
    print(f"DCA net benefit @{dca_reference_threshold:.2f}: {reference_net_benefit:.3f}")

    pdf_path, svg_path = plot_performance(y_true, y_scores, brier_score, output_dir)
    print(f"Saved figures to {pdf_path} and {svg_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OOF predictions from a CSV file.")
    parser.add_argument("--input", default="outputs/pooled_oof_aligned.csv", help="Path to the OOF prediction CSV.")
    parser.add_argument("--output-dir", default="outputs/figures", help="Directory for generated figures.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap resampling.")
    parser.add_argument("--dca-threshold", type=float, default=0.2, help="Reference DCA threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_from_csv(
        csv_path=Path(args.input),
        output_dir=Path(args.output_dir),
        n_bootstraps=args.bootstrap_samples,
        seed=args.seed,
        dca_reference_threshold=args.dca_threshold,
    )
