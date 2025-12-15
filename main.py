import argparse
import pandas as pd
from src.sample import run_xai_analysis
from src.data import prepare_data
from src.mlp import train_model
from src.metrics import evaluate
from src.helper import tune_threshold
from src.helper import plot_confusion_matrix_heatmap

def main(n_samples):
    Xtr, Xte, ytr, yte, features = prepare_data()
    model = train_model(Xtr, ytr)
    threshold, probs = tune_threshold(model, Xte, yte)

    metrics = evaluate(yte, (probs > threshold).astype(int))
    
    print("=== MODEL PERFORMANCE (TEST SET) ===")
    print("Accuracy:", round(metrics["accuracy"], 3))
    print("F1-score:", round(metrics["f1"], 3))
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("="*50)
    print()

    plot_confusion_matrix_heatmap(
        y_true=yte,
        y_pred=(probs > threshold).astype(int),
        class_names=["Non-Diabetes", "Diabetes"],
        filename="figures/confusion_matrix.png"
    )

    df = run_xai_analysis(
        model, Xte, Xtr, yte, features, probs, threshold, n_samples
    )
    print("=== SUMMARY ===")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    main(args.n_samples)
