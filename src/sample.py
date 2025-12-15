import torch
from src.xai import lime_explain, integrated_gradients, shap_explain
from src.helper import plot_xai_bar
import numpy as np
import pandas as pd

def run_xai_for_sample(i, model, Xte, Xtr, features, probs, threshold):
    x = Xte[i:i+1]
    prob = probs[i].item()
    pred = int(prob > threshold)

    def model_fn(x):
        with torch.no_grad():
            return torch.sigmoid(
                model(torch.tensor(x).float())
            ).numpy()

    lime = lime_explain(model_fn, x)
    ig = integrated_gradients(model, torch.tensor(x).float()).numpy()[0]
    shapv = shap_explain(model, Xtr[:100], x)[0]

    return {
        "sample": i,
        "pred_prob": prob,
        "pred_label": pred,
        "lime": lime,
        "ig": ig,
        "shap": shapv
    }

def run_xai_analysis(model, Xte, Xtr, yte, features, probs, threshold, n_samples=5):
    rows = []

    for i in range(n_samples):
        res = run_xai_for_sample(
            i, model, Xte, Xtr, features, probs, threshold
        )

        lime = res["lime"]
        ig   = res["ig"]
        shapv = res["shap"]

        prob = res["pred_prob"]
        pred = res["pred_label"]

        # TOP-K features
        lime_top = np.argsort(np.abs(lime))[-3:][::-1]
        ig_top   = np.argsort(np.abs(ig))[-3:][::-1]
        shap_top = np.argsort(np.abs(shapv))[-3:][::-1]

        # print results
        print(f"\nSample {i}")
        print(f" True label     : {int(yte[i])}")
        print(f" Pred prob      : {prob:.3f}")
        print(f" Pred label     : {pred}")

        print("  LIME top features:")
        for idx in lime_top:
            print(f"   - {features[idx]} : {lime[idx]:.4f}")

        print("  IG top features:")
        for idx in ig_top:
            print(f"   - {features[idx]} : {ig[idx]:.4f}")

        print("  SHAP top features:")
        for idx in shap_top:
            print(f"   - {features[idx]} : {shapv[idx]:.4f}")


        # save plots
        plot_xai_bar(res["lime"], features,
                    f"LIME (Sample {i})", f"figures/lime_{i}.png")
        plot_xai_bar(res["ig"], features,
                    f"IG (Sample {i})", f"figures/ig_{i}.png")
        plot_xai_bar(res["shap"], features,
                    f"SHAP (Sample {i})", f"figures/shap_{i}.png")

        rows.append({
            "sample": i,
            "true_label": int(yte[i]),
            "pred_prob": round(res["pred_prob"], 3),
            "pred_label": res["pred_label"],
            "lime_top": features[np.argmax(np.abs(res["lime"]))],
            "ig_top": features[np.argmax(np.abs(res["ig"]))],
            "shap_top": features[np.argmax(np.abs(res["shap"]))],
        })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/xai_global_summary.csv", index=False)
    return df
