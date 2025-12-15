import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def tune_threshold(model, Xte, yte):
    probs = torch.sigmoid(
        model(torch.tensor(Xte, dtype=torch.float32))
    ).detach().numpy()

    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.2, 0.8, 61):
        preds = (probs > t).astype(int)
        f1 = f1_score(yte, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t, probs


def plot_xai_bar(values, feature_names, title, filename, top_k=6):
    idx = np.argsort(np.abs(values))[-top_k:]
    plt.figure(figsize=(7,4))
    plt.barh(
        [feature_names[i] for i in idx],
        values[idx]
    )
    plt.title(title)
    plt.xlabel("Attribution value")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, filename=None, cmap="Blues"):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    im = plt.imshow(cm, cmap=cmap)

    plt.title("Confusion Matrix", fontsize=12)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30)
    plt.yticks(tick_marks, class_names)

    # Tulis angka di sel
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=11,
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300) 
