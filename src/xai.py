
import torch
import shap
import numpy as np
from sklearn.linear_model import LinearRegression

def shap_explain(model, X_bg, X):
    def model_fn(x):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(x).float())).numpy()
    explainer = shap.Explainer(model_fn, X_bg)
    return explainer(X).values

def lime_explain(model_fn, x, n_samples=800, noise=0.1):
    samples = x + noise * np.random.randn(n_samples, x.shape[1])
    preds = model_fn(samples)
    lr = LinearRegression().fit(samples, preds)
    return lr.coef_

def integrated_gradients(model, x, steps=50):
    baseline = torch.zeros_like(x)
    grads = []
    for a in torch.linspace(0, 1, steps):
        xi = baseline + a * (x - baseline)
        xi.requires_grad_(True)
        out = model(xi)
        out.backward()
        grads.append(xi.grad.detach())
    return torch.mean(torch.stack(grads), dim=0) * (x - baseline)
