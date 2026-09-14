import os
import matplotlib.pyplot as plt

def plot_series(values, path, ylabel, title, drift_points=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 4)); plt.plot(values); [plt.axvline(point, color="red", alpha=.4) for point in drift_points]
    plt.xlabel("Episode"); plt.ylabel(ylabel); plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
