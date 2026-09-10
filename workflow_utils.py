import matplotlib.pyplot as plt

def show_regression_workflow():
    plt.figure(figsize=(10, 2))
    plt.text(
        0.5, 0.5,
        "Dataset → Split → Train (raw) → Scale (fit train) → Train scaled → Test eval → CV",
        ha="center",
        va="center",
        fontsize=11
    )
    plt.axis("off")
    plt.show()

def show_knn_workflow():
    plt.figure(figsize=(10, 2))
    plt.text(
        0.5, 0.5,
        "Split → Scale → Distance → Neighbors → Probability → Prediction → Evaluation",
        ha="center",
        va="center",
        fontsize=12
    )
    plt.axis("off")
    plt.show()
