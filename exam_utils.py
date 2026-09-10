import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_feature_ranges(df, columns, title='Feature ranges'):
    summary = df[columns].agg(['min', 'max']).T
    summary.columns = ['min', 'max']
    print(title)
    display(summary)

def plot_regression_dataset_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(df['area_m2'], df['price'], alpha=0.7)
    axes[0].set_xlabel('Area (m²)')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Area vs. Price')

    axes[1].scatter(df['distance_to_center_km'], df['price'], alpha=0.7)
    axes[1].set_xlabel('Distance to Center (km)')
    axes[1].set_ylabel('Price')
    axes[1].set_title('Distance to Center vs. Price')

    plt.tight_layout()
    plt.show()

def plot_learning_rate_histories(histories, title='Learning Rate Comparison'):
    plt.figure(figsize=(8, 5))
    for learning_rate, history in histories.items():
        history = np.asarray(history)
        plt.plot(history, label=f'learning_rate={learning_rate}')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

def plot_model_selection_scores(x_values, scores, xlabel, ylabel, title):
    x_values = np.asarray(x_values)
    scores = np.asarray(scores)
    plt.figure(figsize=(7, 4))
    plt.plot(x_values, scores, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.show()

def plot_classification_dataset_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    counts = df['churn'].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values)
    axes[0].set_xlabel('Churn Class')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Class Distribution')

    for label in [0, 1]:
        subset = df[df['churn'] == label]
        axes[1].scatter(
            subset['months_as_customer'],
            subset['support_calls'],
            alpha=0.65,
            label=f'class {label}'
        )
    axes[1].set_xlabel('Months as Customer')
    axes[1].set_ylabel('Support Calls')
    axes[1].set_title('Two Features by Class')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
