import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_predictions(true_values, predictions):
    # Reshape data to long format
    num_samples, num_selections = true_values.shape
    selection_ids = list(range(num_selections))
    
    true_df = pd.DataFrame(true_values, columns=[f"Selection {i}" for i in selection_ids])
    pred_df = pd.DataFrame(predictions, columns=[f"Selection {i}" for i in selection_ids])
    
    true_df_melted = true_df.melt(var_name='Selection', value_name='True Value')
    pred_df_melted = pred_df.melt(var_name='Selection', value_name='Predicted Value')
    
    merged_df = pd.concat([true_df_melted, pred_df_melted['Predicted Value']], axis=1)
    
    # Create FacetGrid plot
    g = sns.FacetGrid(merged_df, col="Selection", col_wrap=num_selections, height=3, sharey=False)
    g.map(plt.plot, "True Value", color='blue')
    g.map(plt.plot, "Predicted Value", color='red')
    g.add_legend()
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Predictions vs True Values for Each Selection")
    plt.show()

def plot_residuals(true_values, predictions):
    """
    Plot residuals for each selection in a grid using Seaborn's FacetGrid.
    """
    residuals = predictions - true_values
    num_selections = true_values.shape[1]
    
    # Reshape to long format
    residuals_df = pd.DataFrame(residuals, columns=[f"Selection {i}" for i in range(num_selections)])
    true_df = pd.DataFrame(true_values, columns=[f"Selection {i}" for i in range(num_selections)])
    
    residuals_melted = residuals_df.melt(var_name='Selection', value_name='Residuals')
    true_values_melted = true_df.melt(var_name='Selection', value_name='True Values')
    
    merged_df = pd.concat([true_values_melted, residuals_melted['Residuals']], axis=1)

    # Create FacetGrid plot
    g = sns.FacetGrid(merged_df, col="Selection", col_wrap=num_selections, height=3, sharey=False)
    g.map(sns.scatterplot, "True Values", "Residuals", alpha=0.5)
    g.map(plt.axhline, y=0, color='red', ls='--')
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Residuals for Each Selection")
    plt.show()

def plot_residual_histogram(true_values, predictions, bins=20):
    """
    Plot a histogram of residuals for each selection in a grid using Seaborn's FacetGrid.
    """
    residuals = predictions - true_values
    num_selections = true_values.shape[1]
    
    # Reshape to long format
    residuals_df = pd.DataFrame(residuals, columns=[f"Selection {i}" for i in range(num_selections)])
    residuals_melted = residuals_df.melt(var_name='Selection', value_name='Residuals')
    
    # Create FacetGrid plot
    g = sns.FacetGrid(residuals_melted, col="Selection", col_wrap=num_selections, height=3)
    g.map(plt.hist, "Residuals", bins=bins, edgecolor='k', density=True)
    g.map(sns.kdeplot, "Residuals", color='r')
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Histogram of Residuals for Each Selection")
    plt.show()

def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for sequences, _ in loader:
            outputs = model(sequences)
            all_predictions.extend(outputs.numpy())
    return np.array(all_predictions)
