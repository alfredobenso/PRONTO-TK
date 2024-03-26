import os
import re
from tkinter import filedialog
import tkinter as tk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from tabulate import tabulate
def analyze_and_plot_validation_results(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract parameters from the filename
    filename = os.path.basename(file_path)
    match = re.match(r'inference_epochs_(\d+)_lr_(\d+\.\d+)_model_(\d+)_(\w+).csv', filename)
    if match:
        epochs, lr, model, type = match.groups()
    else:
        match = re.match(r'inference_FT_epochs_(\d+)_lr_(\d+\.\d+)_model_(\d+)_(\w+).csv', filename)
        if match:
            epochs, lr, model, type = match.groups()
        else:
            epochs, lr, model, type = ("-", "-", "-", "single")
    # Display basic statistics for probabilities of label 0 and 1
    print("Basic Statistics for Probabilities:\n")
    stats = df.groupby('Species')[['0', '1']].describe()
    print(tabulate(stats, headers='keys', tablefmt='fancy_grid'))

    #Print how many rows have the column 1 > 0,75
    print(f"Rows with column 1 > 0.75: {len(df[df['1'] > 0.75])}")

    # Compute additional metrics: accuracy, precision, and recall
    print("\nAccuracy, Precision, and Recall for Each Specie:\n")
    species = df['Species'].unique()
    metrics_list = []
    for specie in species:
        df_specie = df[(df['Species'] == specie) & (df['Label'] != -1)]
        # Assuming label 1 is the positive class and using 0.5 as threshold
        if len(df_specie) == 0:
            continue
        predictions = df_specie['1'] >= 0.5
        accuracy = accuracy_score(df_specie['Label'], predictions)
        precision, recall, _, _ = precision_recall_fscore_support(df_specie['Label'], predictions, average='binary')

        # If in the log file there is the column Entry Name, then execute the following code
        if 'Entry Name' in df_specie.columns and 'Annotation' in df_specie.columns:
            casesCount = ""
            for annotation in ['manual', 'automatic']:
                df_annot = df_specie[df_specie['Annotation'] == annotation]

                # Compute TP
                if len(df_annot[(df_annot['Label'] == 1)]) > 0:
                        casesCount += f"{annotation}: {100 * len(df_annot[(df_annot['Label'] == 1) & (df_annot['1'] >= 0.5)]) / len(df_annot[(df_annot['Label'] == 1)]):.4f} % ({len(df_annot[(df_annot['Label'] == 1) & (df_annot['1'] >= 0.5)])}/{len(df_annot[(df_annot['Label'] == 1)])})\n"

            metrics_list.append({'Specie': specie, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'TP/L_1': casesCount})
        else:
            metrics_list.append({'Specie': specie, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall})

    # Convert list to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    if len(metrics_df) == 0:
        #draw and save a violin plot with the distribution of probabilities for Column '1' for all Species (not separated)
        fig, ax = plt.subplots()
        ax = sns.violinplot(x='Species', y='1', data=df, ax=ax, inner=None, scale="area")
        #add scatter
        sns.stripplot(x="Species", y='1', data=df, ax=ax, color='0.3', jitter=0.3, size=2.5)

        ax.set_title(f'Distribution of Probabilities of Label 1 for All Species')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        file_path = file_path.replace("inferences", "results")
        file_path = file_path.replace(".csv", "_results_with_annotations.csv")
        plt.savefig(file_path.replace(".csv", ".jpg"), bbox_inches='tight')
        plt.show()
    else:
        print(tabulate(metrics_df, headers='keys', tablefmt='fancy_grid'))
        file_path = file_path.replace("inferences", "results")
        file_path = file_path.replace(".csv", "_results_with_annotations.csv")
        metrics_df.to_csv(file_path, index=False)

        # Create a figure with four subplots
        fig, axs = plt.subplots(4, figsize=(len(species) * 4, 20))

        # Plot distributions of probabilities for TP, FP, TN, FN
        for i, label in enumerate(['TP', 'FP', 'TN', 'FN']):
            if label == 'TP':
                df_label = df[(df['Label'] == 1) & (df['1'] >= 0.5)]
                txtLabel = '1'
            elif label == 'FP':
                df_label = df[(df['Label'] == 0) & (df['1'] >= 0.5)]
                txtLabel = '1'
            elif label == 'TN':
                df_label = df[(df['Label'] == 0) & (df['0'] >= 0.5)]
                txtLabel = '0'
            else:  # FN
                df_label = df[(df['Label'] == 1) & (df['0'] >= 0.5)]
                txtLabel = '0'

            # Check if 'Species' column in df_label DataFrame is empty
            if df_label['Species'].empty or not (df_label.groupby('Species').filter(lambda x: len(x) == 0).empty):
                continue

            ax = sns.violinplot(x="Species", y='1', data=df_label, ax=axs[i], inner=None, scale="area")
            sns.stripplot(x="Species", y='1', data=df_label, ax=axs[i], color='0.3', jitter=0.3, size=2.5)
            ax.axhline(0.5, color='red', linestyle='--')  # Add horizontal line at 0.5
            for j in range(len(species)):
                specie = species[j]
                df_specie = df_label[df_label['Species'] == specie]
                mean = df_specie['1'].mean()
                median = df_specie['1'].median()
                if not df_specie['1'].empty:
                    conf_int = np.percentile(df_specie['1'], [2.5, 97.5])
                    conf_int_text = f"CI: {conf_int[0]:.2f}-{conf_int[1]:.2f}"
                else:
                    conf_int = 0
                    conf_int_text = "CI: N/A"
                ax.text(j, mean, f"Average: {mean:.2f}\n{conf_int_text}\nMedian: {median:.2f}\nCount: {len(df_specie)}", color='black', ha='center')
            ax.set_title(f'Distribution of Probabilities of Label {txtLabel} for {label} for Each Specie')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            #ax.get_legend().remove()

        # Create a table with, for each Species, Accuracy, Precision, and Recall
        table_data = metrics_df[['Accuracy', 'Precision', 'Recall']].values.round(4)
        table_rows = metrics_df['Specie']
        table_columns = ['Accuracy', 'Precision', 'Recall']

        base_height_per_row = 0.05
        # Calculate total height based on number of rows
        total_height = base_height_per_row * len(table_rows) + 0.1  # extra space for headers and margins
        # Ensure the total height is within the figure (0 to 1)
        total_height = min(1, max(0, total_height))

        table = axs[3].table(cellText=table_data, rowLabels=table_rows, colLabels=table_columns, loc='center', colWidths=[0.1, 0.1, 0.1, 0.1], cellLoc='center', cellColours=None, bbox=[0.2, -1.8, 0.6, total_height])

        table.auto_set_font_size(False)
        table.set_fontsize(12)

        # Set the overall title for the figure
        #if FT is in the filename, then it is a fine-tuning experiment
        if "_FT_" in filename:
            fig.suptitle(f'Validation Results (after Fine Tuning) - Epochs: {epochs}, Learning Rate: {lr}, Model: {model}, Type: {type}', fontsize=16)
        else:
            fig.suptitle(f'Validation Results - Epochs: {epochs}, Learning Rate: {lr}, Model: {model}, Type: {type}', fontsize=16)

        #set file_name to the same filename substite inference with plot and .jpg
        file_name = filename.replace("inference", "plot").replace(".csv", ".jpg")
        #same path than the loaded file
        folder_path = os.path.dirname(file_path)
        folder_path = folder_path.replace("inferences", "results")

        plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight')

        plt.tight_layout()
        plt.show()

def analyze_validation_results(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Display basic statistics for probabilities of label 0 and 1
    print("Basic Statistics for Probabilities:\n")
    print(df.groupby('Species')[['0', '1']].describe())

    # Compute additional metrics: accuracy, precision, and recall
    print("\nAccuracy, Precision, and Recall for Each Specie:\n")
    species = df['Species'].unique()
    metrics_list = []
    for specie in species:
        df_specie = df[df['Species'] == specie]
        # Assuming label 1 is the positive class and using 0.5 as threshold
        predictions = df_specie['1'] >= 0.5
        accuracy = accuracy_score(df_specie['Label'], predictions)
        precision, recall, _, _ = precision_recall_fscore_support(df_specie['Label'], predictions, average='binary')
        metrics_list.append({'Specie': specie, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall})

    # Convert list to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.to_string(index=False))

    # Extracts the path from the file_path
    folder_path = os.path.dirname(file_path)
    # Extrace the filename from the file_path
    file_name = os.path.basename(file_path)
    #replace the subfolder inferences with the folder "results" and create it if it does not exists
    folder_path = folder_path.replace("inferences", "results")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    #add to file_path ".results.csv" and save it to file
    metrics_df.to_csv(os.path.join(folder_path, file_name.replace(".csv", "_results.csv")), index=False)

    return metrics_df

# Example usage:
# analyze_validation_results('/path/to/your/validation.csv')
def main():
    # Analyze the log file for the experiment with the highest number of epochs
    root = tk.Tk()  # Using the Tkinter instance from PySimpleGUI
    root.withdraw()

    log_file = filedialog.askopenfilename(title='Choose a Validation log file', filetypes=[('CSV files', '*.csv')])

    if not log_file:
        print("No file selected")
        exit()

    results = analyze_and_plot_validation_results(log_file)
    print(results)

if __name__ == "__main__":
    main()
