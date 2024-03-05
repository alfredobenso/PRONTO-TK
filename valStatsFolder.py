import os
from tkinter import filedialog
import tkinter as tk
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
import seaborn as sns

'''
This script is used to analyze the validation results of a folder with CSV files. 
It computes basic statistics for the probabilities of label 0 and 1, and additional metrics 
such as accuracy, precision, and recall for each specie. 
It also plots the distribution of probabilities for label 1 for each specie 
and the table with the metrics for each specie. 
It also creates a 3D scatter plot of accuracy, precision, and recall for all the experiments in the folder.
'''
def compute_validation_metrics(file_path):
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

    return metrics_df

'''
This function is used to analyze the validation results of a DataFrame and 
plot the distribution of probabilities for label 1 for each specie.
'''
def analyze_and_plot_validation_results(logger, df, folder_path, file_name):

    # Display basic statistics for probabilities of label 0 and 1
    print("Basic Statistics for Probabilities:\n")
    stats = df.groupby('Species')[['0', '1']].describe()
    print(tabulate(stats, headers='keys', tablefmt='fancy_grid'))

    #Print how many rows have the column 1 > 0,75
    logger.log_message(f"Rows with column 1 > 0.75: {len(df[df['1'] > 0.75])}")

    # Compute additional metrics: accuracy, precision, and recall
    print("Accuracy, Precision, and Recall for Each Specie:")
    species = df['Species'].unique().tolist()
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
    folder_path = folder_path.replace("inferences", "results")
    # remove from filename the string _exclude_<int>
    file_name = re.sub(r'_exclude_\d+', '', file_name)

    if len(metrics_df) == 0:
        #draw and save a violin plot with the distribution of probabilities for Column '1' for all Species (not separated)
        fig, ax = plt.subplots()
        ax = sns.violinplot(x='Species', y='1', data=df, ax=ax, inner=None)
        #add scatter
        sns.stripplot(x="Species", y='1', data=df, ax=ax, color='0.3', jitter=0.3, size=2.5)
        ax.set_title(f'Distribution of Probabilities of Label 1 for All Species')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        file_name = file_name.replace(".csv", "_results_with_annotations.csv")
        file_name = file_name.replace(".csv", ".jpg")

        plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight')
        plt.show()
    else:
        print(tabulate(metrics_df, headers='keys', tablefmt='fancy_grid'))
        file_name = file_name.replace(".csv", "_results_with_annotations.csv")
        metrics_df.to_csv(os.path.join(folder_path, file_name), index=False)

        # Create a figure with four subplots
        fig, axs = plt.subplots(4, figsize=(len(species) * 4, 20))

        # Plot distributions of probabilities for TP, FP, TN, FN
        labels = ['TP', 'FP', 'TN', 'FN']
        txt_labels = ['1', '1', '0', '0']

        for i, (label, txt_label) in enumerate(zip(labels, txt_labels)):
            # Define filtering conditions for each label
            if label == 'TP':
                condition = (df['Label'] == 1) & (df['1'] >= 0.5)
            elif label == 'FP':
                condition = (df['Label'] == 0) & (df['1'] >= 0.5)
            elif label == 'TN':
                condition = (df['Label'] == 0) & (df['0'] >= 0.5)
            else:  # FN
                condition = (df['Label'] == 1) & (df['0'] >= 0.5)

            # Filter the dataframe based on the condition
            df_label = df[condition]

            # Check if 'Species' column is empty or has no non-empty groups
            if df_label['Species'].empty or df_label.groupby('Species').filter(lambda x: len(x) != 0).empty:
                continue  # Skip if empty or only contains empty groups

            # For each species
            if label == 'FN':
                for specie in species:
                    if df_label[df_label['Species'] == specie].empty:
                        # Use concat to add a row with 0.2 and 0.8 for 1 and 0, respectively
                        new_row = pd.DataFrame({'Species': [specie], '1': [0.2], '0': [0.8], 'Entry Name': [False]})
                        df_label = pd.concat([df_label, new_row], ignore_index=True)

            # Plot violin plot, stripplot, and other elements
            ax = sns.violinplot(x="Species", y='1', data=df_label, ax=axs[i], inner=None)
            sns.stripplot(x="Species", y='1', data=df_label, ax=axs[i], color='0.3', jitter=0.3, size=2.5)
            ax.axhline(0.5, color='red', linestyle='--')  # Add horizontal line at 0.5

            # Calculate and format data for annotations
            for j, specie in enumerate(species):
                df_specie = df_label[df_label['Species'] == specie]
                mean = df_specie['1'].mean() if not df_specie['1'].empty else 0.2  # Handle empty series
                median = df_specie['1'].median() if not df_specie['1'].empty else 0.2
                if not df_specie['1'].empty:
                    conf_int = np.percentile(df_specie['1'], [2.5, 97.5])
                    conf_int_text = f"CI: {conf_int[0]:.2f}-{conf_int[1]:.2f}"
                else:
                    conf_int = 0
                    conf_int_text = "CI: N/A"

                text = f"Average: {mean:.2f}\n{conf_int_text}\nMedian: {median:.2f}\nCount: {len(df_specie)}"
                ax.text(j, mean, text, color='black', ha='center')

            ax.set_title(f"Distribution of Probabilities of Label {txt_label} for {label} for Each Specie")
            ax.set_xticklabels(species, rotation=45)

        # Create a table with, for each Species, Accuracy, Precision, and Recall
        table_data = metrics_df[['Accuracy', 'Precision', 'Recall']].values.round(4)
        table_rows = metrics_df['Specie']
        table_columns = ['Accuracy', 'Precision', 'Recall']

        base_height_per_row = 0.15
        # Calculate total height based on number of rows
        total_height = base_height_per_row * len(table_rows) + 0.1  # extra space for headers and margins
        # Ensure the total height is within the figure (0 to 1)
        total_height = min(1, max(0, total_height))

        table = axs[3].table(cellText=table_data, rowLabels=table_rows, colLabels=table_columns, loc='center', colWidths=[0.1, 0.1, 0.1, 0.1], cellLoc='center', cellColours=None, bbox=[0, -1.8, 1, total_height])

        table.auto_set_font_size(False)
        table.set_fontsize(12)

        # Set the overall title for the figure
        #if FT is in the filename, then it is a fine-tuning experiment
        if "_FT_" in file_name:
            fig.suptitle(f'Validation Results (after Fine Tuning)\nData: {file_name}', fontsize=16)
        else:
            fig.suptitle(f'Validation Results\nData: {file_name}', fontsize=16)

        #set file_name to the same filename substite inference with plot and .jpg
        file_name = file_name.replace("inference", "plot").replace(".csv", ".jpg")
        #same path than the loaded file
        plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight')

        plt.tight_layout()
        plt.show()

'''
This function is used to analyze the validation results of a folder with CSV files.
'''
def analyseValidationFolder(logger, folder_path):

    # Initialize a list to store the results
    all_results_df = pd.DataFrame(columns=['FileName', 'Origin', 'Specie', 'Accuracy', 'Precision', 'Recall', 'Epochs', 'Learning Rate', 'Model', 'Batch', 'Type'])

    # Loop over all CSV files in the selected folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and not(file_name.endswith('results.csv')):
            # Extract the number of epochs, learning rate, and model used from the file name
            #match = re.match(r'inference_epochs_(\d+)_lr_(\d+\.\d+)_model_(\d+)_(\w+).csv', file_name)
            match = re.match(r'inference_epochs_(\d+)_lr_(\d+\.\d+)_model_(\d+)_batch_(\d+)_(\w+)?\.csv', file_name)            #match = re.match(r'inference_epochs_(\d+)_lr_(\d+\.\d+)_model_(\d+)\.csv', file_name)
            if match:
                epochs, lr, model, batch, type = match.groups()
            else:
                match = re.match(
                    r'inference_FT_origin_(.+)_epochs_(\d+)_lr_(\d+\.\d+)_model_(\d+)_batch_(\d+)_([-\w]+)?\.csv',
                    file_name)
                if match:
                    origin, epochs, lr, model, batch, type = match.groups()

            if match:
                # Compute the accuracy, precision, and recall for the current CSV file
                metrics_df = compute_validation_metrics(os.path.join(folder_path, file_name))

                #Add to metrics_df the columns Epochs, Learning Rate, Model
                #remove the last "-<int> from the origin. Origin may have many dashes inside
                metrics_df['Origin'] = re.sub(r'-\d+$', '', origin)
                metrics_df['Epochs'] = epochs
                metrics_df['Learning Rate'] = lr
                metrics_df['Batch'] = batch
                metrics_df['Type'] = type
                metrics_df['Model'] = model
                metrics_df['FileName'] = file_name

                # Append the metrics for the current CSV file to all_results_df
                all_results_df.loc[len(all_results_df)] = metrics_df.iloc[0]

    all_results_df['Accuracy'] = pd.to_numeric(all_results_df['Accuracy'], errors='coerce')
    all_results_df['Precision'] = pd.to_numeric(all_results_df['Precision'], errors='coerce')
    all_results_df['Recall'] = pd.to_numeric(all_results_df['Recall'], errors='coerce')

    # After the loop, all_results_df will have the results for each species from each file and for each origin
    # You can then plot these results

    #Now I want to add new rows to the df with Origin, Accuracy, Precision, Recall, Epochs, Learning Rate, Model, Batch grouped by Origin. For each Origin, EPOCHS, LR, Model, Batch, I want the mean of Accuracy, Precision, Recall. In these rows, Species will be set to "all"
    #for each Origin, EPOCHS, LR, Model, Batch, compute the mean of Accuracy, Precision, Recall and append a new row in all_results_df with Species="all"
    for origin in all_results_df['Origin'].unique():
        for epochs in all_results_df['Epochs'].unique():
            for lr in all_results_df['Learning Rate'].unique():
                for model in all_results_df['Model'].unique():
                    for batch in all_results_df['Batch'].unique():
                        #compute the mean of Accuracy, Precision, Recall for the current Origin, EPOCHS, LR, Model, Batch
                        mean_accuracy = all_results_df[(all_results_df['Origin'] == origin) & (all_results_df['Epochs'] == epochs) & (all_results_df['Learning Rate'] == lr) & (all_results_df['Model'] == model) & (all_results_df['Batch'] == batch)]['Accuracy'].mean()
                        mean_precision = all_results_df[(all_results_df['Origin'] == origin) & (all_results_df['Epochs'] == epochs) & (all_results_df['Learning Rate'] == lr) & (all_results_df['Model'] == model) & (all_results_df['Batch'] == batch)]['Precision'].mean()
                        mean_recall = all_results_df[(all_results_df['Origin'] == origin) & (all_results_df['Epochs'] == epochs) & (all_results_df['Learning Rate'] == lr) & (all_results_df['Model'] == model) & (all_results_df['Batch'] == batch)]['Recall'].mean()
                        #append a new row in all_results_df with Species="all"
                        all_results_df.loc[len(all_results_df)] = [None, origin, "all", mean_accuracy, mean_precision, mean_recall, epochs, lr, model, batch, "ALL"]

    #From the rows that have Species == all, compute the best row in terms of Accuracy, Precision, and Recall
    best_index = all_results_df[all_results_df['Specie'] == 'all'][['Accuracy', 'Precision', 'Recall']].mean(axis=1).idxmax()

    # Get the row at this index
    best_row = all_results_df.loc[best_index]

    # Extract the values of Epochs, Learning Rate, and Model from this row
    best_epochs = best_row['Epochs']
    best_learning_rate = best_row['Learning Rate']
    best_model = best_row['Model']
    best_batch = best_row['Batch']
    best_origin = best_row['Origin']
    # Print the best combination
    logger.log_message(f'Best combination: Origin={best_origin} Epochs={best_epochs}, Learning Rate={best_learning_rate}, Batch={best_batch}, Model={best_model}')

    # For each row with Species = all
    for index, row in all_results_df[all_results_df['Specie'] == 'all'].iterrows():
        # Get the filename of all the files with the same combination of Origin, Epochs, Learning Rate, Model, Batch
        files = all_results_df[(all_results_df['Origin'] == row['Origin']) & (all_results_df['Epochs'] == row['Epochs']) & (all_results_df['Learning Rate'] == row['Learning Rate']) & (all_results_df['Model'] == row['Model']) & (all_results_df['Batch'] == row['Batch'])]['FileName']
        #remove rows in files it the value is None
        files = files.dropna()

        #Create a new df that merges all the files
        df_all_species = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in files])
        analyze_and_plot_validation_results(logger, df_all_species, folder_path, files.iloc[0])

    #Count how many different combinations of Origin, Epochs, Learning Rate, Model, Batch are there where Species = all
    combinations = all_results_df[all_results_df['Specie'] == 'all'][['Origin', 'Epochs', 'Learning Rate', 'Model', 'Batch']].drop_duplicates().shape[0]

    #Create a color arra of combinations colors
    color = cm.viridis([i/combinations for i in range(combinations)])
    color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.8)' for c in color]
    #create a new column Color
    all_results_df['Color'] = ''
    # For each row with Species = all
    for index, row in all_results_df[all_results_df['Specie'] == 'all'].iterrows():
        # Get the color for this combination
        rowColor = color[all_results_df[all_results_df['Specie'] == 'all'][['Origin', 'Epochs', 'Learning Rate', 'Model', 'Batch']].drop_duplicates().index.get_loc(index)]        #Assign the same color to all rows with the same combination of Origin, Epochs, Learning Rate, Model, Batch
        all_results_df.loc[all_results_df[
            (all_results_df['Origin'] == row['Origin']) & (all_results_df['Epochs'] == row['Epochs']) & (
                        all_results_df['Learning Rate'] == row['Learning Rate']) & (
                        all_results_df['Model'] == row['Model']) & (
                        all_results_df['Batch'] == row['Batch'])].index, 'Color'] = rowColor
    # Create a size array where the size is larger for rows where Species = ALL
    size_array = [20 if species == 'all' else 10 for species in all_results_df['Specie']]

    # Create a 3D scatter plot of accuracy, precision, and recall
    fig = go.Figure(data=[go.Scatter3d(
        x=all_results_df['Accuracy'],
        y=all_results_df['Precision'],
        z=all_results_df['Recall'],
        mode='markers',
        text=all_results_df[['Origin', 'Model', 'Learning Rate', 'Epochs', 'Batch', 'Specie']].apply(
            lambda row: f'Origin: {row["Origin"]}, Model: {row["Model"]}, Learning Rate: {row["Learning Rate"]}, Epochs: {row["Epochs"]}, Batch: {row["Batch"]}, Species: {row["Specie"]}',
            axis=1),
        hoverinfo='text',
        marker=dict(
            size=size_array,
            color=all_results_df['Color'],
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])

    # Set labels
    fig.update_layout(scene=dict(
        xaxis_title='Cumulative Accuracy',
        yaxis_title='Cumulative Precision',
        zaxis_title='Cumulative Recall'))
    fig.show()


if __name__ == "__main__":
    root = tk.Tk()  # Using the Tkinter instance from PySimpleGUI
    root.withdraw()
    # Ask the user to select a folder
    folder_path = filedialog.askdirectory(initialdir=".", title='Choose a folder with validation log files')

    if not folder_path:
        print("No folder selected")
        exit()

    analyseValidationFolder(folder_path)
    exit(0)
