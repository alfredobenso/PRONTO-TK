import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import re
import os

from tkinter import filedialog
import tkinter as tk

def analyzeLog(logfile):
    # Open the file and read its content
    if not os.path.exists(logfile):
        print(f"File {logfile} does not exists")
        return

    with open(logfile, 'r') as file:
        content = file.readlines()

    # Initialize dictionaries to store the loss and accuracy values for each experiment
    loss_values = {}
    accuracy_values = {}
    current_experiment = None
    model = None #used model
    batch = None #used batch size
    phase = "Training"

    # Initialize a dictionary to store the number of epochs for each experiment
    epochs_count = {}

    # Initialize a dictionary to store the four numbers for each experiment
    experiment_numbers = {}

    # Initialize a variable to store the initial learning rate
    initial_learning_rate = None

    # Iterate over each line in the file content
    for index, line in enumerate(content):
        # Use a regular expression to find the experiment name in each line
        experiment_match = re.search(r'Experiment name: (\w+)', line)
        if experiment_match:
            # If an experiment name is found, use it as the current experiment name
            current_experiment = experiment_match.group(1)
            loss_values[current_experiment] = []
            accuracy_values[current_experiment] = []
            epochs_count[current_experiment] = 0

        # Use a regular expression to find the loss value in each line
        loss_match = re.search(r'loss: (\d+\.?\d*)', line)
        if loss_match and current_experiment:
            # Append the loss value to the list associated with the current experiment name
            loss_values[current_experiment].append(float(loss_match.group(1)))

        # Use a regular expression to find the accuracy value in each line
        accuracy_match = re.search(r'train_acc: (\d+\.?\d*)', line)
        if accuracy_match and current_experiment:
            # Append the accuracy value to the list associated with the current experiment name
            accuracy_values[current_experiment].append(float(accuracy_match.group(1)))

        # If an epoch line is found, increment the count of epochs for the current experiment
        epoch_match = re.search(r'epoch: (\d+)', line)
        if epoch_match and current_experiment:
            epochs_count[current_experiment] += 1

        # If an epoch line is found, increment the count of epochs for the current experiment
        model_match = re.search(r'Model (\d+)', line)
        if model_match:
            model = int(model_match.group(1))

        # If an epoch line is found, increment the count of epochs for the current experiment
        batch_match = re.search(r'Batch Size: (\d+)', line)
        if batch_match:
            batch = int(batch_match.group(1))

        phase_match = re.search(r'Phase: (.*)', line)
        if phase_match:
            phase = phase_match.group(1)

        # Use a regular expression to find the initial learning rate
        #lr_match = re.search(r'Initial Learning Rate:  (\d+\.?\d*$)', line)
        if line.find("Initial Learning Rate:") != -1:
            tokens = line.strip().split(":")
            initial_learning_rate = float(tokens[1])

        # Use a regular expression to find the four numbers at the end of each experiment
        numbers_match = re.search(r'bacc,MCC,final_test_aupr,final_auc_roc\n', line)
        if numbers_match and current_experiment:
            numbers_match = re.search(r'([\d\.,\s]+)', content[index + 1])
            # Store the four numbers in the dictionary
            experiment_numbers[current_experiment] = [format(float(num.replace(",", ".")), '.4f') for num in numbers_match.group(1).split()]

    # Create a figure with a 2x1 grid layout
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the loss values for each experiment in the first row
    axs[0].set_title(f'Loss values (epochs: {epochs_count[current_experiment]}, lr: {initial_learning_rate}, batch: {batch}, model: {model})')
    for experiment, values in loss_values.items():
        axs[0].plot(values, label=experiment)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot the accuracy values for each experiment in the second row
    axs[1].set_title(f'Accuracies (epochs: {epochs_count[current_experiment]}, lr: {initial_learning_rate}, model: {model})')
    for experiment, values in accuracy_values.items():
        axs[1].plot(values, label=experiment)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Create a table with the four numbers for each experiment in a new row, spanning both columns
    table_data = list(experiment_numbers.values())
    table_rows = list(experiment_numbers.keys())
    table_columns = ['bacc', 'MCC', 'final_test_aupr', 'final_auc_roc']

    if len(table_data) > 0:
        # Use plt.table directly with colWidths=[0.2]*len(table_columns) to make it span both columns
        table = axs[1].table(cellText=table_data, rowLabels=table_rows, colLabels=table_columns, loc='center',
                             cellLoc='center', cellColours=None, bbox=[0, -0.7, 1, 0.4], colWidths=[0.2]*len(table_columns))
        table.auto_set_font_size(False)
        table.set_fontsize(12)
    fig.suptitle(f'{phase} Results - Epochs: {epochs_count[current_experiment]}, Learning Rate: {initial_learning_rate}, Model: {model}', fontsize=16)

    # Save the figure in the results folder
    # Extrace the path from the file_path
    folder_path = os.path.dirname(logfile)
    # Extrace the filename from the file_path
    file_name = os.path.basename(logfile)
    file_name = file_name.replace(".log", ".jpg")

    #replace the subfolder validations with the folder "results" and create it if it does not exists
    folder_path = folder_path.replace("logs", "results")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def main():
    # Analyze the log file for the experiment with the highest number of epochs
    root = tk.Tk()  # Using the Tkinter instance from PySimpleGUI
    root.withdraw()

    log_file = filedialog.askopenfilename(title='Choose a Log file', filetypes=[('Log files', '*.log')])

    if not log_file:
        print("No file selected")
        exit()

    analyzeLog(log_file)

if __name__ == "__main__":
    main()
