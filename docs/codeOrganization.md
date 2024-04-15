### Folders organization:
```
PRONTO_TK
│   README.md
│   <project files>.py    
│
└───docs
│   <documentation files>
│   
└───0.base_T5
│   <base T5 model files>
│
└───assets
│   <PRONTO_TK icons and image files>
│
└───Original Input
│   └───<dataset name>
│         └───downloaded
│         │   <donwnloaded Uniprot files>
│         └───embeddings
│         │   <generated embedding files>
└───experiments
    └───_configurations
    │   <experiment configuration files>
    └───<experiment name>
        └───0.DataSet
        │   <TT, FT, FV data files>
        └───1.DL_Training/Model
        │   <trained models>
        └───logs
        │   <log files>
        └───results
        │   <results files>
        └───inferences
            <inference results>
```

### Code organization
The code organization follows a modular approach, where each file or module is responsible for a specific functionality. This makes the code easier to understand, maintain, and extend. The use of a separate thread for the machine learning tasks ensures that the GUI remains responsive even when a task is running.
Here's a brief overview:

### `pronto-tk.py`
- This is the main entry point of the application. It imports the necessary classes and functions from the other files and runs the application.

### `main_window.py`
- This file contains the `MainWindow` class. This class is responsible for handling the main window of the application. It reads a configuration file, sets up the GUI, and starts the processes when the user clicks the buttons.

### `config_window.py`
- This file likely contains the ConfigWindow class. This class is probably responsible for handling the configuration window of the application. It may read a configuration file, set up the GUI, and start the processes when the user clicks the buttons. 

### `jj0_goDownloads.py`
- This file is used to download the necessary files from the GO repository.

### `ab1_analyzeUniprotFiles.py`
- This file analyzes embedding files and produces some simple statistics.

### `ab2_mergeEmbeddingFiles.py`
- This file merges all the embedding files into a single file.

### `jj0_goDownloads.py`
- This file is used to download the necessary files from the GO repository. It creates the required queries to UniProt and monitors their execution.
 
### `jj1_preprocessUniprotFiles.py`
- This file preprocesses the UniProt files and creates the embedding files.

### `jj2_prepareInputFiles.py`
- This file prepares the three main input files for the machine learning model. datasetTT, datasetFT, and datasetFV.

### `jj3_DLtraining.py`
- This file trains the deep learning model using the input files prepared in the previous step.

### `jj4_DLinference.py`
- This file runs inference on the test data using the trained model.
### `logger.py`
- This file contains the `LoggerHandler` class. This class is used to handle logging. It sets up a logger that writes to a file and optionally to stdout. It also has a queue for messages that need to be displayed in the GUI.

### `process.py`
- This file contains the `myProcess` class. This class is used to manage a process that runs in a separate thread. It starts the thread and periodically checks a queue for updates from the process.

### `threads.py`
- This file contains the `thread_jj2`, `thread_jj3`, `thread_jj3ft`, and `thread_jj4` functions. These functions are the tasks that run in separate threads. They perform various machine learning tasks, such as preparing the input files, training and testing the model, fine-tuning the model, and running inference.

### `config.py`
- This file contains the `read_config` and `read_configuration` methods from the `MainWindow` class. These methods are related to reading and handling the configuration file.

### `model_util.py`
- This file contains the neural network model classes. 

### `valStats.py`
- This file is used to calculate and display the validation statistics of the model starting from the log files.
### `logStat.py`
- This file contains the help instructions for the con`logStat` class. This class is used to parse the log files generated during the training and testing of the model. It extracts the relevant information from the log files and calculates the validation statistics.
### `tooltips.py`
- This class is used to create the instructions for the configuration items in the configuration file.
### `valStatsFolder.py`
### `valStats.py`
- This file is used to calculate and display the validation statistics of the model starting from the log files.
