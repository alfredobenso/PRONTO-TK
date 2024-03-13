### Code organization

The code of the project is organized across several Python files, each serving a specific purpose. Here's a brief overview:

### `main.py`
- This is the entry point of the application. It imports the necessary classes and functions from the other files and runs the application.

### `main_window.py`
- This file contains the `MainWindow` class. This class is responsible for handling the main window of the application. It reads a configuration file, sets up the GUI, and starts the processes when the user clicks the buttons.

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

The code organization follows a modular approach, where each file or module is responsible for a specific functionality. This makes the code easier to understand, maintain, and extend. The use of a separate thread for the machine learning tasks ensures that the GUI remains responsive even when a task is running.
