## Installation Guide

1. **Clone the Repository**

   First, you need to clone the repository from GitHub. Open your terminal and navigate to the directory where you want to clone the repository. Then, run the following command:

   ```bash
   git clone https://github.com/alfredobenso/PRONTO-TK
2. **Navigate to the Project Directory**

   After cloning the repository, navigate to the project directory by running:

   ```bash
   cd your-repository-name
   
3. **Install the Dependencies**

   This project uses Python and pip for managing dependencies. To install the dependencies, you need to run:

   ```bash
   pip install -r requirements.txt

4. **Run the Application**

   Now that all the dependencies are installed, you can run the application. The entry point of the application is the `main.py` file. Run the application with the following command:

   ```bash
   python pronto-tk.py

5. **Using the Application**
    The first operation is to select a configuration file. Configuration files are in the experiments/_configurations folder.

## Configuration Manual
see documentation [here](docs/manual.md)

## Run a test experiment
> **WARNING**: sometimes you need to RESIZE or MOVE the app windows to being able to interact with them (like clicking on a Phase to execute it or clicking a button). This is a known issue, and we are working on it.

We included a small dataset of embeddings to test the code. 
You can run the toy example by choosing the configuration file: CFG_Terrabacteria TEST.ini

To execute the whole experiment as reported in [], run sequentially the following two pipelines:
- CFG_Terrabacteria TFBS DATA ONLY.ini
- CFG_Terrabacteria TFBS LeaveOneOut.ini
The first pipeline will generate the data and the second will run the LeaveOneOut experiment. The embedding phase might be the most time-consuming part of the experiment.
