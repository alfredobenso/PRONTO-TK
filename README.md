## Installation Guide

1. **Clone the Repository**

   First, you need to clone the repository from GitHub. Open your terminal and navigate to the directory where you want to clone the repository. Then, run the following command:

   ```bash
   git lfs clone -b naturecomm https://github.com/alfredobenso/PRONTO-TK
   
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
	Select: "CFG_Terrabacteria RBP AutoEthanogenum2.ini"

## Configuration Manual
see documentation (docs/manual.md)

## Run the experiment
> **WARNING**: sometimes you need to RESIZE or MOVE the app windows to being able to interact with them (like clicking on a Phase to execute it or clicking a button). This is a known issue, and we are working on it.

The Uniprot Datasets are already downloaded but the embeddings computation might require a LONG time since it has to regenerate an embeddings file of over 2G.
It is not necessary to run the embeddings computation since we already included the final datasets.

To execute the whole experiment as reported in the paper, run sequentially all the steps in the pipeline.

## Other content
**'scripts'** **folder**:

* tr.tl.rank.bin,r studies the agreement in gene rankings by differential transcription and differential translation analyses.


* consistency.dtl.dtr.r carries out cross-comparison of top 5% gene sets across the three metrics used to evaluate translational regulation (TL-STATUS, TL-PROFILE, and POLY-FRACTIONS)

**'results/DTL'** **folder**:

* results images referenced in the paper

