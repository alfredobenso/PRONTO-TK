# User Manual for PRONTO

## Introduction
PRONTO is a Python-based tool designed to handle various operations related to bioinformatics. It provides a GUI interface for managing and executing different stages of a data pipeline.

## Requirements
- pip install requirements.txt

## Usage
Run the executable pronto-tk.py

## The main window
The tool can be used in two ways: **manual**, by clicking the button of each stage, or **automatically**, by using the two separate "Run" buttons.
The MOST IMPORTANT part of the tool is the CONFIGURATION FILE. The syntax HAS to be respected, otherwise the tool will not work.
**Sometimes the main window does not respond if it is not resized or moved first**. This is a known issue and will be fixed in the future.

### Configuration
The configuration for the application is done through a configuration file. This file is selected when the MainWindow is initialized. The configuration file is in INI format and contains various sections for different parts of the application.

## Configuration parameters
- Configuration files are stored in the **experiments/_configurations** folder. When launching the PRONTO app, you will be asked to open a configuration file. If you don't have one already, open default.ini and then save it with a different name.
- To configure the file you can use a text editor OR click on the **Configure \<exp name>** button.
- **Notice: ALWAYS USE double quotes (") for strings and lists**
- In general: **""** means ALL and **[]** means NONE

### GENERAL 
- **name**: The name of the training process
- **acronym**: A short form or abbreviation of the name
- **folder**: The directory where the training data or results are stored
- **type**: "single"/"leaveoneout": Specifies the type of experiment.
- **originaldataset**: The path to the original dataset used for training
- **silentmode**: integer 0/1: a flag to control the verbosity of the training process. If "1", log is not reproduced on STDIN

### UNIPROT 
- **go_ids**: LIST: GO terms to be searched. You will get results of proteins matchin ANY of these GO Terms. Example: ["GO:0003723", "GO:0003724"]
- **go_includedescendants**: string "True"/"False": a flag to control the GO terms descendants. If "True", all the descendants of the GO terms will be included in the search
- **go_batchsize**: -1 or integer. If -1, the 'stream' REST service is used, otherwise the 'search' REST call is used. 'search' can be used to download in batches when the connection with UniProt is problematic or slow. In that case, 500 is a good value
- **go_maxproteinsdownload**: -1 means NO LIMIT, otherwise it is the MAX number of proteins to be downloaded for each combination of parameters (reviewed/annotation). The minimum number of downloaded proteins is, in any case, equal to batchsize.
- **go_taxonomies**: LIST: Taxonomy IDs to be used for the search. Example: ["1783272"]
- **go_folder**: The folder where the files downloaded from Uniprot are stored and the embeddings will be created
- **datasetname**: The name of the dataset to be created
 
### UNIPROT-Label_1 and UNIPROT-Label_0 
- **reviewed**: it can be: ["true"], ["false"], or ["true","false"]. For each label 0/1, it configures the "reviewed" parameter when downloading proteins from UniProt.
- **annotation**: LIST: e.g.: ["manual", "automatic"]. For each label 0/1, it configures the annotation type (manual/automatic) to be used to select data for download

### EMBEDDINGS
- **per_residue**: "True" or "False". It specifies if per residue embeddings have to be computed (not tested)
- **per_protein**: "True" (or "False"). It specifies if per protein embeddings have to be computed (default)
- **sec_struct**: "False" (or "True"). Used to predict secondary structure and compute embeddings from it (not tested)
- **sequencebatchsize**: Integer. It defines the size of the batch of proteins to be sent to the embeddings network. Sometimes the embeddings procedure crashes because of memory requirements. Lowering this value might help. It depends on the system overall computational power and RAM memory. 25 works on MAC M2 with 64GB RAM
- **createflag**: "ask", "yes", "no". It specifies if the embeddings have to be computed if already present
- **cachedataset**: Files that can be used to retrieve embeddings from a previous run

### ENVIRONMENT
- **torchdevice**: MAC: "mps", WIN: "cpu". Specifies the hardware to be used for training/fine-tuning/inference. If CUDA is available it is automatically selected
- **t5secstructfolder**: The folder where the T5 model for secondary structure prediction is stored - This model is used to compute embeddings for the sequences in the dataset

### TRAINTEST
- **createflag**: yes/no/ask: a flag to control whether a new dataset (or new embeddings) have to be generated if already present
- **percent**: LIST of two floats: it specifies the percentage of data to be used for training and testing. The remaining will be used for validation
- **perc_if_ft_overlap**: IF TT and FT datasets overlap, the percentage of the overlapping data to be used for training and testing
- **_01ratio**: Specifies the ratio of class 0 to class 1 in the data
- **includedspecies**: LIST of strings: Specifies the species to be included in the training process - "" means all species are included
- **excludedspecies**: LIST of strings: Specifies the species to be excluded from the training process - [] means no species are excluded
- **leaveoneoutspecies**: LIST of LIST of strings: each list specifies a species or group of species, to be used for testing in the leave-one-out training process
- **trainflag**: yes/no/ask: a flag to control whether model already existing have to be retrained
- **batch_size**: LIST: specifies the batch size for the training process
- **epoch**: LIST: specifies the number of epochs for the training process
- **learning_rate**: LIST: specifies the learning rate for the training process
- **model_name**: name of the class (in model_util.py) corresponding to the model to be used for training
- **model_dir**: The directory where the model is stored

### TRAINTEST-Label_0 and TRAINTEST-Label_1
- **reviewed**: it can be: ["true"], ["false"], or ["true","false"]. For each label 0/1, it configures the "reviewed" parameter when filtering proteins for the Training operations.
- **annotation**: LIST: e.g.: ["manual", "automatic"]. For each label 0/1, it configures the annotation type (manual/automatic) to be used when filtering proteins for the Training operations

### FINETUNING
- **createflag**: yes/no/ask: a flag to control whether a new dataset (or new embeddings) have to be generated if already present
- **percent**: LIST of two floats: it specifies the percentage of data to be used for fine-tuning the model. The remaining will be used for validation
- **_01ratio**: Specifies the ratio of class 0 to class 1 in the data
- **includedspecies**: LIST of strings: Specifies the species to be included in the fine-tuning process - "" means all species are included
- **excludedspecies**: LIST of strings: Specifies the species to be excluded from the fine-tuning process - [] means no species are excluded
- **trainflag**: yes/no/ask: a flag to control whether model already existing have to be retrained
- **batch_size**: LIST: specifies the batch size for the fine-tuning process
- **epoch**: LIST: specifies the number of epochs for the fine-tuning process
- **learning_rate**: LIST: specifies the learning rate for the fine-tuning process

### FINETUNING-Label_0 and FINETUNING-Label_1
- **reviewed**: it can be: ["true"], ["false"], or ["true","false"]. For each label 0/1, it configures the "reviewed" parameter when filtering proteins for the Fine-Tuning operations.
- **annotation**: LIST: e.g.: ["manual", "automatic"]. For each label 0/1, it configures the annotation type (manual/automatic) to be used when filtering proteins for the Fine-Tuning operations

### VALIDATION
- **createflag**: yes/no/ask: a flag to control whether a new validation dataset have to be generated, if already present
- **percent**: LIST of two floats: it specifies the percentage of data to be used for fine-tuning the model. The remaining will be used for validation
- **_01ratio**: Specifies the ratio of class 0 to class 1 in the data
- **includedspecies**: LIST of strings: Specifies the species to be included in the inference/validation process - "" means all species are included
- **excludedspecies**: LIST of strings: Specifies the species to be excluded from the inference/validation process - [] means no species are excluded
- **inferflag**: yes/no/ask: a flag to control whether an inference has to be repeated in case a log file with the results is already present
- **input_model**: "both"/"tt"/"ft": Specifies the model to be used for inference. "both" means both the TT and FT models are used. "tt" means only the TT model is used. "ft" means only the FT model is used

### VALIDATION-Label
- **reviewed**: it can be: ["true"], ["false"], or ["true","false"]. For each label 0/1, it configures the "reviewed" parameter when filtering proteins for the Fine-Tuning operations.
- **annotation**: LIST: e.g.: ["manual", "automatic"]. For each label 0/1, it configures the annotation type (manual/automatic) to be used when filtering proteins for the Fine-Tuning operations

**Notice**: it is responsability of the user to create a Validation dataset that does not overlap with the Training and Fine-Tuning datasets. The tool does not check for this.
## Running the Data Pipeline
To run the full data pipeline, click on the "Run Data Pipeline" button. This will execute the stages of downloading data from UniProt and computing embeddings.

## Running the Training/Inference Pipeline
To run the full training/inference pipeline, click on the "Run Training/Inference Pipeline" button. This will execute the stages of data filtering, training/testing, fine-tuning, and inference.

## Checking Input/Output
The application provides a way to check the existence of certain files and directories and updates the UI components accordingly. This is done by clicking the "Refresh" button.

## Viewing Results
The results of the operations can be viewed by clicking on the "Results" button. This will open the results directory in the file explorer.

## Viewing Logs
The logs of the operations can be viewed by clicking on the "Logs" button. This will open the logs directory in the file explorer.

## Viewing Inferences
The inferences of the operations can be viewed by clicking on the "Inferences" button. This will open the inferences directory in the file explorer.

## Detailed Data Download and Embeddings Pipeline steps
### Download from UniProt
The first stage of the pipeline is to download data from UniProt. This is done by clicking on the "Download from UniProt" button. This will start the download process and display the progress in the UI.
Files are downloaded into the 'download' folder of the "go_folder" folder in the UNIPROT section of the configuration file.
The downloaded file will include the following columns: 
- "Label"
- "Annotation"
- "Entry"
- "Reviewed"
- "Entry Name"
- "Protein names"
- "Gene Names"
- "Organism"
- "Length"
- "Caution"
- "Gene Ontology (molecular function)"
- "Sequence"

### Compute Embeddings
The second stage of the pipeline is to compute embeddings for the downloaded data. This is done by clicking on the "Compute Embeddings" button. This will start the computation process and display the progress in the UI.

***At the moment the tool works ONLY with 1024 embeddings.***

The model used for embeddings is: "Rostlab/prot_t5_xl_half_uniref50-enc" [https://github.com/agemagician/ProtTrans], and it COULD be changed in jj1_prepocessUniprotFiles.py

This operation is performed on ALL files with extension .dataset.csv that are in the 'download' folder of the "go_folder" folder in the UNIPROT section of the configuration file. In this way it is possible to include files from different sources, as long as they are in the correct format.

## Detailed Training-Test/Fine Tuning/Inference Pipeline steps
### Data Filters
The first stage of the pipeline is to filter the data. This is done by clicking on the "Data Filters" button. This will start the filtering process and display the progress in the UI. 

It will create three minimal datasets: dataset_TT (Test Train), dataset_FT (Fine Tune), dataset_FV (Final Validation or Inference). They will be used and (if necessary) further filtered in each experiment. 

For "leaveOneOut" experiments it will not create the FV_dataset file because it will be created on-the-fly at each experiment iteration.

### Training/Test
The second stage of the pipeline is to train and test the model. This is done by clicking on the "Training/Test" button. This will start the training and testing process and display the progress in the UI.

If the type of experiment is set to "leaveOneOut" in the configuration file, the tool will perform a leave-one-out experiment. In this case, the FV_dataset file will be created on-the-fly at each experiment iteration.
The group of species to be "left out" in each experiment is defined by a LIST of LISTS in the configuration file. Each list will be used as a group of species to be "left out" in each experiment iteration and then used as validation in the final step of the experiment.

#### Model selection
At the moment is only possible to choose from two models. The description of the models can be found in [https://doi.org/10.1093/bib/bbac215]. New models can be added and used by adding the class to the model_util.py file and adding the class name to the
"model" parameter in the configuration file.

#### Parameters exploration
In both "single" and "leaveOneOut" experiments, if the configuration file has more than one value in the "batch_size", "epoch", "learning_rate" (in Training/Test and Fine Tuning), then a different model will be trained for each combination of parameters.

## Conclusion
PRONTO is a comprehensive tool for managing and executing bioinformatics data pipelines. It provides a user-friendly GUI for easy operation and monitoring of the different stages of the pipeline.
