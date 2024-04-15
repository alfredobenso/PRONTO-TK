#You can add tooltips for the parameters of the training process here
tooltips = {
    'name': 'FORMAT: "<string>" - The name of the training process',
    'acronym': 'FORMAT: "<string>" - A short form or abbreviation of the name',
    'folder': 'FORMAT: "<string>" - The directory where the training data or results are stored',
    'type': 'FORMAT: "single" or "leaveoneout" - Specifies the type of training process.',
    'originaldataset': 'FORMAT: "<string>" - The path to the original dataset used for training',
    'silentmode': 'FORMAT: integer 1/0: a flag to control the verbosity of the training process. If "1", log is not reproduced on STDIN',
    'percent': 'FORMAT: python LIST of two floats [<%train>,<%test>] - it specifies the percentage of data to be used for training and testing. The remaining will be used for validation',
    '_01ratio': '"FORMAT: balanced" or "" - Specifies the ratio of class 0 to class 1 in the data',
    'includedspecies': 'FORMAT: python LIST of strings ["<species 1>","<species 2>",...] - Specifies the species to be included in the training process - "" means all species are included',
    'excludedspecies': 'FORMAT: python LIST of strings ["<species 1>","<species 2>",...] - Specifies the species to be excluded from the training process - [] means no species are excluded',
    'leaveoneoutspecies': 'FORMAT: python LIST of LISTS of strings [["<species 1.1>, <species 1.2>"],["<species 2>"],...] - each list specifies a species or group of species, to be used for testing in the leave-one-out training process',
    'createflag': 'FORMAT: "yes"/"no"/"ask" - a flag to control whether a new dataset (or new embeddings) have to be generated if already present',
    'trainflag': 'FORMAT: "yes"/"no"/"ask" - a flag to control whether model already existing have to be retrained',
    'inferflag': 'FORMAT: "yes"/"no"/"ask" - a flag to control whether an inference has to be repeated in case a file with the results is already present',
    'batch_size': 'FORMAT: python LIST of integers [<batch_size_1>,<batch_size_2>,...] - specifies the batch sizes for the training process',
    'epoch': 'FORMAT: python LIST of integers [<epoch_size_1>,<epoch_size_2>,...] - specifies the number of epochs for the training/fine-tuning process',
    'learning_rate': 'FORMAT: python LIST of floats [<lr_1>,<lr_2>,...] - specifies the learning rate for the training/fine-tuning process',
    'model_name': 'FORMAT: "<string>" - name of the class (in model_util.py) corresponding to the model to be used for training/fine-tuning',
    'annotation': 'FORMAT: python LIST ["manual","automatic"] - for each label 0/1 in training/fine-tuning, it specifies the type of annotation (manual/automatic or both) to be used to select data for training/fine-tuning/inference',
    'reviewed': 'FORMAT: python LIST ["true","false"] - for each label 0/1 in training/fine-tuning, it specifies if to use reviewed proteins or not download or to select data for training/fine-tuning/inference',
    'torchdevice': 'Specifies the hardware to be used for training/fine-tuning/inference MAC: "mps", WIN: "cpu", if CUDA is available it is automatically selected',
    't5secstructfolder': 'FORMAT: <string> - The folder where the T5 model for secondary structure prediction is stored - This model is used to compute embeddings for the sequences in the dataset',
    'folder': 'FORMAT: <string> - The folder where the files downloaded from Uniprot are stored and the embeddings will be created',
    'sequencebatchsize': 'FORMAT: <integer> - The batch size to be used for the sequence embeddings computation - It depends on the system overall computational power and RAM memory. 25 works on MAC M2 with 64GB RAM',
    'go_ids': 'FORMAT: python LIST of strings ["GO:<goid>"] - GO terms to be searched. You will get results of proteins matchin ANY of these GO Terms',
    'go_includedescendants' : 'FORMAT: string "True"/"False" - a flag to control whether the descendants of the GO terms have to be included in the search',
    'go_batchsize' : 'FORMAT: -1 or <integer> - If -1, the "stream" REST service is used, otherwise the "search" REST call is used. "search" can be used to download in batches when the connection with UniProt is problematic or slow. In that case, 500 is a good value',
    'go_maxproteinsdownload' : 'FORMAT: -1 means NO LIMIT, otherwise the MAX number of TOTAL proteins to be downloaded - The minimum is in any case batchsize.',
    'go_taxonomies' : 'FORMAT: python LIST of strings ["tax_1","tax_2",...]: Taxonomy IDs to be used for the search',
    'go_folder' : 'FORMAT: <string> - The folder where the files downloaded from Uniprot are stored and the embeddings will be created',
    'datasetname' : 'FORMAT: <string> - The name of the dataset to be created',
    'cachedataset' : 'FORMAT: LIST: a list of dataset.embeddings.csv files that can be used as cache during embeddings computation',
    'input_model': 'FORMAT: "both"/"tt"/"ft" - Specifies the model to be used for inference. "both" means both the TT and FT models are used. "tt" means only the TT model is used. "ft" means only the FT model is used',
}
