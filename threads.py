# This file contains the functions that are called by the threads in the main window
import os.path
import os
import time
import pandas as pd
from itertools import product
import ab1_analyzeUniprotFiles
import ab2_mergeEmbeddingFiles
import jj0_goDownloads
import jj1_preprocessUniprotFiles
import jj2_prepareInputFiles
import jj3_DLtraining
import jj4_DLinference
import logStat
import valStatsFolder

#this is an example function
def thread_function(cfg, logger):
    logger.log_message(cfg["GENERAL"]["acronym"])
    for i in range(100):
        time.sleep(0.2)
        #logger.log_message((parentWindow.progress_bar.set, ((i + 1)/100,)))
        logger.log_message(f'Progress: {(i + 1)/100}')
    logger.log_message('Simulation finished \n')

def thread_jj0(cfg, window, callback=None, semaphore=None):
    if semaphore is not None:
        semaphore.acquire()

    logFolder = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
    if not os.path.exists(logFolder):
        os.makedirs(logFolder)

    SILENT_MODE = cfg["GENERAL"]["silentmode"]

    window.logger_handler.closeHandlers()
    window.logger_handler.filename = os.path.join(logFolder, "datasetDownload.log")
    window.logger_handler.setup_logger(SILENT_MODE)
    window.logger_handler.log_message("Starting downloads ...")
    jj0_goDownloads.downloadUPProteins(cfg, window.logger_handler)
    #Add here other embeddings computation functions
    window.logger_handler.log_message("\nDownloads done ...\n", 1)
    window.logger_handler.log_message(f"NOTICE: if you need to save space, you can now delete the files starting with 'Label_' from the folder 'downloads' in '{cfg['UNIPROT']['go_folder']}'",1)

    # Release the semaphore when done
    if semaphore is not None:
        semaphore.release()

    # Call the callback function if it's not None
    if callback is not None:
        callback()
    return


def thread_jj1(cfg, window, callback=None, semaphore=None):
    if semaphore is not None:
        semaphore.acquire()

    SILENT_MODE = cfg["GENERAL"]["silentmode"]

    logFolder = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
    if not os.path.exists(logFolder):
        os.makedirs(logFolder)

    window.logger_handler.closeHandlers()
    window.logger_handler.filename = os.path.join(logFolder, "datasetEmbeddings.log")
    window.logger_handler.setup_logger(SILENT_MODE)
    window.logger_handler.log_message("Starting Embeddings computation...")
    jj1_preprocessUniprotFiles.computeEmbeddings(cfg, window.logger_handler)
    #Add here other embeddings computation functions
    window.logger_handler.log_message("Analyzing Embedding files...")
    ab1_analyzeUniprotFiles.analyzeEmbeddingFiles(cfg, window.logger_handler)
    window.logger_handler.log_message("Mergin Embedding files...")
    ab2_mergeEmbeddingFiles.mergeEmbeddings(cfg, window.logger_handler)
    window.logger_handler.log_message("\nEmbeddings computation done\n",1)
    window.logger_handler.log_message(f"NOTICE: if you need to save space, you can now delete the 'downloads' and 'embeddings' folders in '{cfg['UNIPROT']['go_folder']}'",1)

    # Release the semaphore when done
    if semaphore is not None:
        semaphore.release()

    # Call the callback function if it's not None
    if callback is not None:
        callback()
    return

def thread_jj2(cfg, window, callback=None, semaphore=None):
    if semaphore is not None:
        semaphore.acquire()

    SILENT_MODE = cfg["GENERAL"]["silentmode"]

    logFolder = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
    if not os.path.exists(logFolder):
        os.makedirs(logFolder)

    window.logger_handler.closeHandlers()
    window.logger_handler.filename = os.path.join(logFolder, "datasetFiltering.log")
    window.logger_handler.setup_logger(SILENT_MODE)
    window.logger_handler.log_message("Starting Dataset filtering...")
    jj2_prepareInputFiles.filterDataSet(cfg, window.logger_handler)
    window.logger_handler.log_message("Dataset done",1)

    # Release the semaphore when done
    if semaphore is not None:
        semaphore.release()

    # Call the callback function if it's not None
    if callback is not None:
        callback()
    return

def thread_jj3(cfg, window, callback=None, semaphore=None):
    if semaphore is not None:
        semaphore.acquire()

    SILENT_MODE = cfg["GENERAL"]["silentmode"]

    if cfg["GENERAL"]["type"] == "single":
        maxIter = len(cfg["TRAINTEST"]["epoch"]) * len(cfg["TRAINTEST"]["learning_rate"]) * len(cfg["TRAINTEST"]["batch_size"])
        curIter = 1
        for EPOCHS, LR, BATCH in product(cfg["TRAINTEST"]["epoch"], cfg["TRAINTEST"]["learning_rate"], cfg["TRAINTEST"]["batch_size"]):
            log_file_name = f'log_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}.log'
            log_file_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
            if not os.path.exists(log_file_path):
                os.makedirs(log_file_path)

            if not os.path.exists(os.path.join(log_file_path, log_file_name)) or cfg["TRAINTEST"]["trainflag"] == "yes":
                window.logger_handler.closeHandlers()
                # TBD: change the log file name
                window.logger_handler.filename = os.path.join(log_file_path, log_file_name)
                window.logger_handler.setup_logger(SILENT_MODE)
                window.logger_handler.log_message("*" * 80)
                window.logger_handler.log_message(f"Experiment name: {cfg['GENERAL']['name']}")
                window.logger_handler.log_message("*" * 80)

                window.logger_handler.log_message(f"\nRunning training: {EPOCHS} epochs, {LR} learning rate, {BATCH} batch size, model {cfg['TRAINTEST']['model_name']} - Iteration {curIter} of {maxIter}", (curIter / maxIter))
                window.logger_handler.log_message(f"Silent Mode is {'ON' if SILENT_MODE else 'OFF'}...")
                window.logger_handler.log_message(f"Using Model {cfg['TRAINTEST']['model_name']}...")

            model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler, _SILENT_RUN=SILENT_MODE, trainFlag=cfg["TRAINTEST"]["trainflag"])

            logStat.analyzeLog(os.path.join(log_file_path, log_file_name))

            if not os.path.exists(model_location):
                window.logger_handler.log_message("Model not found")
                return()

            curIter += 1
    else:
        maxIter = len(cfg["TRAINTEST"]["epoch"]) * len(cfg["TRAINTEST"]["learning_rate"] * len(cfg["TRAINTEST"]["batch_size"]))
        curIter = 1

        MODEL = cfg['TRAINTEST']['model_name']

        for EPOCHS, LR, BATCH in product(cfg["TRAINTEST"]["epoch"], cfg["TRAINTEST"]["learning_rate"], cfg["TRAINTEST"]["batch_size"]):

            df = pd.DataFrame(columns=['Specie', 'bacc', 'MCC', 'Final_Test_Aupr', 'Final_Auc_Roc'])

            # for each specie in _EXPERIMENT_SPECIES, use that species as Validation and all the others as Training/Test
            results = []
            for idx, speciesGroup in enumerate(cfg["TRAINTEST"]["leaveoneoutspecies"]):
                print(f"Processing species group {idx}")
                print(f"Validation Group: {idx}")
                print(f"Validation Species: {speciesGroup}")
                print(f"Training parms: EPOCHS={EPOCHS}, LR={LR}, MODEL={MODEL}")

                # *****
                log_file_name = f'log_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}_exclude_{idx}.log'
                log_file_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
                if not os.path.exists(log_file_path):
                    os.makedirs(log_file_path)

                if not os.path.exists(os.path.join(log_file_path, log_file_name)) or cfg["TRAINTEST"]["trainflag"] == "yes":
                    window.logger_handler.closeHandlers()
                    # TBD: change the log file name
                    window.logger_handler.filename = os.path.join(log_file_path, log_file_name)
                    window.logger_handler.setup_logger(SILENT_MODE)
                    window.logger_handler.log_message("*" * 80)
                    window.logger_handler.log_message(f"Experiment name: {cfg['GENERAL']['name']}")
                    window.logger_handler.log_message("*" * 80)

                    window.logger_handler.log_message(
                        f"\nRunning L1O training: {EPOCHS} epochs, {LR} learning rate, {BATCH} batch size, model {cfg['TRAINTEST']['model_name']} - Iteration {curIter} of {maxIter}", (curIter / maxIter))
                    window.logger_handler.log_message(f"Silent Mode is {'ON' if SILENT_MODE else 'OFF'}...")
                    window.logger_handler.log_message(f"Using Model {cfg['TRAINTEST']['model_name']}...")

                model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler,
                                                         _SILENT_RUN=SILENT_MODE,
                                                         excludeSpeciesGroup = idx,
                                                         trainFlag=cfg["TRAINTEST"]["trainflag"])

                logStat.analyzeLog(os.path.join("experiments", cfg["GENERAL"]["folder"], "logs", log_file_name))

            curIter += 1

    window.logger_handler.log_message("\nTraining completed...",1)

    # Release the semaphore when done
    if semaphore is not None:
        semaphore.release()

    # Call the callback function if it's not None
    if callback is not None:
        callback()

def thread_jj3ft(cfg, window, callback=None, semaphore=None):
    if semaphore is not None:
        semaphore.acquire()

    SILENT_MODE = cfg["GENERAL"]["silentmode"]

    curIter = 1
    #I collect all the names and filename string codes for all models on which I have to train
    model_names = []
    model_strings = []

    if cfg["GENERAL"]["type"] == "single":
        inputModelsCount = len(cfg["TRAINTEST"]["epoch"]) * len(cfg["TRAINTEST"]["learning_rate"]) * len(cfg["TRAINTEST"]["batch_size"])
        maxIter = len(cfg["FINETUNING"]["epoch"]) * len(cfg["FINETUNING"]["learning_rate"]) * len(cfg["FINETUNING"]["batch_size"]) * inputModelsCount
        #Input model names
        for EPOCHS, LR, BATCH in product(cfg["TRAINTEST"]["epoch"], cfg["TRAINTEST"]["learning_rate"], cfg["TRAINTEST"]["batch_size"]):
            model_names.append(f'M_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}_exclude_-1.pl')
            model_strings.append(f'{cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{cfg["TRAINTEST"]["model_name"]}-{BATCH}')
    elif cfg["GENERAL"]["type"] == "leaveoneout":
        inputModelsCount = len(cfg["TRAINTEST"]["epoch"]) * len(cfg["TRAINTEST"]["learning_rate"]) * len(cfg["TRAINTEST"]["batch_size"]) * len(cfg["TRAINTEST"]["leaveoneoutspecies"])
        maxIter = len(cfg["FINETUNING"]["epoch"]) * len(cfg["FINETUNING"]["learning_rate"]) * len(cfg["FINETUNING"]["batch_size"]) * len(cfg["TRAINTEST"]["leaveoneoutspecies"]) * inputModelsCount
        #Input model names
        for EPOCHS, LR, BATCH, L1O in product(cfg["TRAINTEST"]["epoch"], cfg["TRAINTEST"]["learning_rate"], cfg["TRAINTEST"]["batch_size"], range(len(cfg["TRAINTEST"]["leaveoneoutspecies"]))):
            model_names.append(
                f'M_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}_exclude_{L1O}.pl')
            model_strings.append(f'{cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{cfg["TRAINTEST"]["model_name"]}-{BATCH}-exclude-{L1O}')

    #Now I run a FineTuning on each of the models for every combination of FT epochs, learning rate and batch size
    for EPOCHS, LR, BATCH, MODEL in product(cfg["FINETUNING"]["epoch"], cfg["FINETUNING"]["learning_rate"], cfg["FINETUNING"]["batch_size"], range(len(model_names))):
        log_file_name = f'log_FT_origin_{model_strings[MODEL]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}.log'
        log_file_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)

        if not os.path.exists(os.path.join(log_file_path, log_file_name)) or cfg["FINETUNING"]["trainflag"] == "yes":
            window.logger_handler.closeHandlers()
            # TBD: change the log file name
            window.logger_handler.filename = os.path.join(log_file_path, log_file_name)
            window.logger_handler.setup_logger(SILENT_MODE)
            window.logger_handler.log_message("*" * 80)
            window.logger_handler.log_message(f"Experiment name: {cfg['GENERAL']['name']}")
            window.logger_handler.log_message("*" * 80)

            window.logger_handler.log_message(f"\nRunning fine tuning: {EPOCHS} epochs, {LR} learning rate, {BATCH} batch size, model {cfg['TRAINTEST']['model_name']} - Iteration {curIter} of {maxIter}", (curIter / maxIter))
            window.logger_handler.log_message(f"Silent Mode is {'ON' if SILENT_MODE else 'OFF'}...")
            window.logger_handler.log_message(f"Using Model {cfg['TRAINTEST']['model_name']}...")

        if cfg["GENERAL"]["type"] == "single":
            model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler, os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model", model_names[MODEL]), input_modelstring=model_strings[MODEL], _SILENT_RUN=SILENT_MODE, trainFlag=cfg["FINETUNING"]["trainflag"])
        elif cfg["GENERAL"]["type"] == "leaveoneout":
            excludedGroupIndex = int(model_names[MODEL].split("_")[-1].split(".")[0].split("-")[-1])
            model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler, os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model", model_names[MODEL]), input_modelstring=model_strings[MODEL], _SILENT_RUN=SILENT_MODE, trainFlag=cfg["FINETUNING"]["trainflag"], excludeSpeciesGroup=excludedGroupIndex)

        logStat.analyzeLog(os.path.join(log_file_path, log_file_name))

        if not os.path.exists(model_location):
            window.logger_handler.log_message("Model not found")
            return()

        curIter += 1

    window.logger_handler.log_message("\nFine Tuning completed...",1)

    # Release the semaphore when done
    if semaphore is not None:
        semaphore.release()

    # Call the callback function if it's not None
    if callback is not None:
        callback()

def thread_jj4(cfg, window, callback=None, semaphore=None):
    if semaphore is not None:
        semaphore.acquire()

    model_names = []
    if cfg["VALIDATION"]["input_model"] in ["both", "tt"]:
    #if "FINETUNING" not in cfg:
        #get the names of all the models in the folder 1.DL_Training/Model whose filename starts with M_<acronym> and ends with .pl
        for file in os.listdir(os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model")):
            if file.startswith(f'M_TT_') and file.endswith('.pl'):
                model_names.append(file)
    #else:
    if cfg["VALIDATION"]["input_model"] in ["both", "ft"]:
        #get the names of all the models in the folder 1.DL_Training/Model whose filename starts with M_<acronym> and ends with .pl
        for file in os.listdir(os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model")):
            if file.startswith('M_FT_') and file.endswith('.pl'):
                model_names.append(file)

    if "VALIDATION" in cfg or cfg["GENERAL"]["type"] == "leaveoneout":
        datasetFolder = os.path.join("experiments", cfg["GENERAL"]["folder"], "0.DataSet")
        validationDataSet = os.path.join(datasetFolder, "dataset_FV.csv")
        window.logger_handler.log_message("******* Validation *******")
    else:
        window.logger_handler.log_message ("No validation dataset found")
        return

    #Now I run a Validation on each of the models
    for idx, model in enumerate(model_names):
        #I have to extract the parameters from the model name
        model_string = model.split("_")
        if "exclude" in model_string:
            #get the integer after "exclude_" and before ".pl"
            exclude = int(model_string[-1].split(".")[0])
        else:
            exclude = -1

        # The log_file_name is the same as the model, but with "inference" replacing "M" and "pl" replaced by "csv"
        log_file_name = model.replace("M", "inference").replace("pl", "csv")
        log_file_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "inferences")
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
        model_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model")

        if os.path.exists(os.path.join(log_file_path, log_file_name)) and cfg["VALIDATION"]["inferflag"] == "no":
            window.logger_handler.log_message(f'Inference #{idx} already present, skipping...', idx / len(model_names))
        else:
            if cfg["GENERAL"]["type"] == "single":
                window.logger_handler.log_message(f'Inference #{idx}...', idx / len(model_names))
                jj4_DLinference.DL_validate(cfg, os.path.join(model_path,model), cfg["TRAINTEST"]["model_name"], validationDataSet, os.path.join(log_file_path, log_file_name), window.logger_handler, torchdevice = cfg["ENVIRONMENT"]["torchdevice"])
            else:
                window.logger_handler.log_message(f'Inference species: {cfg["TRAINTEST"]["leaveoneoutspecies"][exclude]}...', idx / len(model_names))
                jj4_DLinference.DL_validate(cfg, os.path.join(model_path,model), cfg["TRAINTEST"]["model_name"], validationDataSet, os.path.join(log_file_path, log_file_name), window.logger_handler, validationSpecies=cfg["TRAINTEST"]["leaveoneoutspecies"][exclude], torchdevice = cfg["ENVIRONMENT"]["torchdevice"])

    valStatsFolder.analyseValidationFolder(window.logger_handler, os.path.join("experiments", cfg["GENERAL"]["folder"], "inferences"))

    window.logger_handler.log_message("\nInference completed...",1)

    # Release the semaphore when done
    if semaphore is not None:
        semaphore.release()

    # Call the callback function if it's not None
    if callback is not None:
        callback()
