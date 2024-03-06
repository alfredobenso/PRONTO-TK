import os
import pandas as pd
def mergeEmbeddings(cfg, logger):
    #***********************************************************************************************
    val_folder = os.path.join(cfg["EMBEDDINGS"]["uniprotfolder"],"embeddings")
    #***********************************************************************************************

    #I want to MERGE all csv files in the folder val_folder whose name ends with "_embeddings.csv" and starts with label_ into a single csv file
    #When I take a file, its name contains the mode (manual/automatic)
    #Before merging a file, you add a new column called "Annotation" with the value of the mode
    #The new file will be called "merged_embeddings.csv"
    #The new file will be saved in the same folder as the original files
    #The new file will contain all the columns of the original files
    #The new file will contain all the rows of the original files
    #The new file will contain the same number of columns as the original files plus the new column "Annotation"
    for file in os.listdir(val_folder):
        if file.endswith("_embeddings.csv"):
            logger.log_message(f"Processing file {file}")
            df = pd.read_csv(os.path.join(val_folder, file), delimiter=',', low_memory=False)

            filename = os.path.join(cfg["EMBEDDINGS"]["uniprotfolder"], cfg["EMBEDDINGS"]["outputdatasetname"] + "_embeddings_dataset.csv")
            if filename in os.listdir(val_folder):
                df.to_csv(filename, mode='a', header=False, index=False)
            else:
                df.to_csv(filename, mode='w', header=True, index=False)

    return
