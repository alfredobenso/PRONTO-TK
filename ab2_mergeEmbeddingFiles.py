import os
import shutil

import pandas as pd
def mergeEmbeddings(cfg, logger):
    #***********************************************************************************************
    src_folder = os.path.join(cfg["UNIPROT"]["go_folder"],"embeddings")
    #***********************************************************************************************

    #I want to MERGE all csv files in the folder src_folder whose name ends with "_embeddings.csv" and starts with label_ into a single csv file
    #When I take a file, its name contains the mode (manual/automatic)
    #Before merging a file, you add a new column called "Annotation" with the value of the mode
    #The new file will be called "merged_embeddings.csv"
    #The new file will be saved in the same folder as the original files
    #The new file will contain all the columns of the original files
    #The new file will contain all the rows of the original files
    #The new file will contain the same number of columns as the original files plus the new column "Annotation"
    #if in val folder there is only one file ending with ".embeddings.csv" then copy it to os.path.join(cfg["UNIPROT"]["go_folder"], cfg["UNIPROT"]["datasetname"] + ".embeddings.dataset.csv")
    if len([file for file in os.listdir(src_folder) if file.endswith('.embeddings.csv')]) == 1:
        for file in os.listdir(src_folder):
            if file.endswith(".embeddings.csv"):
                logger.log_message(f"Copying file {file} to {os.path.join(cfg['UNIPROT']['go_folder'], cfg['UNIPROT']['datasetname'] + '.embeddings.dataset.csv')}")
                shutil.copy(os.path.join(src_folder, file), os.path.join(cfg['UNIPROT']['go_folder'], cfg['UNIPROT']['datasetname'] + '.embeddings.dataset.csv'))
                logger.log_message(f"Final embeddings dataset ready: {file}")
    else:
        for file in os.listdir(src_folder):
            if file.endswith(".embeddings.csv"):
                logger.log_message(f"Processing file {file}")
                df = pd.read_csv(os.path.join(src_folder, file), delimiter=',', low_memory=False)

                # Drop unused columns if they exist
                unused_columns = ['Entry', 'Protein names', 'Gene Names', 'Organism', 'Length', 'Caution', 'Gene Ontology (molecular function)']
                df.drop(columns=[col for col in unused_columns if col in df.columns], inplace=True)

                filename = os.path.join(cfg["UNIPROT"]["go_folder"], cfg["UNIPROT"]["datasetname"] + ".embeddings.dataset.csv")
                #if filename exists. filename includes the path to the file

                if os.path.exists(filename):
                    df.to_csv(filename, mode='a', header=False, index=False)
                else:
                    df.to_csv(filename, mode='w', header=True, index=False)
        logger.log_message(f"Final embeddings dataset ready: {filename}")


    return

if __name__ == "__main__":

    import logging
    import sys

    class CustomLogger(logging.Logger):
        def __init__(self, name, level=logging.NOTSET):
            super().__init__(name, level)

        def log_message(self, message, *args, **kwargs):
            print(message)

    cfg = {
        "UNIPROT": {
            "go_folder": "Original Input//RBPnew",
            "datasetname": "RBP"
        }
    }

    logger = CustomLogger('test')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    mergeEmbeddings(cfg, logger)
