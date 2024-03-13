import os
import pandas as pd
def analyzeEmbeddingFiles(cfg, logger):
    #***********************************************************************************************
    val_folder = os.path.join(cfg["UNIPROT"]["go_folder"], "embeddings")
    #***********************************************************************************************
    # Initialize a dictionary to store the species counts
    total_species_counts = {}

    emb_files = [file for file in os.listdir(val_folder) if file.endswith('.embeddings.csv')]
    embCount = len(emb_files)

    #for each file in the folder whose name end with "_embeddings.csv"
    for idx, file in enumerate(emb_files):
        if file.endswith(".embeddings.csv"):
            logger.log_message(f"Analyzing file {file}", idx/embCount)

            #read the file into the first colum of df_si
            df = pd.read_csv(os.path.join(val_folder, file), delimiter=',', low_memory=False)

            if "Annotation" not in df.columns:
                df.insert(0, "Annotation", "n/a")

            # Count the number of labels (1/0) for each Species - Divide between Reviewed and unreviewed looking at the Reviewed column
            if "Label" in df.columns:
                df["Label"] = df["Label"].astype(int)

                # Group by 'Species', 'Annotation', 'Label', and 'Reviewed' and count the number of rows in each group
                grouped_counts = df.groupby(['Species', 'Annotation', 'Label', 'Reviewed']).size()

                # Convert the GroupBy object to a DataFrame
                df_grouped_counts = grouped_counts.reset_index(name='Count')

                # Pivot the DataFrame to get the desired format
                df_pivot = df_grouped_counts.pivot_table(index=['Species'], columns=['Label', 'Annotation', 'Reviewed'],
                                                         values='Count', fill_value=0)

                # Flatten the MultiIndex columns
                df_pivot.columns = [''.join(str(col)) for col in df_pivot.columns]

                # Convert the index to columns
                df_output = df_pivot.reset_index()


    df_output.fillna(0, inplace=True)

    # Display the DataFrame
    #rename the columns present in the df to make them more readable
    #replace U with _unreviewed and R with _reviewed in each column name
    df_output.columns = df_output.columns.str.replace('U', '_unreviewed')
    df_output.columns = df_output.columns.str.replace('R', '_reviewed')

    # Write the DataFrame to a CSV file
    df_output.to_csv(os.path.join(cfg["UNIPROT"]["go_folder"], cfg["UNIPROT"]["datasetname"] + "_species_counts.csv"), index=True)
    logger.log_message(f"File total_species_counts.csv saved in {val_folder}")

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
            "go_folder": "Original Input//tmp",
            "datasetname": "Terrabacteria"
        }
    }

    logger = CustomLogger('test')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    analyzeEmbeddingFiles(cfg, logger)
