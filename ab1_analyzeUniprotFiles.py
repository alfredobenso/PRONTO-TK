import os
import pandas as pd
def analyzeEmbeddingFiles(cfg, logger):
    #***********************************************************************************************
    val_folder = os.path.join(cfg["EMBEDDINGS"]["uniprotfolder"],"embeddings")
    #***********************************************************************************************
    # Initialize a dictionary to store the species counts
    total_species_counts = {}

    emb_files = [file for file in os.listdir(val_folder) if file.endswith('_embeddings.csv')]
    embCount = len(emb_files)

    #for each file in the folder whose name end with "_embeddings.csv"
    for idx, file in enumerate(emb_files):
        if file.endswith("_embeddings.csv"):
            logger.log_message(f"Analyzing file {file}", idx/embCount)

            #read the file into the first colum of df_si
            df = pd.read_csv(os.path.join(val_folder, file), delimiter=',', low_memory=False)

            _FILES_WITH_LABELS = "Label" in df.columns
            _ADD_LABELS = cfg["EMBEDDINGS"]["addlabelsflag"]

            #Add labels from filename
            if not _FILES_WITH_LABELS and _ADD_LABELS=="yes":
                logger.log_message(f"Adding labels .....")

                #Remove the first column
                df = df.iloc[:, 1:]
                #If the file name has the word NOT, add a column called Label initialized to 0 as first column
                if "NOT" in file:
                    df.insert(0, "Label", 0)
                else:
                    df.insert(0, "Label", 1)

                #Save the modified file in a new file that starts with label_0_ or label_1_
                if "NOT" in file:
                    df.to_csv(os.path.join(val_folder, f"{file}"), index=False)
                else:
                    df.to_csv(os.path.join(val_folder, f"{file}"), index=False)

            _FILES_WITH_LABELS = "Label" in df.columns

            if "Annotation" not in df.columns:
                if "manual" in file:
                    df.insert(0, "Annotation", "manual")
                elif "automatic" in file:
                    df.insert(0, "Annotation", "automatic")
                else:
                    df.insert(0, "Annotation", "n/a")

            # Count the number of labels (1/0) for each Species - Divide between Reviewed and unreviewed looking at the Reviewed column
            if _FILES_WITH_LABELS:
                # for each Species
                for annotation in ("manual", "automatic", "n/a"):
                    for species in df['Species'].unique():
                        zero_rev_count = df[
                            (df['Label'] == 0) & (df['Reviewed'] == "reviewed") & (df['Species'] == species) & (
                                        df['Annotation'] == annotation)].shape[0]
                        one_rev_count = df[
                            (df['Label'] == 1) & (df['Reviewed'] == "reviewed") & (df['Species'] == species) & (
                                        df['Annotation'] == annotation)].shape[0]
                        zero_unrev_count = df[
                            (df['Label'] == 0) & (df['Reviewed'] == "unreviewed") & (df['Species'] == species) & (
                                        df['Annotation'] == annotation)].shape[0]
                        one_unrev_count = df[
                            (df['Label'] == 1) & (df['Reviewed'] == "unreviewed") & (df['Species'] == species) & (
                                        df['Annotation'] == annotation)].shape[0]

                        if species not in total_species_counts:
                            total_species_counts[species] = {"manual": {"0U": 0, "1U": 0, "0R": 0, "1R": 0},
                                                             "automatic": {"0U": 0, "1U": 0, "0R": 0, "1R": 0},
                                                             "n/a": {"0U": 0, "1U": 0, "0R": 0, "1R": 0}}
                            total_species_counts[species][annotation]["0U"] = zero_unrev_count
                            total_species_counts[species][annotation]["1U"] = one_unrev_count
                            total_species_counts[species][annotation]["0R"] = zero_rev_count
                            total_species_counts[species][annotation]["1R"] = one_rev_count
                        else:
                            total_species_counts[species][annotation]["0U"] += zero_unrev_count
                            total_species_counts[species][annotation]["1U"] += one_unrev_count
                            total_species_counts[species][annotation]["0R"] += zero_rev_count
                            total_species_counts[species][annotation]["1R"] += one_rev_count
            else:
                total_species_counts = df['Species'].value_counts()

    # Convert the total_species_counts dictionary to a DataFrame adding the key as column
    # Flatten the nested dictionary
    # Create an empty DataFrame with the first-level keys as index and the column named "Species"
    df_output = pd.DataFrame(index=total_species_counts.keys(), columns=["Species"])

    # Iterate over the nested dictionaries
    for first_key, inner_dict in total_species_counts.items():
        # Create columns for each combination of subkey and 'a'/'b' values
        for subkey, sub_dict in inner_dict.items():
            for k, v in sub_dict.items():
                column_name = f"{subkey}_{k}"
                if v > 0:
                    df_output.loc[first_key, column_name] = v

    #replace nan with 0
    df_output.fillna(0, inplace=True)

    # Display the DataFrame
    #rename the columns present in the df to make them more readable
    #replace U with _unreviewed and R with _reviewed in each column name
    df_output.columns = df_output.columns.str.replace('U', '_unreviewed')
    df_output.columns = df_output.columns.str.replace('R', '_reviewed')

    # Write the DataFrame to a CSV file
    df_output.to_csv(os.path.join(cfg["EMBEDDINGS"]["uniprotfolder"], cfg["EMBEDDINGS"]["outputdatasetname"] + "_species_counts.csv"), index=True)
    logger.log_message(f"File total_species_counts.csv saved in {val_folder}")

    return
