#this part merges embeddings and Labels into a single file
import os
import pandas as pd
from tkinter import messagebox

def labelConfig(cfg, label):
    if label in cfg:
        filter = cfg[label]
    else:
        filter = {}

    if (label + "-Label" in cfg):
        filter["Label"] = cfg[label + "-Label"]
    else:
        filter["Label_0"] = cfg[label + "-Label_0"]
        filter["Label_1"] = cfg[label + "-Label_1"]

    return filter

'''
This function is used to filter a dataframe based on the configuration file
'''
def applyFilters2DF(df, filters, logger):
    if len(filters) == 0:
        return df
    #First I manage the Species
    if filters["includedspecies"] != "":
        df = df[df['Species'].isin(filters["includedspecies"])]
    elif len(filters["excludedspecies"]) > 0:
        df = df[~df['Species'].isin(filters["excludedspecies"])]

    #apply remaining filters to the dataframe depending on the Label
    if "Label" in filters:
        if filters["Label"]["annotation"] != "" and filters["Label"]["reviewed"] != "":
            df = df[(df['Annotation'].isin(filters["Label"]["annotation"])) & (df['Reviewed'].isin(filters["Label"]["reviewed"]))]
        elif filters["Label"]["annotation"] != "":
            df = df[df['Annotation'].isin(filters["Label"]["annotation"])]
        elif filters["Label"]["reviewed"] != "":
            df = df[df['Reviewed'].isin(filters["Label"]["reviewed"])]
    else:
        if filters["Label_0"]["annotation"] != "" and filters["Label_0"]["reviewed"] != "" and filters["Label_1"]["annotation"] != "" and filters["Label_1"]["reviewed"] != "":
            df = df[((df['Annotation'].isin(filters["Label_0"]["annotation"])) & (df['Label'] == 0) & (df['Reviewed'].isin(filters["Label_0"]["reviewed"]))) | ((df['Annotation'].isin(filters["Label_1"]["annotation"])) & (df['Label'] == 1) & (df['Reviewed'].isin(filters["Label_1"]["reviewed"])))]
        elif filters["Label_0"]["annotation"] != "" and filters["Label_0"]["reviewed"] != "" and filters["Label_1"]["annotation"] != "":
            df = df[((df['Annotation'].isin(filters["Label_0"]["annotation"])) & (df['Label'] == 0) & (df['Reviewed'].isin(filters["Label_0"]["reviewed"]))) | ((df['Annotation'].isin(filters["Label_1"]["annotation"]) & (df['Label'] == 1)))]
        elif filters["Label_0"]["annotation"] != "" and filters["Label_0"]["reviewed"] != "" and filters["Label_1"]["reviewed"] != "":
            df = df[(((df['Annotation'].isin(filters["Label_0"]["annotation"])) & (df['Label'] == 0) & (df['Reviewed'].isin(filters["Label_0"]["reviewed"]))) | (df['Label'] == 1 & df['Reviewed'].isin(filters["Label_1"]["reviewed"])))]
        elif filters["Label_0"]["annotation"] != "" and filters["Label_1"]["annotation"] != "" and filters["Label_1"]["reviewed"] != "":
            df = df[((df['Annotation'].isin(filters["Label_0"]["annotation"])) & (df['Label'] == 0)) | (df['Annotation'].isin(filters["Label_1"]["annotation"]) & ((df['Label'] == 1) & (df['Reviewed'].isin(filters["Label_1"]["reviewed"]))))]
        elif filters["Label_0"]["reviewed"] != "" and filters["Label_1"]["annotation"] != "" and filters["Label_1"]["reviewed"] != "":
            df = df[((df['Reviewed'].isin(filters["Label_0"]["reviewed"])) & (df['Label'] == 0)) | ((df['Annotation'].isin(filters["Label_1"]["annotation"]) & df['Label'] == 1 & df['Reviewed'].isin(filters["Label_1"]["reviewed"])))]

    if "balanced" in filters["_01ratio"]:
        logger.log_message ("Balancing the labels....")
        #For each Species, make sure that the number of rows with Label = 1 is the same as the number of rows with Label = 0
        #I need to get the number of rows with Label = 1 and Label = 0 for each species
        for species in df['Species'].unique():
        #get the number of rows with Label = 1 and Label = 0 for each species
            numLabel1 = len(df[(df['Species'] == species) & (df['Label'] == 1)])
            numLabel0 = len(df[(df['Species'] == species) & (df['Label'] == 0)])
            #if the number of rows with Label = 1 is greater than the number of rows with Label = 0, then I need to do nothing
            if numLabel1 > numLabel0:
                logger.log_message (f"Species: {species} - Label 1: {numLabel1} - Label 0: {numLabel0}")
            #if the number of rows with Label = 0 is greater than the number of rows with Label = 1, then I need to remove some rows with Label = 0
            elif numLabel0 > numLabel1:
                #get the number of rows with Label = 0 that need to be removed
                numToRemove = numLabel0 - numLabel1
                logger.log_message (f"Species: {species} - Dropped: {numToRemove} label_0 rows")
                #get the index of the rows with Label = 0 that need to be removed
                indexToRemove = df[(df['Species'] == species) & (df['Label'] == 0)].sample(n=numToRemove, random_state=42).index
                #remove the rows with Label = 0 that need to be removed
                df = df.drop(indexToRemove)

    #If is set the Percentage filter, I need to flag all rows with "bin" = "train" or "test" or "val"
    if "percent" in filters:
        #shuffle the dataset
        df = df.sample(frac=1, random_state=42)
        #divide the rows that have label = 1 in 3 non overlapping groups: train, test, val respecting the percentages of the configuration percent[0] for train and percent[1] for test
        df.loc[df['Label'] == 1, 'bin'] = pd.qcut(df[df['Label'] == 1].index, q=[0, filters["percent"][0], filters["percent"][0] + filters["percent"][1], 1], labels=["train", "test", "val"])
        #same with rows that have label = 0 but respecting the percentages of the configuration percent[0] for train and percent[1] for test
        df.loc[df['Label'] == 0, 'bin'] = pd.qcut(df[df['Label'] == 0].index, q=[0, filters["percent"][0], filters["percent"][0] + filters["percent"][1], 1], labels=["train", "test", "val"])
    return df

'''
This function is used to ask the user if a file has to be generated
'''
def askForGenerateFile(path, flag = False):
    if flag == "no":
        return False
    dataSetName = os.path.basename(path)
    if os.path.exists(path) and flag == "ask":
        answer = messagebox.askyesno("Warning", f"Dataset {dataSetName} already present. Do you want to regenerate it?", icon='warning')
        return answer
    else:
        return True

'''
This function creates three datasets: Train/Test, FineTuning, and Final Validation
If the experiment is a leave-one-out, then the final validation dataset is created during the training process
with the species that are not used for training
'''
def filterDataSet(cfg, logger):

    pre_input_prc_folder_out = os.path.join("experiments/" + cfg["GENERAL"]["folder"], "0.DataSet")
    if not os.path.exists(pre_input_prc_folder_out):
        os.makedirs(pre_input_prc_folder_out)

    #get the parameters from the configuration file
    datasetFile = cfg["GENERAL"]["originaldataset"]

    if cfg["tmp"]["createTT"] or cfg["tmp"]["createFT"] or cfg["tmp"]["createFV"]:
        #Load the dataset into a dataframe
        #if datasetFile is a list, then I need to load all the datasets and merge them into a single dataframe
        logger.log_message (f"Loading original datasets into dataframes...", progress = 0.1)
        if type(datasetFile) == list:
            jj_all_data = pd.DataFrame()
            for file in datasetFile:
                df = pd.read_csv(file, delimiter=',', low_memory=False)
                jj_all_data = pd.concat([jj_all_data, df], ignore_index=True)
        else:
            jj_all_data = pd.read_csv(datasetFile, delimiter=',', low_memory=False)

        #Make sure the column Label is int where it is not nan
        logger.log_message (f"Checking labels to 0/1/-1 (-1 means NO LABEL)...", progress = 0.25)
        jj_all_data['Label'] = jj_all_data['Label'].fillna(-1)

        #Convert the Label column to int
        jj_all_data['Label'] = jj_all_data['Label'].astype(int)

        #Free some space
        #Drop these columns: protId, Protein names, Gene Names, Organism, Length, Caution, Gene Ontology (molecular function)
        logger.log_message (f"Dropping unused columns...")
        jj_all_data = jj_all_data.drop(columns=["protID", "Sequence", "Protein names", "Gene Names", "Organism", "Length", "Caution", "Gene Ontology (molecular function)"])

        #jj_all_data.set_index('Entry Name', inplace=True)

        #print total number of rows
        logger.log_message (f"Total number of rows in the dataset: {len(jj_all_data)}")
        logger.log_message ("*" * 80)

        # I apply three different filters, one for Training/Test, one for FineTuning (if necessary), and one for final Validation

        if cfg["tmp"]["createTT"]:
            # ******* TRAIN/TEST FILTERS
            jj_all_data_TT = applyFilters2DF(jj_all_data, labelConfig(cfg, "TrainTest"), logger) #TT Test Train
            #Save the datasets to csv files
            #jj_all_data_TT = jj_all_data_TT.drop(columns=["Annotation", "Reviewed"])
            outFile = os.path.join(pre_input_prc_folder_out, "dataset_TT.csv")
            jj_all_data_TT.to_csv(outFile, ",", mode="w", header=True, index=False)
            logger.log_message(f"TT Dataset created...", progress=0.5)
        else:
            jj_all_data_TT = pd.read_csv(os.path.join(pre_input_prc_folder_out, "dataset_TT.csv"), delimiter=',', low_memory=False)
            logger.log_message(f"TT Dataset loaded...", progress=0.5)


        if cfg["tmp"]["createFT"]:
            #******* FINETUNING FILTERS
            if "FineTuning" in cfg:
                #I need to remove from jj_all_data the rows that are already in jj_all_data_TT
                jj_all_data_FT = jj_all_data[~jj_all_data.index.isin(jj_all_data_TT.index)]
                jj_all_data_FT = applyFilters2DF(jj_all_data_FT, labelConfig(cfg, "FineTuning"), logger) #FT Fine Tuning
                outFile = os.path.join(pre_input_prc_folder_out, "dataset_FT.csv")
                jj_all_data_FT.to_csv(outFile, ",", mode="w", header = True, index=False)
                logger.log_message(f"FT Dataset created...", progress=0.7)
        else:
            jj_all_data_FT = pd.read_csv(os.path.join(pre_input_prc_folder_out, "dataset_FT.csv"), delimiter=',', low_memory=False)
            logger.log_message(f"FT Dataset loaded...", progress=0.7)

        if cfg["tmp"]["createFV"] or cfg["GENERAL"]["type"] == "leaveoneout":
            if cfg["GENERAL"]["type"] == "leaveoneout":
                if "Validation" in cfg:
                    jj_all_data_FV = applyFilters2DF(jj_all_data, labelConfig(cfg, "Validation"), logger)  # FV Final Validation - The code will make sure that validation and train/test never overlap
                else:
                    jj_all_data_FV = applyFilters2DF(jj_all_data, {}, logger)  # FV Final Validation - The code will make sure that validation and train/test never overlap
            #******* VALIDATION FILTERS
            else:
                # I need to remove from jj_all_data the rows that are already in jj_all_data_TT and jj_all_data_FT
                jj_all_data_FV = jj_all_data[~jj_all_data.index.isin(jj_all_data_TT.index)]
                if "FineTuning" in cfg:
                    jj_all_data_FV = jj_all_data_FV[~jj_all_data_FV.index.isin(jj_all_data_FT.index)]
                jj_all_data_FV = applyFilters2DF(jj_all_data_FV, labelConfig(cfg, "Validation"), logger) #FV Final Validation

            if len(jj_all_data_FV) > 0:
                outFile = os.path.join(pre_input_prc_folder_out, "dataset_FV.csv")
                jj_all_data_FV.to_csv(outFile, ",", mode="w", header = True, index=False)
                logger.log_message(f"FV Dataset created...", progress=1)

        logger.log_message("*" * 80)
        logger.log_message(f"Dataset Summary:")
        logger.log_message("*" * 80)
        logger.log_message("Training Test (TT):")
        logger.log_message(f"- Total number of Label 1 in the Test/Train dataset: {len(jj_all_data_TT[jj_all_data_TT['Label'] == 1])}")
        logger.log_message(f"- Total number of Label 0 in the Test/Train dataset: {len(jj_all_data_TT[jj_all_data_TT['Label'] == 0])}")
        # print the total number of rows that have Label = 1 and bin = train
        logger.log_message(f"\t\tTrain: 1: {len(jj_all_data_TT[(jj_all_data_TT['Label'] == 1) & (jj_all_data_TT['bin'] == 'train')])} - 0: {len(jj_all_data_TT[(jj_all_data_TT['Label'] == 0) & (jj_all_data_TT['bin'] == 'train')])}")
        logger.log_message(f"\t\tTest: 1: {len(jj_all_data_TT[(jj_all_data_TT['Label'] == 1) & (jj_all_data_TT['bin'] == 'test')])} - 0: {len(jj_all_data_TT[(jj_all_data_TT['Label'] == 0) & (jj_all_data_TT['bin'] == 'test')])}")
        logger.log_message(f"\t\tVal: 1: {len(jj_all_data_TT[(jj_all_data_TT['Label'] == 1) & (jj_all_data_TT['bin'] == 'val')])} - 0: {len(jj_all_data_TT[(jj_all_data_TT['Label'] == 0) & (jj_all_data_TT['bin'] == 'val')])}")
        logger.log_message("*" * 80)
        if "FineTuning" in cfg:
            logger.log_message("Fine Tuning (FT):")
            logger.log_message(f"Total number of Label 1 in the FineTuning dataset: {len(jj_all_data_FT[jj_all_data_FT['Label'] == 1])}")
            logger.log_message(f"Total number of Label 0 in the FineTuning dataset: {len(jj_all_data_FT[jj_all_data_FT['Label'] == 0])}")
            logger.log_message("*" * 80)
        if "Validation" in cfg or cfg["GENERAL"]["type"] == "leaveoneout":
            logger.log_message(f"Total number of rows in the Validation dataset: {len(jj_all_data_FV)}")
            logger.log_message(f"Total number of Label 1 in the Validation dataset: {len(jj_all_data_FV[jj_all_data_FV['Label'] == 1])}")
            logger.log_message(f"Total number of Label 0 in the Validation dataset: {len(jj_all_data_FV[jj_all_data_FV['Label'] == 0])}")
            logger.log_message("*" * 80)
    else:
        logger.log_message("No dataset creation requested...", progress=1)
    return

def main(cfg, logger):
    filterDataSet(cfg, logger)

if __name__ == "__main__":
    main()

