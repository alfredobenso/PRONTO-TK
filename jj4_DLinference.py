import importlib
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import os
from model_util import *

'''
This function is used to validate a model using a dataset. 
It takes as input the model, the dataset and the output file.
'''
def DL_validate(cfg, inputModel, model_name, inputData, outputFile, logger, validationSpecies = [], torchdevice = "mps"):

    model_dir=inputModel

    logger.log_message(f"Model: {os.path.basename(model_dir)}")
    logger.log_message(f"Output: {os.path.basename(outputFile)}")

    jj_all_data = pd.read_csv(inputData, low_memory=False)

    if len(validationSpecies) > 0:
        #I have to filter the dataframe to keep only the species in validationSpeciesGroup
        #recreate index
        jj_all_data = jj_all_data[jj_all_data['Species'].isin(validationSpecies)]
        jj_all_data = jj_all_data.reset_index(drop=True)

    if jj_all_data.empty:
        logger.log_message (f"Not enough data to validate the Species {validationSpecies}. Skipping...");
        return

    jjdata = torch.tensor(jj_all_data.loc[:, '0':'1023'].values)

    device = torch.device('cuda:0' if torch.cuda.is_available() else torchdevice)

    # device = torch.device("cpu")
    model_func = globals()[cfg["TRAINTEST"]["model_name"]]
    net = model_func().float().to(device)

    model_dir = os.path.join(os.getcwd(), inputModel)
    state_dict = torch.load(model_dir, map_location=device)
    net.load_state_dict(state_dict['model'])

    data_iter = torch.utils.data.DataLoader(jjdata, batch_size=2048, shuffle=False)

    net.eval()
    logits_all=[]
    soft_max=nn.Softmax(dim=1)
    #return data_iter
    from tqdm import tqdm
    for rep in tqdm(data_iter):
        rep=rep.float().to(device)
        with torch.no_grad():
            logits=net(rep)
            #logits_all.append(logits/cl_dict[args.species])
            logits_all.append(logits / 1)

    #print(logits_all)
    logits_in_one=torch.vstack(logits_all)
    smres = soft_max(logits_in_one)
    df = pd.DataFrame(smres.cpu().detach().numpy())

    if "Label" in jj_all_data.columns:
        df['Label'] = jj_all_data['Label']

    if "Entry Name" in jj_all_data.columns:
        df['Entry Name'] = jj_all_data['Entry Name']

    if "Annotation" in jj_all_data.columns:
        df['Annotation'] = jj_all_data['Annotation']

    #if jj_all_data has a column named "Specie", add it to the output dataframe
    df['Species'] = jj_all_data['Species']

    #if the outputFile already exists, append this dataframe to it
    if os.path.exists(outputFile):
        df.to_csv(outputFile, mode='w', header=True, index=False)
    else:
        df.to_csv(outputFile, header=True, index=False)

    return smres

if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()

    #TO UPDATE with new configuration
    file = filedialog.askopenfilename(initialdir = "./experiments/_configurations/", title="Select experiment configuration file ", filetypes=[("Text files", "*.ini")])
    cfgFile = os.path.basename(file).replace(".ini", "")
    #import the configuration file
    cfg = importlib.import_module("experiments._configurations." + cfgFile)

    model = filedialog.askopenfilename(initialdir = "./experiments/" + cfg.expConf['Folder'] + "/1.DL_Training/Model/", title="Select model ", filetypes=[("Text files", "*.pl")])
    dataset = filedialog.askopenfilename(initialdir = "./experiments/" + cfg.expConf['Folder'] + "/0.DataSet/", title="Select dataset ", filetypes=[("Text files", "*.csv")])
    file_name = f'inference_TT_{cfg.expConf["Acronym"]}_model_{cfg.expConf["model_name"]}_{cfg.expConf["Type"]}.csv'


    file_path = os.path.join("experiments", cfg.expConf["Folder"] + "/validations")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    DL_validate(cfg, model, cfg.expConf["model_name"], dataset, os.path.join(file_path, file_name), torchdevice = cfg["ENVIRONMENT"]["torchdevice"])
