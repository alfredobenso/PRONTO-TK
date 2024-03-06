import json
import os.path
import os
import threading
from tkinter import messagebox
from tooltips import tooltips
import customtkinter as ctk
from PIL import Image, ImageTk
import configparser
from CTkToolTip import CTkToolTip
from itertools import product
import webbrowser
import jj2_prepareInputFiles
from process import myProcess
from threads import *

class MainWindow:
    '''
    This is the constructor method where the main window is initialized, and several UI components are created and packed into the window. It also reads the configuration file and sets up the initial state of the application.
    '''
    def __init__(self):
        self.window = ctk.CTk()
        self.cfg_filename = ctk.filedialog.askopenfilename(title="Select configuration file", initialdir=os.path.join("experiments", "_configurations"),filetypes=(("INI files", "*.ini"), ("All files", "*.*")))

        if not self.cfg_filename:
            self.cfg = {}
            raise Exception("No configuration file selected - TBD: handle this case properly!")
        else:
            self.cfg = self.read_configuration(self.cfg_filename)

        self.window.geometry("1100x400")
        self.window.title("RBP ToolKit")

        my_font = ctk.CTkFont(family="Helvetica", size = 16, weight = "bold")
        label = ctk.CTkLabel(self.window, text=f"{self.cfg['GENERAL']['name']}\n{self.cfg['GENERAL']['type'].capitalize()} Experiment", font=my_font)
        label.pack(pady=(5, 10))

        button_frame = ctk.CTkFrame(self.window)
        button_frame.pack()

        # Add the buttons to the new frame
        button_image = ctk.CTkImage(Image.open("assets/settings.icns"), size=(35, 35))
        self.buttonCfg = ctk.CTkButton(button_frame, text=f"Conf. exp.\n {self.cfg['GENERAL']['acronym']}", fg_color="white", text_color="black", image=button_image, command=self.open_cfg_window)
        self.buttonCfg.pack(side=ctk.LEFT)

        button_image = ctk.CTkImage(Image.open("assets/execute.icns"), size=(35, 35))
        self.buttonRun = ctk.CTkButton(button_frame, text="Run Exp.", fg_color="white", text_color="black", image=button_image, command=self.run_full_pipeline)
        self.buttonRun.pack(side=ctk.LEFT, padx=(10,0))

        button_image = ctk.CTkImage(Image.open("assets/refresh.png"), size=(20, 20))
        self.buttonRun = ctk.CTkButton(button_frame, text="Refresh", fg_color="white", text_color="black", image=button_image, command=self.checkIO)
        self.buttonRun.pack(side=ctk.LEFT, padx=(10,0))

        textBoxHeight = 4

        frameDATA = ctk.CTkFrame(self.window)
        self.buttonUPW = ctk.CTkButton(frameDATA, text="", width=60, image=ctk.CTkImage(Image.open("assets/uniprot.png"), size=(60, 60)))
        self.buttonUPW.pack(fill=ctk.BOTH, side=ctk.LEFT)
        self.textboxDATA_UP = ctk.CTkButton(frameDATA, height=textBoxHeight, text="1. Uniprot Dataset", command=lambda: self.open_finder(os.path.dirname(self.cfg["GENERAL"]["originaldataset"][0])))
        self.textboxDATA_UP.pack(fill=ctk.BOTH, pady=25, padx=(5,5), side=ctk.LEFT)
        self.buttonEMB = ctk.CTkButton(frameDATA, text="2. Embeddings", image=ctk.CTkImage(Image.open("assets/embeddings.png"), size=(60, 60)), command=self.run_pipeline_stage_jj1)
        self.buttonEMB.pack(fill=ctk.BOTH, side=ctk.LEFT)
        self.textboxDATA_IN = ctk.CTkButton(frameDATA, height=textBoxHeight, text="Original DataSet", command=lambda: self.open_finder(os.path.dirname(self.cfg["GENERAL"]["originaldataset"][0])))
        self.textboxDATA_IN.pack(fill=ctk.BOTH, pady=25, padx=(5,5), side=ctk.LEFT)
        self.buttonDATA = ctk.CTkButton(frameDATA, text="3. Data Filters", image=ctk.CTkImage(Image.open("assets/data.png"), size=(60, 60)), command = self.run_pipeline_stage_jj2)
        self.buttonDATA.pack(fill=ctk.BOTH, side=ctk.LEFT)
        self.textboxDATA_OUT = ctk.CTkButton(frameDATA, height=textBoxHeight, text="TT/FT/FV Datasets", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
        self.textboxDATA_OUT.pack(fill=ctk.BOTH, pady=25, padx=(5, 5), side=ctk.LEFT)
        frameDATA.pack(expand=True, pady=(20, 20))

        framePIPE = ctk.CTkFrame(self.window)
        self.textboxTT_DATA = ctk.CTkButton(framePIPE, height=textBoxHeight, text="TT Dataset", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
        self.textboxTT_DATA.pack(fill=ctk.BOTH, pady=25, padx=(5, 5), side=ctk.LEFT)
        self.buttonTT = ctk.CTkButton(framePIPE, text="1. Training/Test", image=ctk.CTkImage(Image.open("assets/TT.png"), size=(60, 60)), command=self.run_pipeline_stage_jj3)
        self.buttonTT.pack(fill=ctk.BOTH, side=ctk.LEFT)

        # Create a new frame for the button and textboxes
        frameFT = ctk.CTkFrame(framePIPE)
        frameFT.pack(side=ctk.LEFT, expand=True)

        # Add the textboxes to the new frame
        self.textboxTT_MOD = ctk.CTkButton(frameFT, height=textBoxHeight, text="Model", command=lambda: self.open_finder(
                                               os.path.join("experiments", self.cfg["GENERAL"]["folder"],"1.DL_Training", "Model")))
        self.textboxTT_MOD.pack(fill=ctk.BOTH, pady=(25, 2), padx=(5, 5), side=ctk.TOP)

        self.textboxFT_DATA = ctk.CTkButton(frameFT, height=textBoxHeight, text="FT Dataset",command=lambda: self.open_finder(
                                              os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
        self.textboxFT_DATA.pack(fill=ctk.BOTH, pady=(2, 25), padx=(5, 5), side=ctk.TOP)

        self.buttonFT = ctk.CTkButton(framePIPE, text="2. Fine Tuning", image=ctk.CTkImage(Image.open("assets/FT.png"), size=(60, 60)), command=self.run_pipeline_stage_jj3ft)
        self.buttonFT.pack(fill=ctk.BOTH, side=ctk.LEFT)

        # Create a new frame for the button and textboxes
        frameFV = ctk.CTkFrame(framePIPE)
        frameFV.pack(side=ctk.LEFT, expand=True)

        # Add the textboxes to the new frame
        self.textboxFT_MOD = ctk.CTkButton(frameFV, height=textBoxHeight, text="Fine Tuned Model", command=lambda: self.open_finder(
                                               os.path.join("experiments", self.cfg["GENERAL"]["folder"],"1.DL_Training", "Model")))
        self.textboxFT_MOD.pack(fill=ctk.BOTH, pady=(25, 2), padx=(5, 5), side=ctk.TOP)

        self.textboxFV_DATA = ctk.CTkButton(frameFV, height=textBoxHeight, text="FV Dataset",command=lambda: self.open_finder(
                                              os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
        self.textboxFV_DATA.pack(fill=ctk.BOTH, pady=(2, 25), padx=(5, 5), side=ctk.TOP)


        self.buttonFV = ctk.CTkButton(framePIPE, text="3. Inference", image=ctk.CTkImage(Image.open("assets/FV.png"), size=(60, 60)), command=self.run_pipeline_stage_jj4)
        self.buttonFV.pack(fill=ctk.BOTH, side=ctk.LEFT)

        self.textboxRESULTS = ctk.CTkButton(framePIPE, height=textBoxHeight, fg_color="white", text_color="black", text = "Results", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"],"results")))
        self.textboxRESULTS.pack(fill=ctk.BOTH, pady=(25,2), padx=(5, 5), side=ctk.TOP)
        self.textboxLOGS = ctk.CTkButton(framePIPE, height=textBoxHeight, fg_color="white", text_color="black", text="Logs", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"],"logs")))
        self.textboxLOGS.pack(fill=ctk.BOTH, pady=(2,2), padx=(5, 5), side=ctk.TOP)
        self.textboxVAL = ctk.CTkButton(framePIPE, height=textBoxHeight, fg_color="white", text_color="black", text="Inferences", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"],"inferences")))
        self.textboxVAL.pack(fill=ctk.BOTH, pady=(2,25), padx=(5, 5), side=ctk.TOP)

        framePIPE.pack(expand=True, pady=(0, 20))

        self.checkIO()

    '''
    This method is called when the user clicks on the "Conf. exp." button. 
    It opens a new window where the user can see and edit the configuration file.
    '''
    def read_configuration(self, filename):
        cfg = {}
        tmpCfg = self.read_config(self.cfg_filename)
        for section in tmpCfg.sections():
            cfg[section] = {}
            for key in tmpCfg[section]:
                try:
                    cfg[section][key] = json.loads(tmpCfg[section][key])
                except json.JSONDecodeError as e:
                    cfg[section][key] = tmpCfg[section][key]
        return cfg

    '''
    This method checks the existence of certain files and directories and updates 
    the UI components accordingly.    
    '''
    def checkIO(self):

        #If there are files in the cfg["EMBEDDINGS"]["uniprotfolder"]/downloads, the textbox will be green
        #You need to check if there are files ending with tsv, not if the folder exists
        if len([file for file in os.listdir(os.path.join(self.cfg["EMBEDDINGS"]["uniprotfolder"], "downloads")) if file.endswith('.tsv')]) > 0:
            #textbox background green
            self.textboxDATA_UP.configure(fg_color="green")
            self.buttonEMB.configure(state=ctk.NORMAL)
        else:
            #textbox background red
            self.textboxDATA_UP.configure(fg_color="red")
            self.buttonEMB.configure(state=ctk.DISABLED)

        allOk = True
        for dsFile in self.cfg["GENERAL"]["originaldataset"]:
            if not os.path.exists(dsFile):
                allOk = False
        if not allOk:
            #textbox background red
            self.textboxDATA_IN.configure(fg_color="red")
            self.buttonDATA.configure(state=ctk.DISABLED)
        else:
            #textbox background green
            self.textboxDATA_IN.configure(fg_color="green")
            self.buttonDATA.configure(state=ctk.NORMAL)

        allOk = True
        datasetFolder = os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")
        if not os.path.exists(os.path.join(datasetFolder, "dataset_TT.csv")):
            self.textboxTT_DATA.configure(fg_color="red")
            self.buttonTT.configure(state=ctk.DISABLED)
            allOk = False
        else:
            self.textboxTT_DATA.configure(fg_color="green")
            self.buttonTT.configure(state=ctk.NORMAL)

        if not os.path.exists(os.path.join(datasetFolder, "dataset_FT.csv")) and "FineTuning" in self.cfg:
            self.textboxFT_DATA.configure(fg_color="red")
            self.buttonFT.configure(state=ctk.DISABLED)
            allOk = False
        else:
            self.textboxFT_DATA.configure(fg_color="green")
            self.buttonFT.configure(state=ctk.NORMAL)

        if not os.path.exists(os.path.join(datasetFolder, "dataset_FV.csv")):
            self.textboxFV_DATA.configure(fg_color="red")
            self.buttonFV.configure(state=ctk.DISABLED)
            allOk = False
        else:
            self.textboxFV_DATA.configure(fg_color="green")
            self.buttonFV.configure(state=ctk.NORMAL)

        if not allOk:
            #textbox background red
            self.textboxDATA_OUT.configure(fg_color="red")
        else:
            #textbox background green
            self.textboxDATA_OUT.configure(fg_color="green")

        model_path = os.path.join("experiments", self.cfg["GENERAL"]["folder"], "1.DL_Training", "Model")

        if self.cfg["GENERAL"]["type"] == "single":
            totalModels = len(self.cfg["TrainTest"]["epoch"]) * len(self.cfg["TrainTest"]["learning_rate"]) * len(self.cfg["TrainTest"]["batch_size"])
            availableModels = 0
            for epochs, lr, batch in product(self.cfg["TrainTest"]["epoch"], self.cfg["TrainTest"]["learning_rate"], self.cfg["TrainTest"]["batch_size"]):
                model_file = f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{epochs}_lr_{lr:.7f}_model_{self.cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{batch}_exclude_-1.pl'
                if os.path.exists(os.path.join(model_path, model_file)):
                    availableModels += 1
        elif self.cfg["GENERAL"]["type"] == "leaveoneout":
            totalModels = len(self.cfg["TrainTest"]["epoch"]) * len(self.cfg["TrainTest"]["learning_rate"]) * len(self.cfg["TrainTest"]["batch_size"]) * len(self.cfg["TrainTest"]["leaveoneoutspecies"])
            availableModels = 0
            for epochs, lr, batch, lOneOut in product(self.cfg["TrainTest"]["epoch"], self.cfg["TrainTest"]["learning_rate"], self.cfg["TrainTest"]["batch_size"], range(len(self.cfg["TrainTest"]["leaveoneoutspecies"]))):
                model_file = f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{epochs}_lr_{lr:.7f}_model_{self.cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{batch}_exclude_{lOneOut}.pl'
                if os.path.exists(os.path.join(model_path, model_file)):
                    availableModels += 1

        if availableModels == totalModels:
            #textbox background red
            self.textboxTT_MOD.configure(fg_color="green")
            self.buttonFT.configure(state=ctk.NORMAL)
        elif availableModels == 0:
            #textbox background red
            self.textboxTT_MOD.configure(fg_color="red")
            self.buttonFT.configure(state=ctk.DISABLED)
        else:
            #textbox background orange
            self.textboxTT_MOD.configure(fg_color="orange")
            self.buttonFT.configure(state=ctk.NORMAL)

        #Checking output FT models
        model_names = []
        model_strings = []
        if self.cfg["GENERAL"]["type"] == "single":
            inputModelsCount = len(self.cfg["TrainTest"]["epoch"]) * len(self.cfg["TrainTest"]["learning_rate"]) * len(self.cfg["TrainTest"]["batch_size"])
            totalModels = len(self.cfg["FineTuning"]["epoch"]) * len(self.cfg["FineTuning"]["learning_rate"]) * len(self.cfg["FineTuning"]["batch_size"]) * inputModelsCount
            # Input model names
            for EPOCHS, LR, BATCH in product(self.cfg["TrainTest"]["epoch"], self.cfg["TrainTest"]["learning_rate"],
                                             self.cfg["TrainTest"]["batch_size"]):
                model_names.append(f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}_exclude_-1.pl')
                model_strings.append(f'{self.cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{self.cfg["TrainTest"]["big_or_small_model"] + 1}-{BATCH}')

            availableModels = 0
            for EPOCHS, LR, BATCH, MODEL in product(self.cfg["FineTuning"]["epoch"],
                                                    self.cfg["FineTuning"]["learning_rate"],
                                                    self.cfg["FineTuning"]["batch_size"], range(len(model_names))):
                model_file = f'M_FT_origin_{model_strings[MODEL]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}_exclude_-1.pl'
                if os.path.exists(os.path.join(model_path, model_file)):
                    availableModels += 1

        elif self.cfg["GENERAL"]["type"] == "leaveoneout":
            totalModels = len(self.cfg["FineTuning"]["epoch"]) * len(self.cfg["FineTuning"]["learning_rate"]) * len(self.cfg["FineTuning"]["batch_size"]) * len(self.cfg["TrainTest"]["leaveoneoutspecies"])
            # Input model names
            for EPOCHS, LR, BATCH, L1O in product(self.cfg["TrainTest"]["epoch"], self.cfg["TrainTest"]["learning_rate"],
                                                  self.cfg["TrainTest"]["batch_size"],
                                                  range(len(self.cfg["TrainTest"]["leaveoneoutspecies"]))):
                model_names.append(f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}_exclude_{L1O}.pl')
                model_strings.append(f'{self.cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{self.cfg["TrainTest"]["big_or_small_model"] + 1}-{BATCH}-{L1O}')

            availableModels = 0
            for EPOCHS, LR, BATCH, MODEL in product(self.cfg["FineTuning"]["epoch"],
                                                    self.cfg["FineTuning"]["learning_rate"],
                                                    self.cfg["FineTuning"]["batch_size"], range(len(model_names))):
                for idx, speciesGroup in enumerate(self.cfg["TrainTest"]["leaveoneoutspecies"]):
                    model_file = f'M_FT_origin_{model_strings[MODEL]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}_exclude_{idx}.pl'
                    if os.path.exists(os.path.join(model_path, model_file)):
                        availableModels += 1

        if availableModels == totalModels:
            #textbox background red
            self.textboxFT_MOD.configure(fg_color="green")
            self.buttonFV.configure(state=ctk.NORMAL)
        elif availableModels == 0:
            #textbox background red
            self.textboxFT_MOD.configure(fg_color="red")
            self.buttonFV.configure(state=ctk.DISABLED)
        else:
            #textbox background orange
            self.textboxFT_MOD.configure(fg_color="orange")
            self.buttonFV.configure(state=ctk.NORMAL)

    '''
    This method reads the configuration file and returns a dictionary with 
    the configuration parameters.
    '''
    def read_config(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        return config

    '''
    These methods are used to run different stages of the pipeline. They create instances of 
    the myProcess class and start threads for each stage.  
    '''
    def run_full_pipeline(self):
        #I have to check if the datasets are already present
        pre_input_prc_folder_out = os.path.join("experiments/" + self.cfg["GENERAL"]["folder"], "0.DataSet")
        if not os.path.exists(pre_input_prc_folder_out):
            os.makedirs(pre_input_prc_folder_out)

        self.cfg["tmp"] = {}
        self.cfg["tmp"]["createTT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_TT.csv"), self.cfg["TrainTest"]["createflag"]) if "TrainTest" in self.cfg else False
        self.cfg["tmp"]["createFT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FT.csv"), self.cfg["FineTuning"]["createflag"]) if "FineTuning" in self.cfg else False
        self.cfg["tmp"]["createFV"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FV.csv"), self.cfg["Validation"]["createflag"])  if "Validation" in self.cfg else True

        #get the color of self.textboxFT_MOD
        if self.cfg["TrainTest"]["trainflag"] == "ask" and self.textboxFT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["TrainTest"]["trainflag"] = "yes"
            else:
                self.cfg["TrainTest"]["trainflag"] = "no"

        #get the color of self.textboxFT_MOD
        if self.cfg["FineTuning"]["trainflag"] == "ask" and self.textboxFT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output Fine Tuned models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["FineTuning"]["trainflag"] = "yes"
            else:
                self.cfg["FineTuning"]["trainflag"] = "no"

        #If it exists at least one file whose name starts with inference in the inferences folder...
        if len([f for f in os.listdir(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "inferences")) if f.startswith("inference")]) > 0:
            answer = messagebox.askyesno("Warning", f"Some of the output inferences seem to be already present. Do you want to recompute them?", icon='warning')
            if answer == True:
                self.cfg["Validation"]["inferflag"] = "yes"
            else:
                self.cfg["Validation"]["inferflag"] = "no"

        semaphore = threading.Semaphore(1)

        my_class = myProcess("Dataset Filtering", self.cfg)
        thread1 = my_class.start_thread(thread_jj2, semaphore)
        #thread1.join()  # Wait for the thread to finish


        # Now I want to launch another process, when the previous one is finished
        my_class = myProcess("Training and Test", self.cfg)
        thread2 = my_class.start_thread(thread_jj3, semaphore)
        #thread2.join()  # Wait for the thread to finish

        # Now I want to launch another process, when the previous one is finished
        my_class = myProcess("Fine Tuning", self.cfg)
        thread3 = my_class.start_thread(thread_jj3ft, semaphore)
        #thread3.join()  # Wait for the thread to finish

        # Now I want to launch another process, when the previous one is finished
        my_class = myProcess("Inference", self.cfg)
        thread4 = my_class.start_thread(thread_jj4, semaphore)
        #thread4.join()  # Wait for the thread to finish

        self.checkIO()

    def run_pipeline_stage_jj1(self):

        #check if the file
        if self.cfg["EMBEDDINGS"]["createflag"] == "ask" and os.path.exists(os.path.join(self.cfg["EMBEDDINGS"]["uniprotfolder"], self.cfg["EMBEDDINGS"]["outputdatasetname"] + "_embeddings_dataset.csv")):
            answer = messagebox.askyesno("Warning", f"The final dataset {self.cfg['EMBEDDINGS']['outputdatasetname'] + '_embeddings_dataset.csv'} appears to be already there. Do you want to regenerate it?", icon='warning')
            if answer == True:
                # check if files are already present in the uniprotfolder folder /embeddings
                if not os.path.exists(os.path.join(self.cfg["EMBEDDINGS"]["uniprotfolder"], "embeddings")):
                    os.makedirs(os.path.join(self.cfg["EMBEDDINGS"]["uniprotfolder"], "embeddings"))

                # check if any file whose name ends with _embeddings.csv is present in the embeddings folder
                if self.cfg["EMBEDDINGS"]["createflag"] == "ask" and (
                        len([f for f in os.listdir(os.path.join(self.cfg["EMBEDDINGS"]["uniprotfolder"], "embeddings"))
                             if f.endswith("_embeddings.csv")]) > 0):
                    answer = messagebox.askyesno("Warning",
                                                 f"Some of the embeddings output files are already present. Do you want to recompute them?",
                                                 icon='warning')
                    if answer == True:
                        self.cfg["EMBEDDINGS"]["createflag"] = "yes"
                    else:
                        self.cfg["EMBEDDINGS"]["createflag"] = "no"

                if self.cfg["EMBEDDINGS"]["addlabelsflag"] == "ask":
                    answer = messagebox.askyesno("Warning", f"Do you want to add LABELS (0/1) to embedding files?",
                                                 icon='warning')
                    if answer == True:
                        self.cfg["EMBEDDINGS"]["addlabelsflag"] = "yes"
                    else:
                        self.cfg["EMBEDDINGS"]["addlabelsflag"] = "no"

                my_class = myProcess("Embeddings computation", self.cfg, self.checkIO)
                my_class.start_thread(thread_jj1)


    def run_pipeline_stage_jj2(self):

        #I have to check if the datasets are already present
        pre_input_prc_folder_out = os.path.join("experiments/" + self.cfg["GENERAL"]["folder"], "0.DataSet")
        if not os.path.exists(pre_input_prc_folder_out):
            os.makedirs(pre_input_prc_folder_out)

        self.cfg["tmp"] = {}
        self.cfg["tmp"]["createTT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_TT.csv"), self.cfg["TrainTest"]["createflag"]) if "TrainTest" in self.cfg else False
        self.cfg["tmp"]["createFT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FT.csv"), self.cfg["FineTuning"]["createflag"]) if "FineTuning" in self.cfg else False
        self.cfg["tmp"]["createFV"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FV.csv"), self.cfg["Validation"]["createflag"])  if "Validation" in self.cfg else True

        my_class = myProcess("Dataset Filtering", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj2)

    def run_pipeline_stage_jj3(self):

        #get the color of self.textboxFT_MOD
        if self.cfg["TrainTest"]["trainflag"] == "ask" and self.textboxFT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["TrainTest"]["trainflag"] = "yes"
            else:
                self.cfg["TrainTest"]["trainflag"] = "no"

        my_class = myProcess("Training and Test", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj3)

    def run_pipeline_stage_jj3ft(self):

        #get the color of self.textboxFT_MOD
        if self.cfg["FineTuning"]["trainflag"] == "ask" and self.textboxFT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output Fine Tuned models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["FineTuning"]["trainflag"] = "yes"
            else:
                self.cfg["FineTuning"]["trainflag"] = "no"

        my_class = myProcess("Fine Tuning", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj3ft)

    def run_pipeline_stage_jj4(self):

        if "Validation" not in self.cfg:
            self.cfg["Validation"] = {"trainflag": "yes"}

        #If it exists at least one file whose name starts with inference in the inferences folder...
        if len([f for f in os.listdir(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "inferences")) if f.startswith("inference")]) > 0:
            answer = messagebox.askyesno("Warning", f"Some of the output inferences seem to be already present. Do you want to recompute them?", icon='warning')
            if answer == True:
                self.cfg["Validation"]["inferflag"] = "yes"
            else:
                self.cfg["Validation"]["inferflag"] = "no"

        my_class = myProcess("Inference", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj4)

    def run_pipeline_stage(self):
        pass

    '''
    This method starts the main loop of the application.
    '''
    def run(self):
        self.window.mainloop()

    '''
    This function is called when the user clicks the "Settings" button
    It opens a new window with input fields for each configuration option
    The user can modify the values and save the changes
    '''
    def open_cfg_window(self):
        # Create a new window
        self.cfg_window = ctk.CTk()
        self.cfg_window.geometry("500x500")

        # Create a scrolled frame
        scrolled_frame = ctk.CTkScrollableFrame(self.cfg_window)
        scrolled_frame.pack(fill=ctk.BOTH, expand=True)

        # Create input fields for each configuration option
        self.cfg_entries = {}
        tmpCfg = self.read_config(self.cfg_filename)

        row = 0
        for section in tmpCfg.sections():
            for key in tmpCfg[section]:
                # Create an entry for the configuration option
                label = ctk.CTkLabel(scrolled_frame, text=f"{section} - {key}", anchor='w')
                label.grid(row=row, column=0, sticky='w')
                entry = ctk.CTkEntry(scrolled_frame)
                entry.insert(0, tmpCfg[section][key])
                entry.grid(row=row, column=1)

                # Add the entry to the dictionary of entries
                self.cfg_entries[(section, key)] = entry

                # Create a tooltip for the entry
                tooltip = CTkToolTip(entry, message = tooltips.get(key, 'No explanation available'))
                row += 1

        # Create a frame for the buttons
        button_frame = ctk.CTkFrame(self.cfg_window, height=50, fg_color="white")
        button_frame.pack(expand=False, pady=0)

        # Create a save button
        save_button = ctk.CTkButton(button_frame, text="Save", command=lambda: self.save_cfg(tmpCfg))
        save_button.pack(side=ctk.LEFT, pady=5, padx=(5,2), anchor='center')

        # Create a "Save As" button
        save_as_button = ctk.CTkButton(button_frame, text="Save As", command=self.save_cfg_as)
        save_as_button.pack(side=ctk.LEFT, pady=5, padx=(2,5), anchor='center')

        self.cfg_window.mainloop()

    '''
    This method is called when the user clicks on the "Save" button in the configuration window.
    '''
    def save_cfg(self, cfg):
        # Update the configuration with the new values from the input fields
        for (section, key), entry in self.cfg_entries.items():
            cfg[section][key] = entry.get()

        # Save the configuration to the file
        config = configparser.ConfigParser()
        config.read_dict(cfg)
        with open(self.cfg_filename, 'w') as f:
            config.write(f)

        self.cfg = self.read_configuration(self.cfg_filename)
        self.checkIO()

        # Close the configuration window
        self.cfg_window.destroy()

    '''
    This method is called when the user clicks on the "Save As" button in the configuration window.
    It opens a file dialog to select the new filename and saves the configuration to the new file.
    '''
    def save_cfg_as(self):
        # Open a file dialog to select the new filename
        new_filename = ctk.filedialog.asksaveasfilename(defaultextension=".ini",
                                                        filetypes=(("INI files", "*.ini"), ("All files", "*.*")))

        # If a filename was selected
        if new_filename:
            # Update the configuration with the new values from the input fields
            config = configparser.ConfigParser()
            for (section, key), entry in self.cfg_entries.items():
                if not config.has_section(section):
                    config.add_section(section)
                config.set(section, key, entry.get())

            # Save the configuration to the new file
            with open(new_filename, 'w') as f:
                config.write(f)

    '''
    This method is called when the user clicks on the "Open" button.
    It opens a Finder window with the specified folder path.
    '''
    def open_finder(self, folder_path):
        webbrowser.open(f"file:///{os.path.abspath(folder_path)}")

