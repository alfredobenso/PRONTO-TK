import json
import os.path
import os
import sys
import threading
import time
from tkinter import messagebox
from tooltips import tooltips
import customtkinter as ctk
import logging
import queue

import pandas as pd
from PIL import Image, ImageTk
import configparser
from CTkMessagebox import CTkMessagebox
from CTkToolTip import CTkToolTip
from itertools import product
import webbrowser
import jj2_prepareInputFiles
import jj3_DLtraining
import jj4_DLinference
import logStat
import valStatsFolder
class LoggerHandler:
    def __init__(self, app, silent_mode=True):
        self.queue = queue.Queue()
        self.progress_bar = ctk.CTkProgressBar(app)  # Add progress bar
        self.progress_bar.pack(pady=20)
        self.progress_bar.set(0)
        self.filename = "logfile.log"
        self.logger = self.setup_logger(silent_mode)
        self.logText = ctk.CTkTextbox(app)
        self.logText.pack(pady=20)

    def setup_logger(self, silent_mode):
        logger = logging.getLogger('RBPTK_logger')
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(self.filename, "w")
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        if not silent_mode:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.DEBUG)
            logger.addHandler(stream_handler)

        return logger

    def closeHandlers(self):
        for handler in self.logger.handlers[:]:  # get list of all handlers
            handler.close()
            self.logger.removeHandler(handler)  # close and remove the old handler

    def log_message(self, msg=None, progress=None):
        # If a message is provided
        if msg is not None:
            # Log the message using the logger
            self.logger.debug(msg)
            # Put the message in the queue to be displayed in the GUI's text box
            self.queue.put((self.logText.insert, ('end', msg + '\n')))

        # If a progress value is provided
        if progress is not None:
            # Put the progress value in the queue to update the progress bar in the GUI
            self.queue.put((self.progress_bar.set, (progress,)))

class myProcess:
    def __init__(self, title, cfg, callback=None):
        self.cfg = cfg
        self.SILENT_MODE = cfg["GENERAL"]["silentmode"]
        self.thread = None
        self.app = ctk.CTk()
        self.app.geometry("1000x700")
        self.app.title(title)
        self.logger_handler = LoggerHandler(self.app, self.SILENT_MODE)
        #set the width of the progress bar to 80% of the width of the app geometry
        self.logger_handler.progress_bar.configure(width=int(self.app.geometry().split("x")[0]) * 0.8)
        self.logger_handler.logText.configure(width=int(self.app.geometry().split("x")[0]) * 0.8)

        self.callback = callback  # store the callback function

    def start_thread(self, func):
        self.thread = threading.Thread(target=func, args=(self.cfg, self, self.callback,))
        self.thread.start()
        self.check_queue()

    def check_queue(self):
        while not self.logger_handler.queue.empty():
            command, args = self.logger_handler.queue.get()
            command(*args)
        if self.thread.is_alive():
            self.app.after(100, self.check_queue)

class MainWindow:
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
        self.buttonRun = ctk.CTkButton(button_frame, text="Run Exp.", fg_color="white", text_color="black", image=button_image)
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
        self.buttonEMB = ctk.CTkButton(frameDATA, text="2. Embeddings", image=ctk.CTkImage(Image.open("assets/embeddings.png"), size=(60, 60)), command=self.run_pipeline_stage)
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

    def checkIO(self):
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

    def read_config(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        return config

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

    def run(self):
        self.window.mainloop()

    #This function is called when the user clicks the "Settings" button
    #It opens a new window with input fields for each configuration option
    #The user can modify the values and save the changes
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

        # Create a save button
        save_button = ctk.CTkButton(self.cfg_window, text="Save", command=lambda: self.save_cfg(tmpCfg))
        save_button.pack(pady=10)

        self.cfg_window.mainloop()

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

    def open_finder(self, folder_path):
        webbrowser.open(f"file:///{os.path.abspath(folder_path)}")

#this is an example function
def thread_function(cfg, logger):
    logger.log_message(cfg["GENERAL"]["acronym"])
    for i in range(100):
        time.sleep(0.2)
        #logger.log_message((parentWindow.progress_bar.set, ((i + 1)/100,)))
        logger.log_message(f'Progress: {(i + 1)/100}')
    logger.log_message('Simulation finished \n')

def thread_jj2(cfg, window, callback=None):
    SILENT_MODE = cfg["GENERAL"]["silentmode"]
    logFolder = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
    if not os.path.exists(logFolder):
        os.makedirs(logFolder)

    window.logger_handler.closeHandlers()
    window.logger_handler.filename = os.path.join(logFolder, "datasetFiltering.log")
    window.logger_handler.setup_logger(SILENT_MODE)
    window.logger_handler.log_message("Starting thread_jj2")
    jj2_prepareInputFiles.filterDataSet(cfg, window.logger_handler)
    window.logger_handler.log_message("Dataset done")
    # Call the callback function if it's not None
    if callback is not None:
        callback()

def thread_jj3(cfg, window, callback=None):
    SILENT_MODE = cfg["GENERAL"]["silentmode"]

    if cfg["GENERAL"]["type"] == "single":
        maxIter = len(cfg["TrainTest"]["epoch"]) * len(cfg["TrainTest"]["learning_rate"]) * len(cfg["TrainTest"]["batch_size"])
        curIter = 1
        for EPOCHS, LR, BATCH in product(cfg["TrainTest"]["epoch"], cfg["TrainTest"]["learning_rate"], cfg["TrainTest"]["batch_size"]):
            log_file_name = f'log_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}.log'
            log_file_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
            if not os.path.exists(log_file_path):
                os.makedirs(log_file_path)

            if not os.path.exists(os.path.join(log_file_path, log_file_name)) or cfg["TrainTest"]["trainflag"] == "yes":
                window.logger_handler.closeHandlers()
                # TBD: change the log file name
                window.logger_handler.filename = os.path.join(log_file_path, log_file_name)
                window.logger_handler.setup_logger(SILENT_MODE)
                window.logger_handler.log_message("*" * 80)
                window.logger_handler.log_message(f"Experiment name: {cfg['GENERAL']['name']}")
                window.logger_handler.log_message("*" * 80)

                window.logger_handler.log_message(f"\nRunning training: {EPOCHS} epochs, {LR} learning rate, {BATCH} batch size, model {cfg['TrainTest']['big_or_small_model'] + 1} - Iteration {curIter} of {maxIter}", (curIter / maxIter))
                window.logger_handler.log_message(f"Silent Mode is {'ON' if SILENT_MODE else 'OFF'}...")
                window.logger_handler.log_message(f"Using Model {cfg['TrainTest']['big_or_small_model'] + 1}...")

            model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler, _SILENT_RUN=SILENT_MODE, trainFlag=cfg["TrainTest"]["trainflag"])

            logStat.analyzeLog(os.path.join(log_file_path, log_file_name))

            if not os.path.exists(model_location):
                window.logger_handler.log_message("Model not found")
                return()

            curIter += 1
    else:
        maxIter = len(cfg["TrainTest"]["epoch"]) * len(cfg["TrainTest"]["learning_rate"] * len(cfg["TrainTest"]["batch_size"]))
        curIter = 1

        MODEL = cfg['TrainTest']['big_or_small_model']

        for EPOCHS, LR, BATCH in product(cfg["TrainTest"]["epoch"], cfg["TrainTest"]["learning_rate"], cfg["TrainTest"]["batch_size"]):

            df = pd.DataFrame(columns=['Specie', 'bacc', 'MCC', 'Final_Test_Aupr', 'Final_Auc_Roc'])

            # for each specie in _EXPERIMENT_SPECIES, use that species as Validation and all the others as Training/Test
            results = []
            for idx, speciesGroup in enumerate(cfg["TrainTest"]["leaveoneoutspecies"]):
                print(f"Processing species group {idx}")
                print(f"Validation Group: {idx}")
                print(f"Validation Species: {speciesGroup}")
                print(f"Training parms: EPOCHS={EPOCHS}, LR={LR}, MODEL={MODEL}")

                # *****
                log_file_name = f'log_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}_exclude_{idx}.log'
                log_file_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
                if not os.path.exists(log_file_path):
                    os.makedirs(log_file_path)

                if not os.path.exists(os.path.join(log_file_path, log_file_name)) or cfg["TrainTest"]["trainflag"] == "yes":
                    window.logger_handler.closeHandlers()
                    # TBD: change the log file name
                    window.logger_handler.filename = os.path.join(log_file_path, log_file_name)
                    window.logger_handler.setup_logger(SILENT_MODE)
                    window.logger_handler.log_message("*" * 80)
                    window.logger_handler.log_message(f"Experiment name: {cfg['GENERAL']['name']}")
                    window.logger_handler.log_message("*" * 80)

                    window.logger_handler.log_message(
                        f"\nRunning L1O training: {EPOCHS} epochs, {LR} learning rate, {BATCH} batch size, model {cfg['TrainTest']['big_or_small_model'] + 1} - Iteration {curIter} of {maxIter}", (curIter / maxIter))
                    window.logger_handler.log_message(f"Silent Mode is {'ON' if SILENT_MODE else 'OFF'}...")
                    window.logger_handler.log_message(f"Using Model {cfg['TrainTest']['big_or_small_model'] + 1}...")

                model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler,
                                                         _SILENT_RUN=SILENT_MODE,
                                                         excludeSpeciesGroup = idx,
                                                         trainFlag=cfg["TrainTest"]["trainflag"])

                logStat.analyzeLog(os.path.join("experiments", cfg["GENERAL"]["folder"], "logs", log_file_name))

            curIter += 1

    window.logger_handler.log_message("\nTraining completed...")

    # Call the callback function if it's not None
    if callback is not None:
        callback()

def thread_jj3ft(cfg, window, callback=None):
    SILENT_MODE = cfg["GENERAL"]["silentmode"]

    curIter = 1
    #I collect all the names and filename string codes for all models on which I have to train
    model_names = []
    model_strings = []

    if cfg["GENERAL"]["type"] == "single":
        inputModelsCount = len(cfg["TrainTest"]["epoch"]) * len(cfg["TrainTest"]["learning_rate"]) * len(cfg["TrainTest"]["batch_size"])
        maxIter = len(cfg["FineTuning"]["epoch"]) * len(cfg["FineTuning"]["learning_rate"]) * len(cfg["FineTuning"]["batch_size"]) * inputModelsCount
        #Input model names
        for EPOCHS, LR, BATCH in product(cfg["TrainTest"]["epoch"], cfg["TrainTest"]["learning_rate"], cfg["TrainTest"]["batch_size"]):
            model_names.append(f'M_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}_exclude_-1.pl')
            model_strings.append(f'{cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{cfg["TrainTest"]["big_or_small_model"] + 1}-{BATCH}')
    elif cfg["GENERAL"]["type"] == "leaveoneout":
        inputModelsCount = len(cfg["TrainTest"]["epoch"]) * len(cfg["TrainTest"]["learning_rate"]) * len(cfg["TrainTest"]["batch_size"]) * len(cfg["TrainTest"]["leaveoneoutspecies"])
        maxIter = len(cfg["FineTuning"]["epoch"]) * len(cfg["FineTuning"]["learning_rate"]) * len(cfg["FineTuning"]["batch_size"]) * len(cfg["TrainTest"]["leaveoneoutspecies"]) * inputModelsCount
        #Input model names
        for EPOCHS, LR, BATCH, L1O in product(cfg["TrainTest"]["epoch"], cfg["TrainTest"]["learning_rate"], cfg["TrainTest"]["batch_size"], range(len(cfg["TrainTest"]["leaveoneoutspecies"]))):
            model_names.append(
                f'M_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}_exclude_{L1O}.pl')
            model_strings.append(f'{cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{cfg["TrainTest"]["big_or_small_model"] + 1}-{BATCH}-{L1O}')

    #Now I run a FineTuning on each of the models for every combination of FT epochs, learning rate and batch size
    for EPOCHS, LR, BATCH, MODEL in product(cfg["FineTuning"]["epoch"], cfg["FineTuning"]["learning_rate"], cfg["FineTuning"]["batch_size"], range(len(model_names))):
        log_file_name = f'log_FT_origin_{model_strings[MODEL]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TrainTest"]["big_or_small_model"] + 1}_batch_{BATCH}.log'
        log_file_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "logs")
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)

        if not os.path.exists(os.path.join(log_file_path, log_file_name)) or cfg["FineTuning"]["trainflag"] == "yes":
            window.logger_handler.closeHandlers()
            # TBD: change the log file name
            window.logger_handler.filename = os.path.join(log_file_path, log_file_name)
            window.logger_handler.setup_logger(SILENT_MODE)
            window.logger_handler.log_message("*" * 80)
            window.logger_handler.log_message(f"Experiment name: {cfg['GENERAL']['name']}")
            window.logger_handler.log_message("*" * 80)

            window.logger_handler.log_message(f"\nRunning fine tuning: {EPOCHS} epochs, {LR} learning rate, {BATCH} batch size, model {cfg['TrainTest']['big_or_small_model'] + 1} - Iteration {curIter} of {maxIter}", (curIter / maxIter))
            window.logger_handler.log_message(f"Silent Mode is {'ON' if SILENT_MODE else 'OFF'}...")
            window.logger_handler.log_message(f"Using Model {cfg['TrainTest']['big_or_small_model'] + 1}...")

        if cfg["GENERAL"]["type"] == "single":
            model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler, os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model", model_names[MODEL]), input_modelstring=model_strings[MODEL], _SILENT_RUN=SILENT_MODE, trainFlag=cfg["FineTuning"]["trainflag"])
        elif cfg["GENERAL"]["type"] == "leaveoneout":
            excludedGroupIndex = int(model_names[MODEL].split("_")[-1].split(".")[0].split("-")[-1])
            model_location = jj3_DLtraining.DL_train(cfg, EPOCHS, LR, BATCH, window.logger_handler, os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model", model_names[MODEL]), input_modelstring=model_strings[MODEL], _SILENT_RUN=SILENT_MODE, trainFlag=cfg["FineTuning"]["trainflag"], excludeSpeciesGroup=excludedGroupIndex)

        logStat.analyzeLog(os.path.join(log_file_path, log_file_name))

        if not os.path.exists(model_location):
            window.logger_handler.log_message("Model not found")
            return()

        curIter += 1

    window.logger_handler.log_message("\nFine Tuning completed...",1)

    # Call the callback function if it's not None
    if callback is not None:
        callback()

def thread_jj4(cfg, window, callback=None):
    SILENT_MODE = cfg["GENERAL"]["silentmode"]
    model_names = []
    if "FineTuning" not in cfg:
        #get the names of all the models in the folder 1.DL_Training/Model whose filename starts with M_<acronym> and ends with .pl
        for file in os.listdir(os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model")):
            if file.startswith(f'M_{cfg["GENERAL"]["acronym"]}') and file.endswith('.pl'):
                model_names.append(file)
    else:
        #get the names of all the models in the folder 1.DL_Training/Model whose filename starts with M_<acronym> and ends with .pl
        for file in os.listdir(os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model")):
            if file.startswith('M_FT_') and file.endswith('.pl'):
                model_names.append(file)

    if "Validation" in cfg or cfg["GENERAL"]["type"] == "leaveoneout":
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

        if os.path.exists(os.path.join(log_file_path, log_file_name)) and cfg["Validation"]["inferflag"] == "no":
            window.logger_handler.log_message(f'Inference #{idx} already present, skipping...', idx / len(model_names))
        else:
            if cfg["GENERAL"]["type"] == "single":
                window.logger_handler.log_message(f'Inference #{idx}...', idx / len(model_names))
                jj4_DLinference.DL_validate(os.path.join(model_path,model), cfg["TrainTest"]["big_or_small_model"], validationDataSet, os.path.join(log_file_path, log_file_name), window.logger_handler)
            else:
                window.logger_handler.log_message(f'Inference species: {cfg["TrainTest"]["leaveoneoutspecies"][exclude]}...', idx / len(model_names))
                jj4_DLinference.DL_validate(os.path.join(model_path,model), cfg["TrainTest"]["big_or_small_model"], validationDataSet, os.path.join(log_file_path, log_file_name), window.logger_handler, validationSpecies=cfg["TrainTest"]["leaveoneoutspecies"][exclude])

    valStatsFolder.analyseValidationFolder(window.logger_handler, os.path.join("experiments", cfg["GENERAL"]["folder"], "inferences"))

    window.logger_handler.log_message("\nInference completed...",1)

    # Call the callback function if it's not None
    if callback is not None:
        callback()

if __name__ == "__main__":
    main_window = MainWindow()
    main_window.run()
