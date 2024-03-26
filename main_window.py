import json
import os.path
import threading
from tkinter import messagebox
from tooltips import tooltips
import customtkinter as ctk
from PIL import Image
import configparser
from CTkToolTip import CTkToolTip
import webbrowser
from process import myProcess
from threads import *
from config_window import *

class MainWindow:
    '''
    This is the constructor method where the main window is initialized, and several UI components are created and packed into the window. It also reads the configuration file and sets up the initial state of the application.
    '''
    def __init__(self):
        ctk.set_default_color_theme("green")
        self.window = ctk.CTk()
        self.window.focus_force()  # Add this line to force focus on the window
        self.cfg_filename = ctk.filedialog.askopenfilename(title="Select configuration file", initialdir=os.path.join("experiments", "_configurations"),filetypes=(("INI files", "*.ini"), ("All files", "*.*")))

        if not self.cfg_filename:
            self.cfg = {}
            raise Exception("No configuration file selected - TBD: handle this case properly!")
        else:
            self.cfg = read_configuration(self)

        self.window.geometry("1400x700")
        self.window.title("RBP ToolKit")

        self.my_font = ctk.CTkFont(family="Helvetica", size = 16, weight = "bold")
        self.labelTitle = ctk.CTkLabel(self.window, text=f"{self.cfg['GENERAL']['name']}\n{self.cfg['GENERAL']['type'].capitalize()} Experiment", font=self.my_font)
        self.labelTitle.pack(pady=(5, 10))

        button_frame = ctk.CTkFrame(self.window)
        button_frame.pack()

        textBoxHeight = 4

        # Add the buttons to the new frame
        button_image = ctk.CTkImage(Image.open("assets/settings.icns"), size=(35, 35))
        self.buttonCfg = ctk.CTkButton(button_frame, text=f"Configure\n {self.cfg['GENERAL']['acronym']}", fg_color="lightgrey", text_color="black", image=button_image, command=self.open_cfg_window)
        self.buttonCfg.pack(side=ctk.LEFT)

        if "UNIPROT" in self.cfg or "EMBEDDINGS" in self.cfg:
            button_image = ctk.CTkImage(Image.open("assets/execute.icns"), size=(35, 35))
            self.buttonRun = ctk.CTkButton(button_frame, text="Run Data Pipeline", fg_color="white", text_color="black", image=button_image, command=self.run_full_uniprot_pipeline)
            self.buttonRun.pack(side=ctk.LEFT, padx=(10,0))

        button_image = ctk.CTkImage(Image.open("assets/refresh.png"), size=(20, 20))
        self.buttonRun = ctk.CTkButton(button_frame, text="Refresh", width=100, fg_color="lightgrey", text_color="black", image=button_image, command=self.checkIO)
        self.buttonRun.pack(fill=ctk.BOTH, side=ctk.LEFT, padx=(10,0))

        if "UNIPROT" in self.cfg or "EMBEDDINGS" in self.cfg:
            frameDATA = ctk.CTkFrame(self.window, )
            if "UNIPROT" in self.cfg:
                pipeLabel = ctk.CTkLabel(frameDATA, text="Data Download and Embeddings Pipeline", anchor='nw', font=self.my_font)
                pipeLabel.pack(padx=(0, 0), pady=(5,0))
                self.buttonUPW = ctk.CTkButton(frameDATA, text="1. Download from UniProt", width=60, image=ctk.CTkImage(Image.open("assets/uniprot.png"), size=(60, 60)), command=self.run_pipeline_stage_jj0)
                self.buttonUPW.pack(side=ctk.LEFT, padx=(20,0))

            self.textboxDATA_UP = ctk.CTkButton(frameDATA, height=textBoxHeight, text="Uniprot Dataset", command=lambda: self.open_finder(os.path.dirname(self.cfg["UNIPROT"]["go_folder"])))
            self.textboxDATA_UP.pack(pady=25, padx=(5,5), side=ctk.LEFT)

            if "EMBEDDINGS" in self.cfg:
                self.buttonEMB = ctk.CTkButton(frameDATA, text="2. Embeddings", image=ctk.CTkImage(Image.open("assets/embeddings.png"), size=(60, 60)), command=self.run_pipeline_stage_jj1)
                self.buttonEMB.pack(side=ctk.LEFT)
                self.textboxDATA_IN = ctk.CTkButton(frameDATA, height=textBoxHeight, text="Embeddings Dataset", command=lambda: self.open_finder(os.path.dirname(self.cfg["GENERAL"]["originaldataset"][0])))
                self.textboxDATA_IN.pack(pady=25, padx=(5,0), side=ctk.LEFT)
            frameDATA.pack(expand=True, pady=(20, 20), ipadx=5, ipady=20)

        if "TRAINTEST" in self.cfg or "FINETUNING" in self.cfg or "VALIDATION" in self.cfg:
            button_image = ctk.CTkImage(Image.open("assets/execute.icns"), size=(35, 35))
            self.buttonRun = ctk.CTkButton(self.window, text="Run Training/Inference Pipeline", fg_color="white", text_color="black", image=button_image, command=self.run_full_pipeline)
            self.buttonRun.pack(side=ctk.TOP, padx=(0,0))

            framePIPE = ctk.CTkFrame(self.window)
            pipeLabel = ctk.CTkLabel(framePIPE, text="Training-Test/Fine Tuning/Inference Pipeline", anchor='nw', font=self.my_font)
            pipeLabel.pack(padx=(0, 0), pady=(5, 0))


            if "TRAINTEST-Label_1" in self.cfg or "TRAINTEST-Label_0" in self.cfg or "FINETUNING-Label_0" in self.cfg or "FINETUNING-Label_1" in self.cfg or "VALIDATION-Label" in self.cfg:
                self.buttonDATA = ctk.CTkButton(framePIPE, text="1. Data Filters", image=ctk.CTkImage(Image.open("assets/data.png"), size=(60, 60)), command = self.run_pipeline_stage_jj2)
                self.buttonDATA.pack(side=ctk.LEFT, padx=(5, 0))

            # Create a new frame for the button and textboxes
            if "TRAINTEST" in self.cfg:
                frameDF = ctk.CTkFrame(framePIPE)
                frameDF.pack(side=ctk.LEFT, expand=True)

                self.textboxDATA_OUT = ctk.CTkButton(frameDF, height=textBoxHeight, text="TT/FT/FV Datasets", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
                self.textboxDATA_OUT.pack(pady=(25,2), padx=(2, 25), side=ctk.TOP)

                self.textboxTT_DATA = ctk.CTkButton(frameDF, height=textBoxHeight, text="TT Dataset", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
                self.textboxTT_DATA.pack(pady=(2,25), padx=(25, 2), side=ctk.TOP)


                self.buttonTT = ctk.CTkButton(framePIPE, text="2. Training/Test", image=ctk.CTkImage(Image.open("assets/TT.png"), size=(60, 60)), command=self.run_pipeline_stage_jj3)
                self.buttonTT.pack(side=ctk.LEFT)

            if "FINETUNING" in self.cfg or "TRAINTEST" in self.cfg:
                # Create a new frame for the button and textboxes
                frameFT = ctk.CTkFrame(framePIPE)
                frameFT.pack(side=ctk.LEFT, expand=True)

                if "TRAINTEST" in self.cfg:
                    # Add the textboxes to the new frame
                    self.textboxTT_MOD = ctk.CTkButton(frameFT, height=textBoxHeight, text="Model", command=lambda: self.open_finder(
                                                       os.path.join("experiments", self.cfg["GENERAL"]["folder"],"1.DL_Training", "Model")))
                    self.textboxTT_MOD.pack(pady=(25, 2), padx=(5, 5), side=ctk.TOP)

                if "FINETUNING" in self.cfg:
                    self.textboxFT_DATA = ctk.CTkButton(frameFT, height=textBoxHeight, text="FT Dataset",command=lambda: self.open_finder(
                                                          os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
                    self.textboxFT_DATA.pack(pady=(2, 25), padx=(5, 5), side=ctk.TOP)

                    self.buttonFT = ctk.CTkButton(framePIPE, text="3. Fine Tuning", image=ctk.CTkImage(Image.open("assets/FT.png"), size=(60, 60)), command=self.run_pipeline_stage_jj3ft)
                    self.buttonFT.pack(side=ctk.LEFT)

            if "VALIDATION" in self.cfg:
                # Create a new frame for the button and textboxes
                frameFV = ctk.CTkFrame(framePIPE)
                frameFV.pack(side=ctk.LEFT, expand=True)

                # Add the textboxes to the new frame
                if "FINETUNING" in self.cfg:
                    self.textboxFT_MOD = ctk.CTkButton(frameFV, height=textBoxHeight, text="Fine Tuned Model", command=lambda: self.open_finder(
                                                           os.path.join("experiments", self.cfg["GENERAL"]["folder"],"1.DL_Training", "Model")))
                else:
                    self.textboxFT_MOD = ctk.CTkButton(frameFV, height=textBoxHeight, text="Model",
                                                       command=lambda: self.open_finder(
                                                           os.path.join("experiments", self.cfg["GENERAL"]["folder"],
                                                                        "1.DL_Training", "Model")))
                self.textboxFT_MOD.pack(pady=(25, 2), padx=(5, 5), side=ctk.TOP)

                self.textboxFV_DATA = ctk.CTkButton(frameFV, height=textBoxHeight, text="FV Dataset",command=lambda: self.open_finder(
                                                      os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")))
                self.textboxFV_DATA.pack(pady=(2, 25), padx=(5, 5), side=ctk.TOP)


                self.buttonFV = ctk.CTkButton(framePIPE, text="4. Inference", image=ctk.CTkImage(Image.open("assets/FV.png"), size=(60, 60)), command=self.run_pipeline_stage_jj4)
                self.buttonFV.pack(side=ctk.LEFT, padx=(0, 5))

            frameOUT = ctk.CTkFrame(self.window)
            frameOUT.pack(side=ctk.BOTTOM, pady=0, expand=True)

            self.textboxRESULTS = ctk.CTkButton(frameOUT, height=textBoxHeight, fg_color="white", text_color="black", text = "Results", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"],"results")))
            self.textboxRESULTS.pack(pady=(5), padx=(5, 10), side=ctk.LEFT)
            self.textboxLOGS = ctk.CTkButton(frameOUT, height=textBoxHeight, fg_color="white", text_color="black", text="Logs", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"],"logs")))
            self.textboxLOGS.pack(pady=(5), padx=(10, 10), side=ctk.LEFT)
            self.textboxVAL = ctk.CTkButton(frameOUT, height=textBoxHeight, fg_color="white", text_color="black", text="Inferences", command=lambda: self.open_finder(os.path.join("experiments", self.cfg["GENERAL"]["folder"],"inferences")))
            self.textboxVAL.pack(pady=(5), padx=(10, 20), side=ctk.LEFT)
            framePIPE.pack(expand=True, pady=(10, 0), ipadx=5, ipady=20)

        self.checkIO()


    '''
    This method checks the existence of certain files and directories and updates 
    the UI components accordingly.    
    '''
    def checkIO(self):

        self.labelTitle.configure(text=f"{self.cfg['GENERAL']['name']}\n{self.cfg['GENERAL']['type'].capitalize()} Experiment", font=self.my_font)

        #If there are files in the cfg["UNIPROT"]["go_folder"], the textbox will be green
        if "UNIPROT" in self.cfg or "EMBEDDINGS" in self.cfg:
            if "UNIPROT" in self.cfg and os.path.exists(os.path.join(self.cfg["UNIPROT"]["go_folder"], "downloads", self.cfg["UNIPROT"]["datasetname"] + ".dataset.csv")):
                #textbox background green
                self.textboxDATA_UP.configure(fg_color="green")
                self.buttonEMB.configure(state=ctk.NORMAL)
            else:
                #textbox background red
                self.textboxDATA_UP.configure(fg_color="red")
                self.buttonEMB.configure(state=ctk.DISABLED)

        allOk = True
        count = 0
        for dsFile in self.cfg["GENERAL"]["originaldataset"]:
            if not os.path.exists(dsFile):
                allOk = False
                count += 1

        if "EMBEDDINGS" in self.cfg:
            if not allOk:
                if count < len(self.cfg["GENERAL"]["originaldataset"]):
                    self.textboxDATA_IN.configure(fg_color="orange")
                else:
                    self.textboxDATA_IN.configure(fg_color="red")
            else:
                #textbox background green
                self.textboxDATA_IN.configure(fg_color="green")

        if "TRAINTEST-Label_1" in self.cfg or "TRAINTEST-Label_0" in self.cfg or "FINETUNING-Label_0" in self.cfg or "FINETUNING-Label_1" in self.cfg or "VALIDATION-Label" in self.cfg:
            if not allOk:
                if count < len(self.cfg["GENERAL"]["originaldataset"]):
                    self.buttonDATA.configure(state=ctk.NORMAL)
                else:
                    self.buttonDATA.configure(state=ctk.DISABLED)
            else:
                #textbox background green
                self.buttonDATA.configure(state=ctk.NORMAL)


        allOk = True
        datasetFolder = os.path.join("experiments", self.cfg["GENERAL"]["folder"], "0.DataSet")
        if not os.path.exists(datasetFolder):
            os.makedirs(datasetFolder)

        if "TRAINTEST" in self.cfg:
            if not os.path.exists(os.path.join(datasetFolder, "dataset_TT.csv")):
                self.textboxTT_DATA.configure(fg_color="red")
                self.buttonTT.configure(state=ctk.DISABLED)
                allOk = False
            else:
                self.textboxTT_DATA.configure(fg_color="green")
                self.buttonTT.configure(state=ctk.NORMAL)

        if "FINETUNING" in self.cfg:
            if not os.path.exists(os.path.join(datasetFolder, "dataset_FT.csv")):
                self.textboxFT_DATA.configure(fg_color="red")
                self.buttonFT.configure(state=ctk.DISABLED)
                allOk = False
            else:
                self.textboxFT_DATA.configure(fg_color="green")
                self.buttonFT.configure(state=ctk.NORMAL)

        if "VALIDATION" in self.cfg:
            if not os.path.exists(os.path.join(datasetFolder, "dataset_FV.csv")):
                self.textboxFV_DATA.configure(fg_color="red")
                self.buttonFV.configure(state=ctk.DISABLED)
                allOk = False
            else:
                self.textboxFV_DATA.configure(fg_color="green")
                self.buttonFV.configure(state=ctk.NORMAL)

        if "TRAINTEST" in self.cfg:
            if not allOk:
                #textbox background red
                self.textboxDATA_OUT.configure(fg_color="red")
            else:
                #textbox background green
                self.textboxDATA_OUT.configure(fg_color="green")

        model_path = os.path.join("experiments", self.cfg["GENERAL"]["folder"], "1.DL_Training", "Model")

        if "TRAINTEST" in self.cfg:
            if self.cfg["GENERAL"]["type"] == "single":
                totalModels = len(self.cfg["TRAINTEST"]["epoch"]) * len(self.cfg["TRAINTEST"]["learning_rate"]) * len(self.cfg["TRAINTEST"]["batch_size"])
                availableModels = 0
                for epochs, lr, batch in product(self.cfg["TRAINTEST"]["epoch"], self.cfg["TRAINTEST"]["learning_rate"], self.cfg["TRAINTEST"]["batch_size"]):
                    model_file = f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{epochs}_lr_{lr:.7f}_model_{self.cfg["TRAINTEST"]["model_name"]}_batch_{batch}_exclude_-1.pl'
                    if os.path.exists(os.path.join(model_path, model_file)):
                        availableModels += 1
            elif self.cfg["GENERAL"]["type"] == "leaveoneout":
                totalModels = len(self.cfg["TRAINTEST"]["epoch"]) * len(self.cfg["TRAINTEST"]["learning_rate"]) * len(self.cfg["TRAINTEST"]["batch_size"]) * len(self.cfg["TRAINTEST"]["leaveoneoutspecies"])
                availableModels = 0
                for epochs, lr, batch, lOneOut in product(self.cfg["TRAINTEST"]["epoch"], self.cfg["TRAINTEST"]["learning_rate"], self.cfg["TRAINTEST"]["batch_size"], range(len(self.cfg["TRAINTEST"]["leaveoneoutspecies"]))):
                    model_file = f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{epochs}_lr_{lr:.7f}_model_{self.cfg["TRAINTEST"]["model_name"]}_batch_{batch}_exclude_{lOneOut}.pl'
                    if os.path.exists(os.path.join(model_path, model_file)):
                        availableModels += 1

            if "FINETUNING" in self.cfg:
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

        if "FINETUNING" in self.cfg and "TRAINTEST" in self.cfg:
            #Checking output FT models
            model_names = []
            model_strings = []
            if self.cfg["GENERAL"]["type"] == "single":
                inputModelsCount = len(self.cfg["TRAINTEST"]["epoch"]) * len(self.cfg["TRAINTEST"]["learning_rate"]) * len(self.cfg["TRAINTEST"]["batch_size"])
                totalModels = len(self.cfg["FINETUNING"]["epoch"]) * len(self.cfg["FINETUNING"]["learning_rate"]) * len(self.cfg["FINETUNING"]["batch_size"]) * inputModelsCount
                # Input model names
                for EPOCHS, LR, BATCH in product(self.cfg["TRAINTEST"]["epoch"], self.cfg["TRAINTEST"]["learning_rate"],
                                                 self.cfg["TRAINTEST"]["batch_size"]):
                    model_names.append(f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}_exclude_-1.pl')
                    model_strings.append(f'{self.cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{self.cfg["TRAINTEST"]["model_name"]}-{BATCH}')

                availableModels = 0
                for EPOCHS, LR, BATCH, MODEL in product(self.cfg["FINETUNING"]["epoch"],
                                                        self.cfg["FINETUNING"]["learning_rate"],
                                                        self.cfg["FINETUNING"]["batch_size"], range(len(model_names))):
                    model_file = f'M_FT_origin_{model_strings[MODEL]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}_exclude_-1.pl'
                    if os.path.exists(os.path.join(model_path, model_file)):
                        availableModels += 1

            elif self.cfg["GENERAL"]["type"] == "leaveoneout":
                totalModels = len(self.cfg["FINETUNING"]["epoch"]) * len(self.cfg["FINETUNING"]["learning_rate"]) * len(self.cfg["FINETUNING"]["batch_size"]) * len(self.cfg["TRAINTEST"]["leaveoneoutspecies"])
                # Input model names
                for EPOCHS, LR, BATCH, L1O in product(self.cfg["TRAINTEST"]["epoch"], self.cfg["TRAINTEST"]["learning_rate"],
                                                      self.cfg["TRAINTEST"]["batch_size"],
                                                      range(len(self.cfg["TRAINTEST"]["leaveoneoutspecies"]))):
                    model_names.append(f'M_{self.cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}_exclude_{L1O}.pl')
                    model_strings.append(f'{self.cfg["GENERAL"]["acronym"]}-{EPOCHS}-{LR:.7f}-{self.cfg["TRAINTEST"]["model_name"]}-{BATCH}-exclude-{L1O}')

                availableModels = 0
                for EPOCHS, LR, BATCH, MODEL in product(self.cfg["FINETUNING"]["epoch"],
                                                        self.cfg["FINETUNING"]["learning_rate"],
                                                        self.cfg["FINETUNING"]["batch_size"], range(len(model_names))):
                    for idx, speciesGroup in enumerate(self.cfg["TRAINTEST"]["leaveoneoutspecies"]):
                        model_file = f'M_FT_origin_{model_strings[MODEL]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{self.cfg["TRAINTEST"]["model_name"]}_batch_{BATCH}_exclude_{idx}.pl'
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
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(filename)
        return config

    def run_full_uniprot_pipeline(self):

        #check if the file
        if self.cfg["UNIPROT"]["createflag"] == "ask" and os.path.exists(os.path.join(self.cfg["UNIPROT"]["go_folder"], "downloads", self.cfg["UNIPROT"]["datasetname"] + ".dataset.csv")):
            answer = messagebox.askyesno("Warning", f"The final dataset {self.cfg['UNIPROT']['datasetname'] + '_dataset.csv'} appears to be already there. Do you want to download it again?", icon='warning')
            if answer == True:
                self.cfg["UNIPROT"]["createflag"] = "yes"
            else:
                self.cfg["UNIPROT"]["createflag"] = "no"

        #check if the file
        if self.cfg["EMBEDDINGS"]["createflag"] == "ask" and os.path.exists(os.path.join(self.cfg["UNIPROT"]["go_folder"], self.cfg["UNIPROT"]["datasetname"] + ".embeddings.dataset.csv")):
            answer = messagebox.askyesno("Warning", f"The final dataset {self.cfg['UNIPROT']['datasetname'] + '.embeddings.dataset.csv'} appears to be already there. Do you want to regenerate it?", icon='warning')
            if answer == True:
                # check if files are already present in the uniprotfolder folder /embeddings
                if not os.path.exists(os.path.join(self.cfg["UNIPROT"]["go_folder"], "embeddings")):
                    os.makedirs(os.path.join(self.cfg["UNIPROT"]["go_folder"], "embeddings"))
                self.cfg["EMBEDDINGS"]["createflag"] = "yes"
            else:
                self.cfg["EMBEDDINGS"]["createflag"] = "no"


        semaphore = threading.Semaphore(1)

        if self.cfg["UNIPROT"]["createflag"] != "no":
            my_class = myProcess("Uniprot download", self.cfg, self.checkIO)
            my_class.start_thread(thread_jj0, semaphore)

        if self.cfg["EMBEDDINGS"]["createflag"] != "no":
            my_class = myProcess("Embeddings computation", self.cfg, self.checkIO)
            my_class.start_thread(thread_jj1, semaphore)


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
        self.cfg["tmp"]["createTT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_TT.csv"), self.cfg["TRAINTEST"]["createflag"]) if "TRAINTEST" in self.cfg else False
        self.cfg["tmp"]["createFT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FT.csv"), self.cfg["FINETUNING"]["createflag"]) if "FINETUNING" in self.cfg else False
        self.cfg["tmp"]["createFV"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FV.csv"), self.cfg["VALIDATION"]["createflag"])  if "VALIDATION" in self.cfg else True

        #get the color of self.textboxFT_MOD
        if "TRAINTEST" in self.cfg and self.cfg["TRAINTEST"]["trainflag"] == "ask" and self.textboxTT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["TRAINTEST"]["trainflag"] = "yes"
            else:
                self.cfg["TRAINTEST"]["trainflag"] = "no"

        #get the color of self.textboxFT_MOD
        if "FINETUNING" in self.cfg and self.cfg["FINETUNING"]["trainflag"] == "ask" and self.textboxFT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output Fine Tuned models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["FINETUNING"]["trainflag"] = "yes"
            else:
                self.cfg["FINETUNING"]["trainflag"] = "no"

        #If it exists at least one file whose name starts with inference in the inferences folder...
        #If the inference folder does not exists, create it
        if not os.path.exists(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "inferences")):
            os.makedirs(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "inferences"))

        if len([f for f in os.listdir(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "inferences")) if f.startswith("inference")]) > 0:
            answer = messagebox.askyesno("Warning", f"Some of the output inferences seem to be already present. Do you want to recompute them?", icon='warning')
            if answer == True:
                self.cfg["VALIDATION"]["inferflag"] = "yes"
            else:
                self.cfg["VALIDATION"]["inferflag"] = "no"

        semaphore = threading.Semaphore(1)

        my_class = myProcess("Dataset Filtering", self.cfg)
        thread1 = my_class.start_thread(thread_jj2, semaphore)
        #thread1.join()  # Wait for the thread to finish


        # Now I want to launch another process, when the previous one is finished
        my_class = myProcess("Training and Test", self.cfg)
        thread2 = my_class.start_thread(thread_jj3, semaphore)
        #thread2.join()  # Wait for the thread to finish

        if "FINETUNING" in self.cfg:
            # Now I want to launch another process, when the previous one is finished
            my_class = myProcess("Fine Tuning", self.cfg)
            thread3 = my_class.start_thread(thread_jj3ft, semaphore)
            #thread3.join()  # Wait for the thread to finish

        # Now I want to launch another process, when the previous one is finished
        my_class = myProcess("Inference", self.cfg)
        thread4 = my_class.start_thread(thread_jj4, semaphore)
        #thread4.join()  # Wait for the thread to finish

        self.checkIO()

    def run_pipeline_stage_jj0(self):

        #check if the file
        if self.cfg["UNIPROT"]["createflag"] == "ask" and os.path.exists(os.path.join(self.cfg["UNIPROT"]["go_folder"], "downloads", self.cfg["UNIPROT"]["datasetname"] + ".dataset.csv")):
            answer = messagebox.askyesno("Warning", f"The final dataset {self.cfg['UNIPROT']['datasetname'] + '_dataset.csv'} appears to be already there. Do you want to download it again?", icon='warning')
            if answer == True:
                self.cfg["UNIPROT"]["createflag"] = "yes"
            else:
                self.cfg["UNIPROT"]["createflag"] = "no"

        my_class = myProcess("Uniprot Download", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj0)

    def run_pipeline_stage_jj1(self):

        #check if the file
        if self.cfg["EMBEDDINGS"]["createflag"] == "ask" and os.path.exists(os.path.join(self.cfg["UNIPROT"]["go_folder"], self.cfg["UNIPROT"]["datasetname"] + ".embeddings.dataset.csv")):
            answer = messagebox.askyesno("Warning", f"The final dataset {self.cfg['UNIPROT']['datasetname'] + '.embeddings.dataset.csv'} appears to be already there. Do you want to regenerate it?", icon='warning')
            if answer == True:
                # check if files are already present in the uniprotfolder folder /embeddings
                if not os.path.exists(os.path.join(self.cfg["UNIPROT"]["go_folder"], "embeddings")):
                    os.makedirs(os.path.join(self.cfg["UNIPROT"]["go_folder"], "embeddings"))

                self.cfg["EMBEDDINGS"]["createflag"] = "yes"
            else:
                self.cfg["EMBEDDINGS"]["createflag"] = "no"

        my_class = myProcess("Embeddings computation", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj1)


    def run_pipeline_stage_jj2(self):

        #I have to check if the datasets are already present
        pre_input_prc_folder_out = os.path.join("experiments/" + self.cfg["GENERAL"]["folder"], "0.DataSet")
        if not os.path.exists(pre_input_prc_folder_out):
            os.makedirs(pre_input_prc_folder_out)

        self.cfg["tmp"] = {}
        self.cfg["tmp"]["createTT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_TT.csv"), self.cfg["TRAINTEST"]["createflag"]) if "TRAINTEST" in self.cfg else False
        self.cfg["tmp"]["createFT"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FT.csv"), self.cfg["FINETUNING"]["createflag"]) if "FINETUNING" in self.cfg else False
        self.cfg["tmp"]["createFV"] = jj2_prepareInputFiles.askForGenerateFile(os.path.join(pre_input_prc_folder_out, "dataset_FV.csv"), self.cfg["VALIDATION"]["createflag"])  if "VALIDATION" in self.cfg else True

        my_class = myProcess("Dataset Filtering", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj2)

    def run_pipeline_stage_jj3(self):

        #get the color of self.textboxTT_MOD
        if self.cfg["TRAINTEST"]["trainflag"] == "ask" and self.textboxTT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["TRAINTEST"]["trainflag"] = "yes"
            else:
                self.cfg["TRAINTEST"]["trainflag"] = "no"

        my_class = myProcess("Training and Test", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj3)

    def run_pipeline_stage_jj3ft(self):

        #get the color of self.textboxFT_MOD
        if self.cfg["FINETUNING"]["trainflag"] == "ask" and self.textboxFT_MOD.cget("fg_color") != "red":
            answer = messagebox.askyesno("Warning", f"Some of the output Fine Tuned models are already present. Do you want to retrain them?", icon='warning')
            if answer == True:
                self.cfg["FINETUNING"]["trainflag"] = "yes"
            else:
                self.cfg["FINETUNING"]["trainflag"] = "no"

        my_class = myProcess("Fine Tuning", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj3ft)

    def run_pipeline_stage_jj4(self):

        if "VALIDATION" not in self.cfg:
            self.cfg["VALIDATION"] = {"trainflag": "yes"}

        #If it exists at least one file whose name starts with inference in the inferences folder...
        if len([f for f in os.listdir(os.path.join("experiments", self.cfg["GENERAL"]["folder"], "inferences")) if f.startswith("inference")]) > 0:
            answer = messagebox.askyesno("Warning", f"Some of the output inferences seem to be already present. Do you want to recompute them?", icon='warning')
            if answer == True:
                self.cfg["VALIDATION"]["inferflag"] = "yes"
            else:
                self.cfg["VALIDATION"]["inferflag"] = "no"

        my_class = myProcess("Inference", self.cfg, self.checkIO)
        my_class.start_thread(thread_jj4)

    def run_pipeline_stage(self):
        pass

    '''
    This method starts the main loop of the application.
    '''
    def run(self):
        self.window.update()
        self.window.mainloop()

    def open_cfg_window(self):
        self.cfg_window = ConfigWindow(self, self.cfg_filename, self.my_font)


    '''
    This method is called when the user clicks on the "Open" button.
    It opens a Finder window with the specified folder path.
    '''
    def open_finder(self, folder_path):
        webbrowser.open(f"file:///{os.path.abspath(folder_path)}")

