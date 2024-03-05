import threading
import customtkinter as ctk
from logger import LoggerHandler

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

    '''
    This function starts a thread to run the given function. The function takes the configuration dictionary,
    the process object, the callback function and a semaphore as input arguments.
    The semaphore is used to control the sequential execution of the threads.
    '''
    def start_thread(self, func, semaphore=None):
        thread = threading.Thread(target=func, args=(self.cfg, self, self.callback, semaphore,))
        thread.start()
        self.thread = thread
        self.check_queue()
        return thread  # Return the thread object

    def check_queue(self):
        while not self.logger_handler.queue.empty():
            command, args = self.logger_handler.queue.get()
            command(*args)
        if self.thread.is_alive():
            self.app.after(100, self.check_queue)

