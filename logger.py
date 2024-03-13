import sys
import customtkinter as ctk
import logging
import queue

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
        logger = logging.getLogger('PRONTO_logger')
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

