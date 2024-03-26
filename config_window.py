import configparser
import json

import customtkinter as ctk
from CTkToolTip import CTkToolTip

from tooltips import tooltips

class ConfigWindow(ctk.CTkToplevel):
    def __init__(self, parent, cfg_filename, my_font):
        self.parent = parent
        self.cfg_filename = cfg_filename
        self.my_font = my_font

        # Create a new window
        self.cfg_window = ctk.CTk()
        self.cfg_window.geometry("550x800")

        # Create a scrolled frame
        self.scrolled_frame = ctk.CTkScrollableFrame(self.cfg_window)
        self.scrolled_frame.pack(fill=ctk.BOTH, expand=True)

        # Create input fields for each configuration option
        self.cfg_entries = {}
        self.tmpCfg = self.read_config(cfg_filename)

        row = 0
        for section in self.tmpCfg.sections():
            frame = ctk.CTkFrame(master=self.scrolled_frame, fg_color="lightblue")
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=2)
            frame.grid(row=row, column=0, columnspan = 2, pady=10, padx=10, ipady=5, sticky="nsew")
            #frame.grid(row=row, column=0, sticky='nsew')

            label = ctk.CTkLabel(frame, text=f"{section}", anchor='w', font=self.my_font)
            label.grid(row=row, column=0, columnspan = 2, pady=5, padx=5, sticky='w')

            row += 1
            for key in self.tmpCfg[section]:
                # Create a grey horizontal line
                separator = ctk.CTkLabel(frame, bg_color="lightgrey", text=key, font=self.my_font, height=1)
                separator.grid(row=row, column=0, columnspan=2, sticky='ew')
                row += 1

                # Create a tooltip for the entry
                #tooltip = CTkToolTip(entry, message = tooltips.get(key, 'No explanation available'))
                tooltip = ctk.CTkLabel(frame, wraplength=500, justify="left", text = tooltips.get(key, 'No explanation available'))
                tooltip.grid(row=row, column=0, columnspan=2, pady=(3,3), sticky='w')
                row += 1

                # Create an entry for the configuration option
                #label = ctk.CTkLabel(frame, text=f"{key}", anchor='w', font=self.my_font, justify="left", width=labelWidth)
                #label.grid(row=row, column=0, sticky='w')
                entryWidth = 150

                entry = ctk.CTkEntry(frame, width=250)
                entry.insert(0, self.tmpCfg[section][key])
                entry.grid(row=row, column=0, columnspan=2, pady=(0, 10))

                # Add the entry to the dictionary of entries
                self.cfg_entries[(section, key)] = entry

                row += 1

            # Create a frame for the buttons
        button_frame = ctk.CTkFrame(self.cfg_window, height=50, fg_color="white")
        button_frame.pack(expand=False, pady=0)

        # Create a save button
        save_button = ctk.CTkButton(button_frame, text="Save", command=lambda: self.save_cfg(self.tmpCfg))
        save_button.pack(side=ctk.LEFT, pady=5, padx=(5, 2), anchor='center')

        # Create a "Save As" button
        save_as_button = ctk.CTkButton(button_frame, text="Save As", command=self.save_cfg_as)
        save_as_button.pack(side=ctk.LEFT, pady=5, padx=(2, 5), anchor='center')

        #self.cfg_window.transient(self.parent.window)
        self.cfg_window.grab_set()
        self.parent.window.wait_window(self.cfg_window)  # wait for current window to close
        self.cfg_window.mainloop()


    def read_config(self, filename):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(filename)
        return config


    def save_cfg(self, cfg):
        # Update the configuration with the new values from the input fields
        for (section, key), entry in self.cfg_entries.items():
            cfg[section][key] = entry.get()

        # Save the configuration to the file
        config = configparser.ConfigParser()
        config.read_dict(cfg)
        with open(self.cfg_filename, 'w') as f:
            config.write(f)

        self.parent.cfg = read_configuration(self)
        self.parent.checkIO()

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
    This method is called when the user clicks on the "Conf. exp." button. 
    It opens a new window where the user can see and edit the configuration file.
    '''
def read_configuration(self):
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
