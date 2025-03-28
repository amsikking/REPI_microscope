# Imports from the python standard library:
import os
import time
import tkinter as tk
from datetime import datetime
from idlelib.tooltip import Hovertip
from tkinter import filedialog
from tkinter import font

# Third party imports, installable via pip:
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tifffile import imread, imwrite

# Our code, one .py file per module, copy files to your local directory:
import repi_microscope as repi # github.com/amsikking/Repi_microscope
import tkinter_compound_widgets as tkcw # github.com/amsikking/tkinter

class GuiMicroscope:
    def __init__(self, init_microscope=True): # set False for GUI design...
        self.root = tk.Tk()
        self.root.title('REPI Microscope GUI')
        # adjust font size and delay:
        size = 10 # default = 9
        font.nametofont("TkDefaultFont").configure(size=size)
        font.nametofont("TkFixedFont").configure(size=size)
        font.nametofont("TkTextFont").configure(size=size)
        # load hardware GUI's:
        self.init_transmitted_light()
        self.init_led_box()
        self.init_dichroic_mirror()
        self.init_filter_wheel()
        self.init_camera()
        self.init_focus_piezo()
        self.init_autofocus()
        self.init_XY_stage()
        # load microscope GUI's and quit:
        self.init_grid_navigator()  # navigates an XY grid of points
        self.init_tile_navigator()  # generates and navigates XY tiles
        self.init_settings()        # collects settings from GUI
        self.init_settings_output() # shows output from settings
        self.init_position_list()   # navigates position lists
        self.init_acquire()         # microscope methods
        self.init_running_mode()    # toggles between different modes
        # optionally initialize microscope:
        if init_microscope:
            self.max_allocated_bytes = 10e9
            self.scope = repi.Microscope(
                max_allocated_bytes=self.max_allocated_bytes,
                ao_rate=1e5,
                print_warnings=False)
            self.max_bytes_per_buffer = self.scope.max_bytes_per_buffer
            # configure any hardware preferences: (place holder)
            # make mandatory call to 'apply_settings':
            self.scope.apply_settings(
                objective_name       = self.objective_name.get(),
                channels_per_image   = ('TL_LED',),
                power_per_channel    = (self.power_tl.value.get(),),
                dichroic_mirror      = self.dichroic_mirror.get(),
                emission_filter      = self.emission_filter.get(),
                illumination_time_us = self.illumination_time_us.value.get(),
                height_px            = self.height_px.value.get(),
                width_px             = self.width_px.value.get(),
                images_per_buffer    = self.images_per_buffer.value.get(),
                ).get_result() # finish
            # get XYZ direct from hardware and update gui to aviod motion:
            self.focus_piezo_z_um.update_and_validate(
                int(round(self.scope.focus_piezo_z_um)))
            self._update_XY_stage_position(
                self.scope.XY_stage_position_mm)
            # check microscope periodically:
            def _run_check_microscope():
                self.scope.apply_settings().get_result() # update attributes
                # check memory:
                self.data_bytes.set(self.scope.bytes_per_data_buffer)
                self.data_buffer_exceeded.set(self.scope.data_buffer_exceeded)
                self.total_bytes.set(self.scope.total_bytes)
                self.total_bytes_exceeded.set(self.scope.total_bytes_exceeded)
                # calculate voltages:
                self.buffer_time_s.set(self.scope.buffer_time_s)
                self.frames_per_s.set(self.scope.frames_per_s)
                # check autofocus and joystick:
                self._check_autofocus()
                self._check_joystick()
                self.root.after(int(1e3/10), _run_check_microscope) # 30fps
                return None
            _run_check_microscope()
            # make session folder:
            dt = datetime.strftime(datetime.now(),'%Y-%m-%d_%H-%M-%S_')
            self.session_folder = dt + 'repi_gui_session\\'
            os.makedirs(self.session_folder)
            # snap and enable scout mode:
            self.last_acquire_task = self.scope.acquire()
            self.running_scout_mode.set(True)
        # add close function + any commands for when the user hits the 'X'
        def _close():
            if init_microscope: self.scope.close()
            self.root.destroy()
        self.root.protocol("WM_DELETE_WINDOW", _close)        
        # start event loop:
        self.root.mainloop() # blocks here until 'X'

    def init_transmitted_light(self):
        frame = tk.LabelFrame(self.root, text='TRANSMITTED LIGHT', bd=6)
        frame.grid(row=1, column=0, padx=5, pady=5, sticky='n')
        frame_tip = Hovertip(
            frame,
            "The 'TRANSMITTED LIGHT' illuminates the sample from above.\n" +
            "NOTE: either the 'TRANSMITTED LIGHT' or at least 1 'LED'\n" +
            "must be selected.")
        self.power_tl = tkcw.CheckboxSliderSpinbox(
            frame,
            label='525/50nm (%)',
            checkbox_default=True,
            slider_length=200,
            default_value=15,
            width=5)
        self.power_tl.checkbox_value.trace_add(
            'write', self._apply_channel_settings)        
        self.power_tl.value.trace_add(
            'write', self._apply_channel_settings)
        return None

    def init_led_box(self):
        frame = tk.LabelFrame(self.root, text='LED BOX', bd=6)
        frame.grid(row=2, column=0, rowspan=6, padx=5, pady=5, sticky='n')
        frame_tip = Hovertip(
            frame,
            "The 'LED' illuminates the sample from the objective (epi).\n" +
            "NOTE: either the 'TRANSMITTED LIGHT' or at least 1\n" +
            "'LED' must be selected.")
        # 395/25:
        self.power_395 = tkcw.CheckboxSliderSpinbox(
            frame,
            label='395/25nm (%)',
            color='magenta',
            slider_length=200,
            default_value=5,
            width=5)
        self.power_395.checkbox_value.trace_add(
            'write', self._apply_channel_settings)
        self.power_395.value.trace_add(
            'write', self._apply_channel_settings)
        # 440/20:
        self.power_440 = tkcw.CheckboxSliderSpinbox(
            frame,
            label='440/20nm (%)',
            color='magenta',
            slider_length=200,
            default_value=5,
            row=1,
            width=5)
        self.power_440.checkbox_value.trace_add(
            'write', self._apply_channel_settings)        
        self.power_440.value.trace_add(
            'write', self._apply_channel_settings)
        for child in self.power_440.winfo_children():
            child.configure(state='disable')
        # 470/24:
        self.power_470 = tkcw.CheckboxSliderSpinbox(
            frame,
            label='470/24nm (%)',
            color='blue',
            slider_length=200,
            default_value=5,
            row=2,
            width=5)
        self.power_470.checkbox_value.trace_add(
            'write', self._apply_channel_settings)        
        self.power_470.value.trace_add(
            'write', self._apply_channel_settings)
        # 510/25:
        self.power_510 = tkcw.CheckboxSliderSpinbox(
            frame,
            label='510/25nm (%)',
            color='blue',
            slider_length=200,
            default_value=5,
            row=3,
            width=5)
        self.power_510.checkbox_value.trace_add(
            'write', self._apply_channel_settings)        
        self.power_510.value.trace_add(
            'write', self._apply_channel_settings)
        for child in self.power_510.winfo_children():
            child.configure(state='disable')
        # 550/15:
        self.power_550 = tkcw.CheckboxSliderSpinbox(
            frame,
            label='550/15nm (%)',
            color='green',
            slider_length=200,
            default_value=5,
            row=4,
            width=5)
        self.power_550.checkbox_value.trace_add(
            'write', self._apply_channel_settings)        
        self.power_550.value.trace_add(
            'write', self._apply_channel_settings)
        # 640/30:
        self.power_640 = tkcw.CheckboxSliderSpinbox(
            frame,
            label='640/30nm (%)',
            color='red',
            slider_length=200,
            default_value=5,
            row=5,
            width=5)
        self.power_640.checkbox_value.trace_add(
            'write', self._apply_channel_settings)        
        self.power_640.value.trace_add(
            'write', self._apply_channel_settings)
        return None

    def _apply_channel_settings(self, var, index, mode):
        # var, index, mode are passed from .trace_add but not used
        channels_per_image, power_per_channel = [], []
        if self.power_tl.checkbox_value.get():
            channels_per_image.append('TL_LED')
            power_per_channel.append(self.power_tl.value.get())
        if self.power_395.checkbox_value.get():
            channels_per_image.append('395/25')
            power_per_channel.append(self.power_395.value.get())
        if self.power_440.checkbox_value.get():
            channels_per_image.append('440/20')
            power_per_channel.append(self.power_440.value.get())
        if self.power_470.checkbox_value.get():
            channels_per_image.append('470/24')
            power_per_channel.append(self.power_470.value.get())
        if self.power_510.checkbox_value.get():
            channels_per_image.append('510/25')
            power_per_channel.append(self.power_510.value.get())
        if self.power_550.checkbox_value.get():
            channels_per_image.append('550/15')
            power_per_channel.append(self.power_550.value.get())
        if self.power_640.checkbox_value.get():
            channels_per_image.append('640/30')
            power_per_channel.append(self.power_640.value.get())
        if len(channels_per_image) > 0: # at least 1 channel selected
            self.scope.apply_settings(channels_per_image=channels_per_image,
                                      power_per_channel=power_per_channel)
        return None

    def init_dichroic_mirror(self):
        frame = tk.LabelFrame(self.root, text='DICHROIC MIRROR', bd=6)
        frame.grid(row=9, column=0, padx=5, pady=5, sticky='n')
        frame_tip = Hovertip(
            frame,
            "The 'DICHROIC MIRROR' couples the LED light into the\n" +
            "microscope (and blocks some of the emission light). Search\n" +
            "the part number to see the specification.")
        inner_frame = tk.LabelFrame(frame, text='fixed')
        inner_frame.grid(row=0, column=0, padx=10, pady=10)
        dichroic_mirror_options = tuple(repi.dichroic_mirror_options.keys())
        self.dichroic_mirror = tk.StringVar()
        self.dichroic_mirror.set(dichroic_mirror_options[0]) # set default
        option_menu = tk.OptionMenu(
            inner_frame,
            self.dichroic_mirror,
            *dichroic_mirror_options)
        option_menu.config(width=46, height=2) # match to TL and lasers
        option_menu.grid(row=0, column=0, padx=10, pady=10)
        self.dichroic_mirror.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                dichroic_mirror=self.dichroic_mirror.get()))        
        return None

    def init_filter_wheel(self):
        frame = tk.LabelFrame(self.root, text='FILTER WHEEL', bd=6)
        frame.grid(row=10, column=0, padx=5, pady=5, sticky='n')
        frame_tip = Hovertip(
            frame,
            "The 'FILTER WHEEL' has a choice of 'emission filters'\n" +
            "(typically used to stop LED light reaching the camera).\n" +
            "Search the part numbers to see the specifications.")
        inner_frame = tk.LabelFrame(frame, text='choice')
        inner_frame.grid(row=0, column=0, padx=10, pady=10)
        emission_filter_options = tuple(repi.emission_filter_options.keys())
        self.emission_filter = tk.StringVar()
        self.emission_filter.set(emission_filter_options[6]) # set default
        option_menu = tk.OptionMenu(
            inner_frame,
            self.emission_filter,
            *emission_filter_options)
        option_menu.config(width=46, height=2) # match to TL and lasers
        option_menu.grid(row=0, column=0, padx=10, pady=10)
        self.emission_filter.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                emission_filter=self.emission_filter.get()))
        return None

    def init_camera(self):
        frame = tk.LabelFrame(self.root, text='CAMERA', bd=6)
        frame.grid(row=1, column=1, rowspan=4, columnspan=2,
                   padx=5, pady=5, sticky='n')
        # illumination_time_us:
        self.illumination_time_us = tkcw.CheckboxSliderSpinbox(
            frame,
            label='illumination time (us)',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=100,
            max_value=475000,
            default_value=10000,
            columnspan=2,
            row=0,
            width=10,
            sticky='w')
        self.illumination_time_us.value.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                illumination_time_us=self.illumination_time_us.value.get()))
        illumination_time_us_tip = Hovertip(
            self.illumination_time_us,
            "The 'illumination time (us)' determines how long the sample\n" +
            "will be exposed to light (i.e. the camera will collect the\n" +
            "emmitted light during this time).\n" +
            "NOTE: the range in the GUI is 100us to 475000us (0.475s).")
        # height_px:
        self.height_px = tkcw.CheckboxSliderSpinbox(
            frame,
            label='height pixels',
            orient='vertical',
            checkbox_enabled=False,
            slider_length=190,
            tickinterval=4,
            slider_flipped=True,
            min_value=10,
            max_value=2048,
            default_value=2048,
            row=1,
            width=5)
        self.height_px.value.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                height_px=self.height_px.value.get()))
        height_px_tip = Hovertip(
            self.height_px,
            "The 'height pixels' determines how many vertical pixels are\n" +
            "used by the camera. Less pixels is a smaller field of view\n" +
            "(FOV) and less data.\n" +
            "NOTE: less vertical pixels speeds up the acquisition!")
        # width_px:
        self.width_px = tkcw.CheckboxSliderSpinbox(
            frame,
            label='width pixels',
            checkbox_enabled=False,
            slider_length=235,
            tickinterval=4,
            min_value=64,
            max_value=2048,
            default_value=2048,
            row=2,
            column=1,
            sticky='s',
            width=5)
        self.width_px.value.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                width_px=self.width_px.value.get()))
        width_px_tip = Hovertip(
            self.width_px,
            "The 'width pixels' determines how many horizontal pixels are\n" +
            "used by the camera. Less pixels is a smaller field of view\n" +
            "(FOV) and less data.\n")
        # ROI display:
        tkcw.CanvasRectangleSliderTrace2D(
            frame,
            self.width_px,
            self.height_px,
            row=1,
            column=1,
            fill='yellow')
        return None

    def init_focus_piezo(self):
        self.focus_piezo_frame = tk.LabelFrame(
            self.root, text='FOCUS PIEZO', bd=6)
        self.focus_piezo_frame.grid(
            row=6, column=1, rowspan=4, padx=5, pady=5, sticky='nw')
        frame_tip = Hovertip(
            self.focus_piezo_frame,
            "The 'FOCUS PIEZO' is a (fast) fine focus device for precisley\n" +
            "adjusting the focus of the primary objective over a short\n" +
            "range.")
        min_um, max_um = 0, 400
        small_move_um, large_move_um = 1, 5
        center_um = int(round((max_um - min_um) / 2))
        # slider:
        self.focus_piezo_z_um = tkcw.CheckboxSliderSpinbox(
            self.focus_piezo_frame,
            label='position (um)',
            orient='vertical',
            checkbox_enabled=False,
            slider_fast_update=True,
            slider_length=235,
            tickinterval=8,
            slider_flipped=True,
            min_value=min_um,
            max_value=max_um,
            rowspan=5,
            width=5)
        def _move():
            self.scope.apply_settings(
                focus_piezo_z_um=(self.focus_piezo_z_um.value.get(),
                                  'absolute'))
            if self.running_scout_mode.get():
                self._snap_and_display()
            return None
        self.focus_piezo_z_um.value.trace_add(
            'write',
            lambda var, index, mode: _move())
        def _update_position(how):
            # check current position:
            z_um = self.focus_piezo_z_um.value.get()
            # check which direction:
            if how == 'large_up':     z_um += large_move_um
            if how == 'small_up':     z_um += small_move_um
            if how == 'center':       z_um  = center_um
            if how == 'small_down':   z_um -= small_move_um
            if how == 'large_down':   z_um -= large_move_um
            # update:
            self.focus_piezo_z_um.update_and_validate(z_um)
            return None
        button_width, button_height = 8, 1
        # large up button:
        button_large_move_up = tk.Button(
            self.focus_piezo_frame,
            text="+ %ium"%large_move_um,
            command=lambda d='large_up': _update_position(d),
            width=button_width,
            height=button_height)
        button_large_move_up.grid(row=0, column=1, padx=10, pady=10)
        # small up button:
        button_small_move_up = tk.Button(
            self.focus_piezo_frame,
            text="+ %ium"%small_move_um,
            command=lambda d='small_up': _update_position(d),
            width=button_width,
            height=button_height)
        button_small_move_up.grid(row=1, column=1, sticky='s')
        # center button:
        button_center_move = tk.Button(
            self.focus_piezo_frame,
            text="center",
            command=lambda d='center': _update_position(d),
            width=button_width,
            height=button_height)
        button_center_move.grid(row=2, column=1, padx=5, pady=5)
        # small down button:
        button_small_move_down = tk.Button(
            self.focus_piezo_frame,
            text="- %ium"%small_move_um,
            command=lambda d='small_down': _update_position(d),
            width=button_width,
            height=button_height)
        button_small_move_down.grid(row=3, column=1, sticky='n')
        # large down button:
        button_large_move_down = tk.Button(
            self.focus_piezo_frame,
            text="- %ium"%large_move_um,
            command=lambda d='large_down': _update_position(d),
            width=button_width,
            height=button_height)
        button_large_move_down.grid(row=4, column=1, padx=10, pady=10)
        return None

    def init_autofocus(self):
        frame = tk.LabelFrame(self.root, text='AUTOFOCUS', bd=6)
        frame.grid(row=6, column=2, rowspan=4, padx=5, pady=5, sticky='ne')
        spinbox_width = 20
        # objective name:
        objective_options = tuple(repi.objective_options['name'])
        self.objective_name = tk.StringVar()
        self.objective_name.set(objective_options[2]) # set default
        objective_name_option_menu = tk.OptionMenu(
            frame,
            self.objective_name,
            *objective_options)
        objective_name_option_menu.config(width=spinbox_width, height=2)
        objective_name_option_menu.grid(row=0, column=0, padx=10, pady=10)
        self.objective_name.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                objective_name=self.objective_name.get()))
        objective_name_option_menu_tip = Hovertip(
            objective_name_option_menu,
            "The current primary objective according to the GUI.\n" +
            "NOTE: this should match the physical objective on the\n" +
            "microscope!")
        # sample flag:
        self.autofocus_sample_flag = tk.BooleanVar()
        autofocus_sample_flag_textbox = tkcw.Textbox(
            frame,
            label='Sample flag',
            default_text='None',
            row=1,
            width=spinbox_width,
            height=1)
        autofocus_sample_flag_textbox.textbox.tag_add('color', '1.0', 'end')
        def _update_autofocus_sample_flag():
            autofocus_sample_flag_textbox.textbox.delete('1.0', 'end')
            text, bg = 'False', 'white'
            if self.autofocus_sample_flag.get():
                text, bg = 'True', 'green'
            autofocus_sample_flag_textbox.textbox.tag_config(
                'color', background=bg)
            autofocus_sample_flag_textbox.textbox.insert('1.0', text, 'color')
            return None
        self.autofocus_sample_flag.trace_add(
            'write',
            lambda var, index, mode: _update_autofocus_sample_flag())
        autofocus_sample_flag_textbox_tip = Hovertip(
            autofocus_sample_flag_textbox,
            "Shows the status of the 'Sample flag' from the hardware\n" +
            "autofocus.\n" +
            "NOTE: the 'Sample flag' must be 'True' to lock the autofocus.")
        # offset lens:
        self.autofocus_offset_lens = tk.IntVar()
        autofocus_offset_lens_textbox = tkcw.Textbox(
            frame,
            label='Offset lens',
            default_text='None',
            row=2,
            width=spinbox_width,
            height=1)
        def _update_autofocus_offset_lens():
            autofocus_offset_lens_textbox.textbox.delete('1.0', 'end')
            autofocus_offset_lens_textbox.textbox.insert(
                '1.0', self.autofocus_offset_lens.get())
            return None
        self.autofocus_offset_lens.trace_add(
            'write',
            lambda var, index, mode: _update_autofocus_offset_lens())
        autofocus_offset_lens_textbox_tip = Hovertip(
            autofocus_offset_lens_textbox,
            "The current position of the autofocus offset lens.\n" +
            "The offset lens adjusts the lock position of the autofocus\n" +
            "(active when the autofocus is enabled).\n" +
            "NOTE: this is adjusted with the 'knob' on the autofocus box")
        # focus flag:
        self.autofocus_focus_flag = tk.BooleanVar()
        autofocus_focus_flag_textbox = tkcw.Textbox(
            frame,
            label='Focus flag',
            default_text='None',
            row=3,
            width=spinbox_width,
            height=1)
        autofocus_focus_flag_textbox.textbox.tag_add('color', '1.0', 'end')
        def _update_autofocus_focus_flag():
            autofocus_focus_flag_textbox.textbox.delete('1.0', 'end')
            text, bg = 'False', 'white'
            if self.autofocus_focus_flag.get():
                text, bg = 'True', 'green'
            autofocus_focus_flag_textbox.textbox.tag_config(
                'color', background=bg)
            autofocus_focus_flag_textbox.textbox.insert('1.0', text, 'color')
            return None
        self.autofocus_focus_flag.trace_add(
            'write',
            lambda var, index, mode: _update_autofocus_focus_flag())
        autofocus_focus_flag_textbox_tip = Hovertip(
            autofocus_focus_flag_textbox,
            "Shows the status of the 'Focus flag' from the hardware\n" +
            "autofocus.\n" +
            "NOTE: the 'focus flag' should be 'True' if the autofocus is\n" +
            "locked.")
        def _autofocus():
            if self.autofocus_enabled.get():
                # hide focus piezo:
                self.focus_piezo_frame.grid_remove()
                # attempt autofocus:
                self.scope.apply_settings(autofocus_enabled=True).get_result()
                if not self.scope.autofocus_enabled: # autofocus failed
                    def _cancel():                        
                        # show focus piezo:
                        self.focus_piezo_frame.grid()
                        # release button:
                        self.autofocus_enabled.set(0)
                    self.root.after(int(1e3/2), _cancel) # 2fps
                else:
                    self._snap_and_display()
            else:
                self.scope.apply_settings(autofocus_enabled=False).get_result()
                # update gui with any changes from autofocus:
                self.focus_piezo_z_um.update_and_validate(
                    int(round(self.scope.focus_piezo_z_um)))
                # show focus piezo:
                self.focus_piezo_frame.grid()
            return None
        self.autofocus_enabled = tk.BooleanVar()
        autofocus_button = tk.Checkbutton(
            frame,
            text="Enable/Disable",
            variable=self.autofocus_enabled,
            command=_autofocus,
            indicatoron=0,
            width=25,
            height=2)
        autofocus_button.grid(row=4, column=0, padx=10, pady=10)
        autofocus_button_tip = Hovertip(
            autofocus_button,
            "The 'AUTOFOCUS' will attempt to continously maintain a set\n" +
            "distance between the primary objective and the sample. This\n" +
            "distance (focus) can be adjusted by turning the 'knob' on the\n" +
            "'PRIOR PureFocus850 controller'.\n" +
            "NOTE: this typically only works if the sample is already\n " +
            "'very close' to being in focus:\n " +
            "-> It is NOT intented to find the sample or find focus.\n " +
            "-> Do NOT press any of the buttons on the controller.\n ")
        return None

    def _check_autofocus(self):
        self.autofocus_offset_lens.set(
            self.scope.autofocus._get_offset_lens_position())
        self.autofocus_sample_flag.set(self.scope.autofocus.get_sample_flag())
        self.autofocus_focus_flag.set(self.scope.autofocus.get_focus_flag())
        if self.autofocus_enabled.get() and self.running_scout_mode.get():
            offset = self.scope.autofocus.offset_lens_position
            if offset != self.scope.autofocus._get_offset_lens_position():
                self._snap_and_display()
        return None

    def init_XY_stage(self):
        frame = tk.LabelFrame(self.root, text='XY STAGE', bd=6)
        frame.grid(row=10, column=1, rowspan=2, columnspan=2,
                   padx=5, pady=5, sticky='n')
        frame_tip = Hovertip(
            frame,
            "The 'XY STAGE' moves the sample in XY with a high degree of\n" +
            "accuracy (assuming the sample does not move).\n"
            "To help with XY navigation this panels shows:\n"
            "- The direction of the 'last move'.\n" +
            "- The absolute '[X, Y] position (mm)'.\n" +
            "- The absolute '[Z] position (mm)'.\n" +
            "- Move buttons for 'left', 'right', 'up' and 'down'.\n" +
            "- A slider bar for the 'step size (% of FOV)', which \n" +
            "determines how much the move buttons will move as a % of the\n" +
            "current field of view (FOV).")
        # position:
        self.XY_stage_position_mm = tk.StringVar()
        self.XY_stage_position_mm.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                XY_stage_position_mm=(self.X_stage_position_mm,
                                      self.Y_stage_position_mm,
                                      'absolute')))
        # position textbox:
        self.XY_stage_position_textbox = tkcw.Textbox(
            frame,
            label='[X, Y] position (mm)',
            row=1,
            column=1,
            height=1,
            width=20)
        # z position textbox:
        self.Z_stage_position_mm = tk.DoubleVar()
        Z_position_textbox = tkcw.Textbox(
            frame,
            label='Z position (mm)',
            column=2,
            height=1,
            width=10)
        def _update_z_position():
            Z_position_textbox.textbox.delete('1.0', 'end')
            Z_string = '[%0.3f]'%self.Z_stage_position_mm.get()
            Z_position_textbox.textbox.insert('1.0', Z_string)
            return None
        self.Z_stage_position_mm.trace_add(
            'write',
            lambda var, index, mode: _update_z_position())
        # last move textbox:
        self.last_move = tk.StringVar()
        last_move_textbox = tkcw.Textbox(
            frame,
            label='last move',
            default_text='None',
            height=1,
            width=10)
        def _update_last_move():
            last_move_textbox.textbox.delete('1.0', 'end')
            last_move_textbox.textbox.insert('1.0', self.last_move.get())
            return None
        self.last_move.trace_add(
            'write',
            lambda var, index, mode: _update_last_move())
        def _update_position(how):
            # calculate move size:
            move_factor = move_pct.value.get() / 100
            ud_fov_um = self.scope.height_px * self.scope.sample_px_um
            ud_move_mm = 1e-3 * ud_fov_um * move_factor
            lr_fov_um = self.scope.width_px * self.scope.sample_px_um
            lr_move_mm = 1e-3 * lr_fov_um * move_factor
            # check which direction:
            if how == 'up (+Y)':       move_mm = (0,  ud_move_mm)
            if how == 'down (-Y)':     move_mm = (0, -ud_move_mm)
            if how == 'left (-X)':     move_mm = (-lr_move_mm, 0)
            if how == 'right (+X)':    move_mm = ( lr_move_mm, 0)
            # update:
            self.last_move.set(how)
            self._update_XY_stage_position(
                [self.X_stage_position_mm + move_mm[0],
                 self.Y_stage_position_mm + move_mm[1]])
            if self.running_scout_mode.get():
                self._snap_and_display()
            return None
        # move size:
        move_pct = tkcw.CheckboxSliderSpinbox(
            frame,
            label='step size (% of FOV)',
            checkbox_enabled=False,
            slider_length=300,
            tickinterval=6,
            min_value=1,
            max_value=100,
            default_value=50,
            row=4,
            columnspan=3,
            width=5)
        button_width, button_height = 10, 2
        # up button:
        button_up = tk.Button(
            frame,
            text="up",
            command=lambda d='up (+Y)': _update_position(d),
            width=button_width,
            height=button_height)
        button_up.grid(row=0, column=1, padx=5, pady=5)
        # down button:
        button_down = tk.Button(
            frame,
            text="down",
            command=lambda d='down (-Y)': _update_position(d),
            width=button_width,
            height=button_height)
        button_down.grid(row=2, column=1, padx=5, pady=5)
        # left button:
        button_left = tk.Button(
            frame,
            text="left",
            command=lambda d='left (-X)': _update_position(d),
            width=button_width,
            height=button_height)
        button_left.grid(row=1, column=0, padx=5, pady=5)
        # right button:
        button_right = tk.Button(
            frame,
            text="right",
            command=lambda d='right (+X)': _update_position(d),
            width=button_width,
            height=button_height)
        button_right.grid(row=1, column=2, padx=5, pady=5)
        return None

    def _update_XY_stage_position(self, XY_stage_position_mm):
        X, Y = XY_stage_position_mm[0], XY_stage_position_mm[1]
        XY_string = '[%0.3f, %0.3f]'%(X, Y)
        # textbox:
        self.XY_stage_position_textbox.textbox.delete('1.0', 'end')
        self.XY_stage_position_textbox.textbox.insert('1.0', XY_string)
        # attributes
        self.X_stage_position_mm, self.Y_stage_position_mm = X, Y
        self.XY_stage_position_mm.set(XY_string)
        return None

    def _check_joystick(self):
        XY_mm = list(self.scope.XY_stage_position_mm)
        joystick_active = False
        status_byte = self.scope.XYZ_stage._get_status_byte()
        if status_byte != '10 10 10':
            joystick_active = True
        if (joystick_active and self.running_scout_mode.get()):
            self._snap_and_display()
        if (not joystick_active and (
            XY_mm[0] != self.X_stage_position_mm or
            XY_mm[1] != self.Y_stage_position_mm)):
            self._update_XY_stage_position(XY_mm)
        # update z:
        self.Z_stage_position_mm.set(self.scope.Z_stage_position_mm)
        return None

    def _snap_and_display(self):
        if self.images_per_buffer.value.get() != 1:
            self.images_per_buffer.update_and_validate(1)
        self.last_acquire_task.get_result() # don't accumulate
        self.last_acquire_task = self.scope.acquire()
        return None

    def _get_folder_name(self):
        dt = datetime.strftime(datetime.now(),'%Y-%m-%d_%H-%M-%S_')
        folder_index = 0
        folder_name = (
            self.session_folder + dt +
            '%03i_'%folder_index + self.label_textbox.text)
        while os.path.exists(folder_name): # check before overwriting
            folder_index +=1
            folder_name = (
                self.session_folder + dt +
                '%03i_'%folder_index + self.label_textbox.text)
        return folder_name

    def init_grid_navigator(self):
        frame = tk.LabelFrame(self.root, text='GRID NAVIGATOR', bd=6)
        frame.grid(row=1, column=4, rowspan=4, padx=5, pady=5, sticky='n')
        button_width, button_height = 25, 1
        spinbox_width = 20
        # load from file:
        def _load_grid_from_file():
            # get file from user:
            file_path = tk.filedialog.askopenfilename(
                parent=self.root,
                initialdir=os.getcwd(),
                title='Please choose a previous "grid" file (.txt)')        
            with open(file_path, 'r') as file:
                grid_data = file.read().splitlines()
            # parse and update attributes:
            self.grid_rows.update_and_validate(int(grid_data[0].split(':')[1]))
            self.grid_cols.update_and_validate(int(grid_data[1].split(':')[1]))
            self.grid_um.update_and_validate(int(grid_data[2].split(':')[1]))
            # show user:
            _create_grid_popup()
            # reset state of grid buttons:
            self.set_grid_location_button.config(state='normal')
            self.move_to_grid_location_button.config(state='disabled')
            self.start_grid_preview_button.config(state='disabled')
            return None
        load_grid_from_file_button = tk.Button(
            frame,
            text="Load from file",
            command=_load_grid_from_file,
            font=('Segoe UI', '10', 'underline'),
            width=button_width,
            height=button_height)
        load_grid_from_file_button.grid(row=0, column=0, padx=10, pady=10)
        load_grid_from_file_tip = Hovertip(
            load_grid_from_file_button,
            "Use the 'Load from file' button to select a text file\n" +
            "'grid_navigator_parameters.txt' from a previous \n" +
            "'gui_session' folder and load these settings into\n" +
            "the GUI.\n"
            "NOTE: this will overwrite any existing grid parameters")
        # create grid popup:
        create_grid_popup = tk.Toplevel()
        create_grid_popup.title('Create grid')
        x, y = self.root.winfo_x(), self.root.winfo_y() # center popup
        create_grid_popup.geometry("+%d+%d" % (x + 800, y + 400))
        create_grid_popup.withdraw()
        def _close_create_grid_popup():
            create_grid_popup.withdraw()
            create_grid_popup.grab_release()
            return None
        create_grid_popup.protocol(
            "WM_DELETE_WINDOW", _close_create_grid_popup)        
        # popup input:
        spinbox_width = 20
        self.grid_rows = tkcw.CheckboxSliderSpinbox(
            create_grid_popup,
            label='How many rows? (1-16)',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=1,
            max_value=16,
            default_value=2,
            row=0,
            width=spinbox_width,
            sticky='n')
        self.grid_cols = tkcw.CheckboxSliderSpinbox(
            create_grid_popup,
            label='How many columns? (1-24)',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=1,
            max_value=24,
            default_value=4,
            row=1,
            width=spinbox_width,
            sticky='n')
        self.grid_um = tkcw.CheckboxSliderSpinbox(
            create_grid_popup,
            label='What is the spacing (um)?',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=1,
            max_value=20000,
            default_value=100,
            row=2,
            width=spinbox_width,
            sticky='n')
        # popup create button:
        def _create_grid():
            # tidy up any previous display:
            if hasattr(self, 'create_grid_buttons_frame'):
                self.create_grid_buttons_frame.destroy()
            # generate grid list and show buttons:
            self.create_grid_buttons_frame = tk.LabelFrame(
                create_grid_popup, text='XY GRID', bd=6)
            self.create_grid_buttons_frame.grid(
                row=0, column=1, rowspan=5, padx=10, pady=10)
            self.grid_list = []
            for r in range(self.grid_rows.value.get()):
                for c in range(self.grid_cols.value.get()):
                    name = '%s%i'%(chr(ord('@')+r + 1), c + 1)
                    grid_button = tk.Button(
                        self.create_grid_buttons_frame,
                        text=name,
                        width=5,
                        height=2)
                    grid_button.grid(row=r, column=c, padx=10, pady=10)
                    grid_button.config(state='disabled')
                    self.grid_list.append([r, c, None])
            # set button status:
            self.set_grid_location_button.config(state='normal')
            self.move_to_grid_location_button.config(state='disabled')
            self.start_grid_preview_button.config(state='disabled')
            # overwrite grid file:
            with open(self.session_folder +
                      "grid_navigator_parameters.txt", "w") as file:
                file.write('rows:%i'%self.grid_rows.value.get() + '\n')
                file.write('columns:%i'%self.grid_cols.value.get() + '\n')
                file.write('spacing_um:%i'%self.grid_um.value.get() + '\n')
            return None
        create_grid_button = tk.Button(
            create_grid_popup,
            text="Create",
            command=_create_grid,
            height=button_height,
            width=button_width)
        create_grid_button.grid(row=3, column=0, padx=10, pady=10, sticky='n')
        # create grid popup button:
        def _create_grid_popup():
            create_grid_popup.deiconify()
            create_grid_popup.grab_set() # force user to interact
            return None
        create_grid_popup_button = tk.Button(
            frame,
            text="Create grid",
            command=_create_grid_popup,
            width=button_width,
            height=button_height)
        create_grid_popup_button.grid(row=1, column=0, padx=10, pady=10)
        create_grid_tip = Hovertip(
            create_grid_popup_button,
            "Use the 'Create grid' button to create a new grid of points\n" +
            "you want to navigate (by specifying the rows, columns and\n" +
            "spacing). For example, this tool can be used to move around\n" +
            "multiwell plates (or any grid like sample).\n" +
            "NOTE: this will overwrite any existing grid parameters")
        # set location popup:
        set_grid_location_popup = tk.Toplevel()
        set_grid_location_popup.title('Set current location')
        x, y = self.root.winfo_x(), self.root.winfo_y() # center popup
        set_grid_location_popup.geometry("+%d+%d" % (x + 800, y + 400))
        set_grid_location_popup.withdraw()
        def _close_set_grid_location_popup():
            set_grid_location_popup.withdraw()
            set_grid_location_popup.grab_release()
            return None
        set_grid_location_popup.protocol(
            "WM_DELETE_WINDOW", _close_set_grid_location_popup)        
        # set location button:
        def _set_grid_location():
            set_grid_location_popup.deiconify()
            set_grid_location_popup.grab_set() # force user to interact
            # show grid buttons:
            set_grid_location_buttons_frame = tk.LabelFrame(
                set_grid_location_popup, text='XY GRID', bd=6)
            set_grid_location_buttons_frame.grid(
                row=0, column=1, rowspan=5, padx=10, pady=10)
            def _set(grid):
                # update:
                self.grid_location.set(grid)
                row, col, p_mm = self.grid_list[grid]
                # find home position:
                grid_mm = self.grid_um.value.get() / 1000
                r0c0_mm = [self.X_stage_position_mm + col * grid_mm,
                           self.Y_stage_position_mm - row * grid_mm]
                # generate positions:
                positions_mm = []
                for r in range(self.grid_rows.value.get()):
                    for c in range(self.grid_cols.value.get()):
                        positions_mm.append([r0c0_mm[0] - (c * grid_mm),
                                             r0c0_mm[1] + (r * grid_mm)])
                # update grid list:
                for g in range(len(self.grid_list)):
                    self.grid_list[g][2] = positions_mm[g]                
                # allow moves:
                self.move_to_grid_location_button.config(state='normal')
                self.start_grid_preview_button.config(state='disabled')
                if grid == 0:
                    self.start_grid_preview_button.config(state='normal')
                # exit:
                set_grid_location_buttons_frame.destroy()
                set_grid_location_popup.withdraw()
                set_grid_location_popup.grab_release()
                return None
            for g in range(len(self.grid_list)):
                r, c, p_mm = self.grid_list[g]
                name = '%s%i'%(chr(ord('@')+r + 1), c + 1)
                grid_button = tk.Button(
                    set_grid_location_buttons_frame,
                    text=name,
                    command=lambda grid=g: _set(grid),
                    width=5,
                    height=2)
                grid_button.grid(row=r, column=c, padx=10, pady=10)
            return None
        self.set_grid_location_button = tk.Button(
            frame,
            text="Set grid location",
            command=_set_grid_location,
            width=button_width,
            height=button_height)
        self.set_grid_location_button.grid(row=2, column=0, padx=10, pady=10)
        self.set_grid_location_button.config(state='disabled')
        set_grid_location_tip = Hovertip(
            self.set_grid_location_button,
            "Use the 'Set grid location' button to specify where you are\n" +
            "currently located in the grid. \n" +
            "NOTE: all other grid points will then be referenced by this\n" +
            "operation (i.e. this operation 'homes' the grid). To change\n" +
            "the grid origin simply update with this button")
        # current location:
        def _update_grid_location():
            r, c, p_mm = self.grid_list(self.grid_location.get())
            name = '%s%i'%(chr(ord('@') + r + 1), c + 1)
            self.grid_location_textbox.textbox.delete('1.0', 'end')
            self.grid_location_textbox.textbox.insert('1.0', name)
            return None
        self.grid_location = tk.IntVar()
        self.grid_location_textbox = tkcw.Textbox(
            frame,
            label='Grid location',
            default_text='None',
            height=1,
            width=20)
        self.grid_location_textbox.grid(
            row=3, column=0, padx=10, pady=10)
        self.grid_location.trace_add(
            'write',
            lambda var, index, mode: _update_grid_location)
        grid_location_tip = Hovertip(
            self.grid_location_textbox,
            "The 'Current grid location' displays the last grid location\n" +
            "that was moved to (or set) with the 'GRID NAVIGATOR' panel.\n" +
            "NOTE: it does not display the current position and is not \n" +
            "aware of XY moves made elsewhere (e.g. with the joystick \n" +
            "or 'XY STAGE' panel).")
        # move to location popup:
        move_to_grid_location_popup = tk.Toplevel()
        move_to_grid_location_popup.title('Move to location')
        x, y = self.root.winfo_x(), self.root.winfo_y() # center popup
        move_to_grid_location_popup.geometry("+%d+%d" % (x + 800, y + 400))
        move_to_grid_location_popup.withdraw()
        # cancel popup button:
        def _close_move_to_grid_location_popup():
            move_to_grid_location_popup.withdraw()
            move_to_grid_location_popup.grab_release()
            return None
        move_to_grid_location_popup.protocol(
            "WM_DELETE_WINDOW", _close_move_to_grid_location_popup)
        # move to location button:
        def _move_to_grid_location():
            move_to_grid_location_popup.deiconify()
            move_to_grid_location_popup.grab_set() # force user to interact
            # show grid buttons:
            move_to_grid_location_buttons_frame = tk.LabelFrame(
                move_to_grid_location_popup, text='XY GRID', bd=6)
            move_to_grid_location_buttons_frame.grid(
                row=0, column=1, rowspan=5, padx=10, pady=10)            
            def _move(grid):
                # update position and display:
                self._update_XY_stage_position(self.grid_list[grid][2])
                self._snap_and_display()
                # update attributes and buttons:
                self.grid_location.set(grid)
                self.start_grid_preview_button.config(state='disabled')
                if grid == 0:
                    self.start_grid_preview_button.config(state='normal')
                # exit:
                _close_move_to_grid_location_popup()
                return None
            for g in range(len(self.grid_list)):
                r, c, p_mm = self.grid_list[g]
                name = '%s%i'%(chr(ord('@') + r + 1), c + 1)
                grid_button = tk.Button(
                    move_to_grid_location_buttons_frame,
                    text=name,
                    command=lambda grid=g: _move(grid),
                    width=5,
                    height=2)
                grid_button.grid(row=r, column=c, padx=10, pady=10)
                if g == self.grid_location.get():
                    grid_button.config(state='disabled')
            return None
        self.move_to_grid_location_button = tk.Button(
            frame,
            text="Move to grid location",
            command=_move_to_grid_location,
            width=button_width,
            height=button_height)
        self.move_to_grid_location_button.grid(
            row=4, column=0, padx=10, pady=10)
        self.move_to_grid_location_button.config(state='disabled')
        move_to_grid_location_tip = Hovertip(
            self.move_to_grid_location_button,
            "The 'Move to grid location' button moves to the chosen grid\n" +
            "location based on the absolute XY grid positions that have\n" +
            "been loaded or created. The grid origin is set by the 'Set\n" +
            "grid location' button.\n")
        # save position:
        self.save_grid_position = tk.BooleanVar()
        save_grid_position_button = tk.Checkbutton(
            frame,
            text='Save position',
            variable=self.save_grid_position)
        save_grid_position_button.grid(
            row=5, column=0, padx=10, pady=10, sticky='w')
        save_grid_position_tip = Hovertip(
            save_grid_position_button,
            "If 'Save position' is enabled then the 'Start grid preview \n" +
            "(from A1)' button will populate the 'POSITION LIST'.")
        # tile the grid:
        self.tile_the_grid = tk.BooleanVar()
        tile_the_grid_button = tk.Checkbutton(
            frame,
            text='Tile the grid',
            variable=self.tile_the_grid)
        tile_the_grid_button.grid(
            row=6, column=0, padx=10, pady=10, sticky='w')
        tile_the_grid_tip = Hovertip(
            tile_the_grid_button,
            "If 'Tile the grid' is enabled then the 'Start grid preview\n" +
            "(from A1)' button will tile the grid locations with the number\n" +
            "of tiles set by the 'TILE NAVIGATOR'.")
        # start grid preview:
        def _start_grid_preview():
            print('\nGrid preview -> started')
            self._set_running_mode('grid_preview')
            if self.images_per_buffer.value.get() != 1:
                self.images_per_buffer.update_and_validate(1)
            if not self.tile_the_grid.get():
                folder_name = self._get_folder_name() + '_grid'
                self.grid_preview_list = self.grid_list
            else:
                folder_name = self._get_folder_name() + '_grid_tile'
                # calculate move size:
                tile_X_mm = (
                    1e-3 * self.width_px.value.get() * self.scope.sample_px_um)
                tile_Y_mm = 1e-3 * self.scan_range_um.value.get()
                # update preview list:
                self.grid_preview_list = []
                for g in range(len(self.grid_list)):
                    gr, gc, g_mm = self.grid_list[g]
                    for tr in range(self.tile_rc.value.get()):
                        for tc in range(self.tile_rc.value.get()):
                            p_mm = [g_mm[0] - tc * tile_X_mm,
                                    g_mm[1] + tr * tile_Y_mm]
                            self.grid_preview_list.append(
                                (gr, gc, tr, tc, p_mm))
            self.current_grid_preview = 0
            def _run_grid_preview():
                # get co-ords/name and update location:
                if not self.tile_the_grid.get():
                    gr, gc, p_mm = self.grid_preview_list[
                        self.current_grid_preview]
                    name = '%s%i'%(chr(ord('@') + gr + 1), gc + 1)
                    self.grid_location.set(self.current_grid_preview)
                else:
                    gr, gc, tr, tc, p_mm = self.grid_preview_list[
                        self.current_grid_preview]
                    name = '%s%i_r%ic%i'%(
                        chr(ord('@') + gr + 1), gc + 1, tr, tc)
                    if (tr, tc) == (0, 0):
                        self.grid_location.set(gr + gc)
                # move:
                self._update_XY_stage_position(p_mm)
                # check mode:
                if self.save_grid_position.get():
                    self._update_position_list()
                # get image:
                filename = name + '.tif'
                self.scope.acquire(
                    filename=filename,
                    folder_name=folder_name,
                    description=self.description_textbox.text).get_result()
                grid_preview_filename = (folder_name + '\\data\\' + filename)
                while not os.path.isfile(grid_preview_filename):
                    self.root.after(int(1e3/30)) # 30fps
                image = imread(grid_preview_filename)
                if len(image.shape) == 2:
                    image = image[np.newaxis,:] # add channels, no image series
                if self.scope.timestamp_mode == "binary+ASCII":
                    image = image[:,8:,:]
                shape = image.shape
                # add reference:
                XY = (int(0.1 * min(shape[-2:])),
                      shape[1] - int(0.15 * min(shape[-2:])))
                font_size = int(0.1 * min(shape[-2:]))
                font = ImageFont.truetype('arial.ttf', font_size)
                for ch in range(shape[0]):
                    # convert 2D image to PIL format for ImageDraw:                    
                    im = Image.fromarray(image[ch,:]) # convert to ImageDraw
                    ImageDraw.Draw(im).text(XY, name, fill=0, font=font)
                    image[ch,:] = im
                # make grid image:
                if not self.tile_the_grid.get():
                    if self.current_grid_preview == 0:
                        self.grid_preview = np.zeros(
                            (shape[0],
                             self.grid_rows.value.get() * shape[1],
                             self.grid_cols.value.get() * shape[2]),
                            'uint16')
                    self.grid_preview[:,
                                      gr * shape[1]:(gr + 1) * shape[1],
                                      gc * shape[2]:(gc + 1) * shape[2]
                                      ] = image
                else:
                    if self.current_grid_preview == 0:
                        self.grid_preview = np.zeros(
                            (shape[0],
                             self.grid_rows.value.get() *
                             shape[1] * self.tile_rc.value.get(),
                             self.grid_cols.value.get() *
                             shape[2] * self.tile_rc.value.get()),
                            'uint16')
                    self.grid_preview[
                        :,
                        (gr * self.tile_rc.value.get() + tr) * shape[1]:
                        (gr * self.tile_rc.value.get() + tr + 1) * shape[1],
                        (gc * self.tile_rc.value.get() + tc) * shape[2]:
                        (gc * self.tile_rc.value.get() + tc + 1) * shape[2]
                        ] = image
                # display:
                self.scope.display.show_grid_image(self.grid_preview)
                # check before re-run:
                if (self.running_grid_preview.get() and
                    self.current_grid_preview < len(
                        self.grid_preview_list) - 1):
                    self.current_grid_preview += 1
                    self.root.after(int(1e3/30), _run_grid_preview) # 30fps
                else:
                    self._set_running_mode('None')
                    print('Grid preview -> finished\n')
                return None
            _run_grid_preview()
            return None
        self.running_grid_preview = tk.BooleanVar()
        self.start_grid_preview_button = tk.Checkbutton(
            frame,
            text="Start grid preview (from A1)",
            variable=self.running_grid_preview,
            command=_start_grid_preview,
            indicatoron=0,
            font=('Segoe UI', '10', 'italic'),
            width=button_width,
            height=button_height)
        self.start_grid_preview_button.grid(row=7, column=0, padx=10, pady=10)
        self.start_grid_preview_button.config(state='disabled')
        start_grid_preview_tip = Hovertip(
            self.start_grid_preview_button,
            "The 'Start grid preview (from A1)' button will start to \n" +
            "collect data for the whole grid of points (starting \n" +
            "at A1). Consider using 'Save position' and 'Tile the grid'\n" +
            "for extra functionality.")
        return None

    def init_tile_navigator(self):
        frame = tk.LabelFrame(self.root, text='TILE NAVIGATOR', bd=6)
        frame.grid(row=6, column=4, rowspan=4, padx=5, pady=5, sticky='n')
        button_width, button_height = 25, 2
        spinbox_width = 20
        # tile array width:
        self.tile_rc = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Array height and width (tiles)',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=2,
            max_value=9,
            default_value=2,
            row=0,
            width=spinbox_width)
        tile_array_width_tip = Hovertip(
            self.tile_rc,
            "The 'Array height and width (tiles)' determines how many tiles\n" +
            "the 'Start tile' button will generate. For example, 2 gives a\n" +
            "2x2 array of tiles, 3 a 3x3 array, etc.")
        # save position:
        self.save_tile_position = tk.BooleanVar()
        save_tile_position_button = tk.Checkbutton(
            frame,
            text='Save position',
            variable=self.save_tile_position)
        save_tile_position_button.grid(
            row=1, column=0, padx=10, pady=10, sticky='w')
        save_tile_data_and_position_tip = Hovertip(
            save_tile_position_button,
            "If 'Save position' is enabled then the 'Start tile' button\n" +
            "will populate the 'POSITION LIST'.")
        # start tile preview:
        def _start_tile_preview():
            print('\nTile preview -> started')
            self._set_running_mode('tile_preview')
            if self.images_per_buffer.value.get() != 1:
                self.images_per_buffer.update_and_validate(1)
            folder_name = self._get_folder_name() + '_tile'
            # calculate move size:
            X_move_mm = 1e-3 * self.scope.width_px * self.scope.sample_px_um
            Y_move_mm = 1e-3 * self.scope.height_px * self.scope.sample_px_um
            # generate tile list:
            self.tile_list = []
            for r in range(self.tile_rc.value.get()):
                for c in range(self.tile_rc.value.get()):
                    p_mm = (self.X_stage_position_mm - c * X_move_mm,
                            self.Y_stage_position_mm + r * Y_move_mm)
                    self.tile_list.append((r, c, p_mm))
            self.current_tile = 0
            def _run_tile_preview():
                # update position:
                r, c, p_mm = self.tile_list[self.current_tile]
                self._update_XY_stage_position(p_mm)
                # get tile:
                name = "r%ic%i"%(r, c)
                filename = name + '.tif'
                if self.save_tile_position.get():
                    self._update_position_list()
                self.scope.acquire(
                    filename=filename,
                    folder_name=folder_name,
                    description=self.description_textbox.text).get_result()
                tile_filename = (folder_name + '\\data\\' + filename)
                while not os.path.isfile(tile_filename):
                    self.root.after(int(1e3/30)) # 30fps
                tile = imread(tile_filename)
                if len(tile.shape) == 2:
                    tile = tile[np.newaxis,:] # add channels, no image series
                if self.scope.timestamp_mode == "binary+ASCII":
                    tile = tile[:,8:,:]
                shape = tile.shape
                # add reference:
                XY = (int(0.1 * min(shape[-2:])),
                      shape[1] - int(0.15 * min(shape[-2:])))
                font_size = int(0.1 * min(shape[-2:]))
                font = ImageFont.truetype('arial.ttf', font_size)
                for ch in range(shape[0]):
                    # convert 2D image to PIL format for ImageDraw:
                    t = Image.fromarray(tile[ch,:])
                    ImageDraw.Draw(t).text(XY, name, fill=0, font=font)
                    tile[ch,:] = t
                # make base image:
                if self.current_tile == 0:
                    self.tile_preview = np.zeros(
                        (shape[0],
                         self.tile_rc.value.get() * shape[1],
                         self.tile_rc.value.get() * shape[2]),
                        'uint16')
                # add current tile:
                self.tile_preview[:,
                                  r * shape[1]:(r + 1) * shape[1],
                                  c * shape[2]:(c + 1) * shape[2]] = tile
                # display:
                self.scope.display.show_tile_image(self.tile_preview)
                if (self.running_tile_preview.get() and
                    self.current_tile < len(self.tile_list) - 1): 
                    self.current_tile += 1
                    self.root.after(int(1e3/30), _run_tile_preview) # 30fps
                else:
                    self._set_running_mode('None')
                    self.move_to_tile_button.config(state='normal')
                    print('Tile preview -> finished\n')
                return None
            _run_tile_preview()
            return None
        self.running_tile_preview = tk.BooleanVar()
        start_tile_preview_button = tk.Checkbutton(
            frame,
            text="Start tile",
            variable=self.running_tile_preview,
            command=_start_tile_preview,
            indicatoron=0,
            font=('Segoe UI', '10', 'italic'),
            width=button_width,
            height=button_height)
        start_tile_preview_button.grid(row=2, column=0, padx=10, pady=10)
        start_tile_tip = Hovertip(
            start_tile_preview_button,
            "The 'Start tile' button will start to generate previews for\n" +
            "the tile array using the current XY position as the first\n" +
            "tile (the top left position r0c0). Consider using 'Save\n" +
            "position' for extra functionality.")
        # move to tile popup:
        move_to_tile_popup = tk.Toplevel()
        move_to_tile_popup.title('Move to tile')
        x, y = self.root.winfo_x(), self.root.winfo_y() # center popup
        move_to_tile_popup.geometry("+%d+%d" % (x + 800, y + 400))
        move_to_tile_popup.withdraw()
        def _close_move_to_tile_popup():
            move_to_tile_popup.withdraw()
            move_to_tile_popup.grab_release()
            return None
        move_to_tile_popup.protocol(
            "WM_DELETE_WINDOW", _close_move_to_tile_popup)
        # move to tile button:
        def _move_to_tile():
            move_to_tile_popup.deiconify()
            move_to_tile_popup.grab_set() # force user to interact
            # make buttons:
            tile_buttons_frame = tk.LabelFrame(
                move_to_tile_popup, text='XY TILES', bd=6)
            tile_buttons_frame.grid(
                row=0, column=1, rowspan=5, padx=10, pady=10)
            def _move(tile):
                self._update_XY_stage_position(self.tile_list[tile][2])
                self._snap_and_display()
                self.current_tile = tile
                _close_move_to_tile_popup()
                return None
            for t in range(len(self.tile_list)):
                r, c, p_mm = self.tile_list[t]
                tile_button = tk.Button(
                    tile_buttons_frame,
                    text='r%ic%i'%(r, c),
                    command=lambda tile=t: _move(tile),
                    width=5,
                    height=2)
                tile_button.grid(row=r, column=c, padx=10, pady=10)
                if t == self.current_tile:
                    tile_button.config(state='disabled')
            return None
        self.move_to_tile_button = tk.Button(
            frame,
            text="Move to tile",
            command=_move_to_tile,
            width=button_width,
            height=button_height)
        self.move_to_tile_button.grid(row=4, column=0, padx=10, pady=10)
        self.move_to_tile_button.config(state='disabled')
        move_to_tile_tip = Hovertip(
            self.move_to_tile_button,
            "The 'Move to tile' button moves to the chosen tile location\n" +
            "based on the absolute XY tile positions from the last tile\n" +
            "routine.")
        return None

    def init_settings(self):
        frame = tk.LabelFrame(self.root, text='SETTINGS (misc)', bd=6)
        frame.grid(row=1, column=5, rowspan=9, padx=5, pady=5, sticky='n')
        button_width, button_height = 25, 1
        spinbox_width = 20
        # load from file:
        def _load_settings_from_file():
            # get file from user:
            file_path = tk.filedialog.askopenfilename(
                parent=self.root,
                initialdir=os.getcwd(),
                title='Please choose a previous "metadata" file (.txt)')        
            with open(file_path, 'r') as file:
                metadata = file.read().splitlines()
            # format into settings and values:
            file_settings = {}
            for data in metadata:
                file_settings[data.split(':')[0]] = (
                    data.split(':')[1:][0].lstrip())
            # re-format strings from file settings for gui:
            channels = file_settings[
                'channels_per_image'].strip('(').strip(')').split(',')
            powers   = file_settings[
                'power_per_channel'].strip('(').strip(')').split(',')
            channels_per_image, power_per_channel = [], []
            for i, c in enumerate(channels):
                if c == '': break # avoid bug from tuple with single entry
                channels_per_image.append(c.split("'")[1])
                power_per_channel.append(int(powers[i]))
            # turn off all illumination:
            self.power_tl.checkbox_value.set(0)
            self.power_395.checkbox_value.set(0)
            self.power_440.checkbox_value.set(0)
            self.power_470.checkbox_value.set(0)
            self.power_510.checkbox_value.set(0)
            self.power_550.checkbox_value.set(0)
            self.power_640.checkbox_value.set(0)
            # apply file settings to gui:
            for i, channel in enumerate(channels_per_image):
                if channel == 'TL_LED':
                    self.power_tl.checkbox_value.set(1)
                    self.power_tl.update_and_validate(power_per_channel[i])
                if channel == '395/25':
                    self.power_395.checkbox_value.set(1)
                    self.power_395.update_and_validate(power_per_channel[i])
                if channel == '440/20':
                    self.power_440.checkbox_value.set(1)
                    self.power_440.update_and_validate(power_per_channel[i])
                if channel == '470/24':
                    self.power_470.checkbox_value.set(1)
                    self.power_470.update_and_validate(power_per_channel[i])
                if channel == '510/25':
                    self.power_510.checkbox_value.set(1)
                    self.power_510.update_and_validate(power_per_channel[i])
                if channel == '550/15':
                    self.power_550.checkbox_value.set(1)
                    self.power_550.update_and_validate(power_per_channel[i])
                if channel == '640/30':
                    self.power_640.checkbox_value.set(1)
                    self.power_640.update_and_validate(power_per_channel[i])
            self.objective_name.set(file_settings['objective_name'])
            self.dichroic_mirror.set(file_settings['dichroic_mirror'])
            self.emission_filter.set(file_settings['emission_filter'])
            self.illumination_time_us.update_and_validate(
                int(file_settings['illumination_time_us']))
            self.height_px.update_and_validate(int(file_settings['height_px']))
            self.width_px.update_and_validate(
                int(file_settings['width_px']))
            self.images_per_buffer.update_and_validate(
                int(file_settings['images_per_buffer']))
            return None
        load_from_file_button = tk.Button(
            frame,
            text="Load from file",
            command=_load_settings_from_file,
            font=('Segoe UI', '10', 'underline'),
            width=button_width,
            height=button_height)
        load_from_file_button.grid(
            row=0, column=0, columnspan=2, padx=10, pady=10)
        load_from_file_tip = Hovertip(
            load_from_file_button,
            "Use the 'Load from file' button to select a '.txt' file from\n" +
            "the 'metadata' folder of a previous acquisition and load\n" +
            "these settings into the GUI. The loaded settings are:\n" +
            "- 'TRANSMITTED LIGHT'.\n" +
            "- 'LED BOX'.\n" +
            "- 'DICHROIC MIRROR'.\n" +
            "- 'FILTER WHEEL'.\n" +
            "- 'CAMERA'.\n" +
            "- 'OBJECTIVE NAME'.\n" +            
            "- 'Images per acquire'.\n" +
            "NOTE: 'FOCUS PIEZO', 'XY STAGE', 'Folder label' and \n" +
            "'Description' are not loaded. To load previous XYZ\n" +
            "positions use the 'POSITION LIST' panel.")
        # label textbox:
        self.label_textbox = tkcw.Textbox(
            frame,
            label='Folder label',
            default_text='repi',
            row=1,
            width=spinbox_width,
            height=1,
            columnspan=2)
        label_textbox_tip = Hovertip(
            self.label_textbox,
            "The label that will be used for the data folder (after the\n" +
            "date and time stamp). Edit to preference")
        # description textbox:
        self.description_textbox = tkcw.Textbox(
            frame,
            label='Description',
            default_text='what are you doing?',
            row=2,
            width=spinbox_width,
            height=3,
            columnspan=2)
        description_textbox_tip = Hovertip(
            self.description_textbox,
            "The text that will be recorded in the metadata '.txt' file\n" +
            "(along with the microscope settings for that acquisition).\n" +
            "Describe what you are doing here.")       
        # images spinbox:
        self.images_per_buffer = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Images per acquire',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=1,
            max_value=1e3,
            default_value=1,
            row=3,
            width=spinbox_width,
            columnspan=2)
        self.images_per_buffer.value.trace_add(
            'write',
            lambda var, index, mode: self.scope.apply_settings(
                images_per_buffer=self.images_per_buffer.value.get()))
        images_per_buffer_tip = Hovertip(
            self.images_per_buffer,
            "In short: How many back to back (as fast as possible) images\n" +
            "did you want for a given acquisition?\n" +
            "(If you are not sure or don't care then leave this as 1!)\n" +
            "In detail: increasing this number (above 1 image) pre-loads\n" +
            "more acquisitions onto the analogue out (AO) card. This has\n" +
            "pro's and con's.\n" +
            "Pros:\n" +
            "- It allows successive images to be taken with minimal \n" +
            "latency.\n" +
            "- The timing for successive images can be 'us' precise.\n" +
            "Cons:\n" +
            "- It takes time to 'load' and 'play' many images. More images\n" +
            "takes more time, and once requested this operation cannot\n"
            "be cancelled.\n" +
            "- The data from a single 'play' of the AO card is recording\n" +
            "into a single file. More images is more data and a bigger\n" +
            "file. It's possible to end up with a huge file that is not a\n" +
            "'legal' .tiff (<~4GB) and is tricky to manipulate.\n")
        # z stack:
        self.z_stack = tk.BooleanVar()
        z_stack_button = tk.Checkbutton(
            frame,
            text='Z stack',
            variable=self.z_stack)
        z_stack_button.grid(
            row=4, column=0, padx=10, pady=10, sticky='w')
        z_stack_tip = Hovertip(
            z_stack_button,
            "If checked, the 'Run acquire' button will run a 'Z stack'\n" +
            "for every position.\n" +
            "NOTE: this can take a long time and generate a lot of data!")
        self.z_stack.trace_add(
            'write',
            lambda var, index, mode: _update_z_stack_num_steps())
        # z bidirectional:
        self.z_bidirectional = tk.BooleanVar()
        z_bidirectional_button = tk.Checkbutton(
            frame,
            text='bidirectional',
            variable=self.z_bidirectional)
        z_bidirectional_button.grid(
            row=4, column=1, padx=10, pady=10, sticky='w')
        z_bidirectional_tip = Hovertip(
            z_bidirectional_button,
            "If checked, the 'Z stack' will go from -'Z stack range (um)'\n" +
            "to +'Z stack range (um)' .\n" +
            "NOTE: the focus piezo must be positioned to allow this range!")
        self.z_bidirectional.trace_add(
            'write',
            lambda var, index, mode: _update_z_stack_num_steps())
        # z range spinbox:
        self.z_range_um = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Z stack range (um)',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=1,
            max_value=100,
            increment=0.1,
            default_value=10,
            integers_only=False,
            row=5,
            width=spinbox_width,
            columnspan=2)
        z_range_um_tip = Hovertip(
            self.z_range_um,
            "How deep into the sample do you want to image?\n" +
            "-> The 'Z stack' will go from the current focus piezo\n" +
            "position to +'Z stack range (um)'\n" +
            "NOTE: the focus piezo must be positioned to allow this range!")
        self.z_range_um.value.trace_add(
            'write',
            lambda var, index, mode: _update_z_stack_num_steps())
        # z step spinbox:
        self.z_step_um = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Z stack step size (um)',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=0.1,
            max_value=99.9,
            increment=0.1,
            default_value=0.5,
            integers_only=False,
            row=6,
            width=spinbox_width,
            columnspan=2)
        z_step_um_tip = Hovertip(
            self.z_step_um,
            "What 'Z stack step size (um)' do you want?\n" +
            "NOTE: the correct way to choose this value is calculating\n" +
            "'Nyquist' step size for the PSF of the current objective. In\n" +
            "practice 0.5um is usually enough for most objectives, but\n" +
            "will over sample low numerical aperture objectives.")
        self.z_step_um.value.trace_add(
            'write',
            lambda var, index, mode: _update_z_stack_num_steps())
        # z stack num steps:
        self.z_stack_num_steps = tk.IntVar()
        z_stack_num_steps_textbox = tkcw.Textbox(
            frame,
            label='Z stack steps',
            default_text='None',
            row=7,
            width=spinbox_width,
            height=1,
            columnspan=2)
        z_stack_num_steps_tip = Hovertip(
            self.z_step_um,
            "How many 'Z stack step steps' will result from the settings\n")
        def _update_z_stack_num_steps():
            num_steps = int(round(
                self.z_range_um.value.get() / self.z_step_um.value.get())) + 1
            if self.z_bidirectional.get():
                num_steps = 2 * num_steps - 1
            self.z_stack_num_steps.set(num_steps)
            text = '%03.i'%num_steps
            z_stack_num_steps_textbox.textbox.delete('1.0', 'end')
            z_stack_num_steps_textbox.textbox.insert('1.0', text)
            return None
        # loop over positions:
        self.loop_over_position_list = tk.BooleanVar()
        loop_over_position_list_button = tk.Checkbutton(
            frame,
            text='Loop over position list',
            variable=self.loop_over_position_list)
        loop_over_position_list_button.grid(
            row=8, column=0, columnspan=2, padx=10, pady=10, sticky='w')
        loop_over_position_list_tip = Hovertip(
            loop_over_position_list_button,
            "If checked, the 'Run acquire' button will loop over the XYZ\n" +
            "positions stored in the 'POSITION LIST'.\n" +
            "NOTE: it can take a significant amount of time to image \n" +
            "many positions so this should be taken into consideration \n" +
            "(especially for a time series).")
        # acquire number spinbox:
        self.acquire_number = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Acquire number',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=1,
            max_value=1e6,
            default_value=1,
            row=9,
            width=spinbox_width,
            columnspan=2)
        acquire_number_spinbox_tip = Hovertip(
            self.acquire_number,
            "How many acquisitions did you want when you press\n" +
            "the 'Run acquire' button?\n" +
            "NOTE: there is no immediate limit here, but data \n" +
            "accumulation can limit in practice.")
        # delay spinbox:
        self.delay_s = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Inter-acquire delay (s) >=',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=0,
            max_value=3600,
            default_value=0,
            row=10,
            width=spinbox_width,
            columnspan=2)
        delay_spinbox_tip = Hovertip(
            self.delay_s,
            "How long do you want to wait between acquisitions?\n" +
            "NOTE: the GUI will attempt to achieve the requested interval.\n" +
            "However, if the acquisition (which may include multiple \n" +
            "colors/positions) takes longer than the requested delay then\n" +
            "it will simply run as fast as it can.\n")        
        return None

    def init_settings_output(self):
        frame = tk.LabelFrame(self.root, text='SETTINGS OUTPUT', bd=6)
        frame.grid(row=9, column=5, rowspan=3, padx=5, pady=5, sticky='s')
        button_width, button_height = 25, 2
        spinbox_width = 20
        # frames per second textbox:
        self.frames_per_s = tk.DoubleVar()
        frames_per_s_textbox = tkcw.Textbox(
            frame,
            label='Frames per second',
            default_text='None',
            row=0,
            width=spinbox_width,
            height=1)
        def _update_frames_per_s():            
            text = '%0.3f'%self.frames_per_s.get()
            frames_per_s_textbox.textbox.delete('1.0', 'end')
            frames_per_s_textbox.textbox.insert('1.0', text)
            return None
        self.frames_per_s.trace_add(
            'write',
            lambda var, index, mode: _update_frames_per_s())
        frames_per_s_textbox_tip = Hovertip(
            frames_per_s_textbox,
            "Shows the 'Frames per second' (fps) based on the settings\n" +
            "that were last applied to the microscope.\n" +
            "NOTE: this is the frame rate for the acquisition (i.e. during\n" +
            "the analogue out 'play') and does reflect any delays or\n" +
            "latency between acquisitions.")
        # data memory textbox:
        self.data_bytes = tk.IntVar()
        self.data_buffer_exceeded = tk.BooleanVar()
        data_memory_textbox = tkcw.Textbox(
            frame,
            label='Data memory (GB)',
            default_text='None',
            row=1,
            width=spinbox_width,
            height=1)
        data_memory_textbox.textbox.tag_add('color', '1.0', 'end')
        def _update_data_memory():
            data_memory_gb = 1e-9 * self.data_bytes.get()
            max_memory_gb = 1e-9 * self.max_bytes_per_buffer
            memory_pct = 100 * data_memory_gb / max_memory_gb
            text = '%0.3f (%0.2f%% max)'%(data_memory_gb, memory_pct)
            data_memory_textbox.textbox.delete('1.0', 'end')
            bg = 'white'
            if self.data_buffer_exceeded.get(): bg = 'red'
            data_memory_textbox.textbox.tag_config('color', background=bg)
            data_memory_textbox.textbox.insert('1.0', text, 'color')
            return None
        self.data_bytes.trace_add(
            'write',
            lambda var, index, mode: _update_data_memory())
        data_memory_textbox_tip = Hovertip(
            data_memory_textbox,
            "Shows the 'data buffer memory' (GB) that the microscope\n" +
            "will need to run the settings that were last applied.\n" +
            "NOTE: this can be useful for monitoring resources and \n" +
            "avoiding memory limits.")
        # total memory textbox:
        self.total_bytes = tk.IntVar()
        self.total_bytes_exceeded = tk.BooleanVar()
        total_memory_textbox = tkcw.Textbox(
            frame,
            label='Total memory (GB)',
            default_text='None',
            row=2,
            width=spinbox_width,
            height=1)
        total_memory_textbox.textbox.tag_add('color', '1.0', 'end')
        def _update_total_memory():
            total_memory_gb = 1e-9 * self.total_bytes.get()
            max_memory_gb = 1e-9 * self.max_allocated_bytes
            memory_pct = 100 * total_memory_gb / max_memory_gb
            text = '%0.3f (%0.2f%% max)'%(total_memory_gb, memory_pct)
            total_memory_textbox.textbox.delete('1.0', 'end')
            bg = 'white'
            if self.total_bytes_exceeded.get(): bg = 'red'
            total_memory_textbox.textbox.tag_config('color', background=bg)
            total_memory_textbox.textbox.insert('1.0', text, 'color')
            return None
        self.total_bytes.trace_add(
            'write',
            lambda var, index, mode: _update_total_memory())
        total_memory_textbox_tip = Hovertip(
            total_memory_textbox,
            "Shows the 'total memory' (GB) that the microscope\n" +
            "will need to run the settings that were last applied.\n" +
            "NOTE: this can be useful for monitoring resources and \n" +
            "avoiding memory limits.")
        # total storage textbox:
        total_storage_textbox = tkcw.Textbox(
            frame,
            label='Total storage (GB)',
            default_text='None',
            row=3,
            width=spinbox_width,
            height=1)
        def _update_total_storage():
            positions = 1
            if self.loop_over_position_list.get():
                positions = max(len(self.XY_stage_position_list), 1)
            acquires = self.acquire_number.value.get()
            data_gb = 1e-9 * self.data_bytes.get()
            total_storage_gb = data_gb * positions * acquires
            if self.z_stack.get():
                total_storage_gb = (
                    total_storage_gb * self.z_stack_num_steps.get())
            text = '%0.3f'%total_storage_gb
            total_storage_textbox.textbox.delete('1.0', 'end')
            total_storage_textbox.textbox.insert('1.0', text)
            return None
        self.total_bytes.trace_add(
            'write',
            lambda var, index, mode: _update_total_storage())
        self.z_stack_num_steps.trace_add(
            'write',
            lambda var, index, mode: _update_total_storage())
        total_storage_textbox_tip = Hovertip(
            total_storage_textbox,
            "Shows the 'total storage' (GB) that the microscope will \n" +
            "need to save the data if 'Run acquire' is pressed (based \n" +
            "on the settings that were last applied).\n" +
            "NOTE: this can be useful for monitoring resources and \n" +
            "avoiding storage limits.")
        # min time textbox:
        self.buffer_time_s = tk.DoubleVar()
        min_time_textbox = tkcw.Textbox(
            frame,
            label='Minimum acquire time (s)',
            default_text='None',
            row=4,
            width=spinbox_width,
            height=1)
        def _update_min_time():
            positions = 1
            if self.loop_over_position_list.get():
                positions = max(len(self.XY_stage_position_list), 1)
            acquires = self.acquire_number.value.get()
            min_acquire_time_s = self.buffer_time_s.get() * positions
            min_total_time_s = min_acquire_time_s * acquires
            if self.z_stack.get():
                min_total_time_s = (
                    min_total_time_s * self.z_stack_num_steps.get())
            delay_s = self.delay_s.value.get()
            if delay_s > min_acquire_time_s:
                min_total_time_s = ( # start -> n-1 delays -> final acquire
                    delay_s * (acquires - 1) + min_acquire_time_s)
            text = '%0.6f (%0.0f min)'%(
                min_total_time_s, (min_total_time_s / 60))
            min_time_textbox.textbox.delete('1.0', 'end')
            min_time_textbox.textbox.insert('1.0', text)
            return None
        self.buffer_time_s.trace_add(
            'write',
            lambda var, index, mode: _update_min_time())
        self.z_stack_num_steps.trace_add(
            'write',
            lambda var, index, mode: _update_min_time())
        min_time_textbox_tip = Hovertip(
            min_time_textbox,
            "Shows the 'Minimum acquire time (s)' that the microscope will\n" +
            "need if 'Run acquire' is pressed (based on the settings that\n" +
            "were last applied).\n" +
            "NOTE: this value does not take into account the 'move time'\n" +
            "when using the 'Loop over position list' option (so the actual\n" +
            "time will be significantly more).")
        return None

    def init_position_list(self):
        frame = tk.LabelFrame(self.root, text='POSITION LIST', bd=6)
        frame.grid(row=1, column=6, rowspan=7, padx=5, pady=5, sticky='n')
        button_width, button_height = 25, 1
        spinbox_width = 20
        # set list defaults:
        self.focus_piezo_position_list = []
        self.XY_stage_position_list = []
        # load from folder:
        def _load_positions_from_folder():
            # get folder from user:
            folder_path = tk.filedialog.askdirectory(
                parent=self.root,
                initialdir=os.getcwd(),
                title='Please choose a previous "gui session" folder')
            # read files, parse into lists and update attributes:
            focus_piezo_file_path = (
                folder_path + '\\focus_piezo_position_list.txt')
            XY_stage_file_path = (
                folder_path + '\\XY_stage_position_list.txt')
            with open(focus_piezo_file_path, 'r') as file:
                focus_piezo_position_list = file.read().splitlines()
            with open(XY_stage_file_path, 'r') as file:
                XY_stage_position_list = file.read().splitlines()
            assert len(focus_piezo_position_list) == len(XY_stage_position_list)
            for i, element in enumerate(focus_piezo_position_list):
                focus_piezo_z_um = int(element.strip(','))
                focus_piezo_position_list[i] = focus_piezo_z_um
                self.focus_piezo_position_list.append(focus_piezo_z_um)
            for i, element in enumerate(XY_stage_position_list):
                XY_stage_position_mm = [
                    float(element.strip('[').strip(']').split(',')[0]),
                    float(element.strip('[').split(',')[1].strip(']').lstrip())]
                XY_stage_position_list[i] = XY_stage_position_mm
                self.XY_stage_position_list.append(XY_stage_position_mm)
            # append positions to files:
            with open(self.session_folder +
                      "focus_piezo_position_list.txt", "a") as file:
                for i in range(len(focus_piezo_position_list)):
                    file.write(str(focus_piezo_position_list[i]) + ',\n')
            with open(self.session_folder +
                      "XY_stage_position_list.txt", "a") as file:
                for i in range(len(XY_stage_position_list)):
                    file.write(str(XY_stage_position_list[i]) + ',\n')
            # update gui:
            self.total_positions.update_and_validate(
                len(XY_stage_position_list))
            return None
        load_from_folder_button = tk.Button(
            frame,
            text="Load from folder",
            command=_load_positions_from_folder,
            font=('Segoe UI', '10', 'underline'),
            width=button_width,
            height=button_height)
        load_from_folder_button.grid(row=0, column=0, padx=10, pady=10)
        load_from_folder_tip = Hovertip(
            load_from_folder_button,
            "Use the 'Load from folder' button to select a previous \n" +
            "'sols_gui_session' folder and load the associated position\n" +
            "list into the GUI.\n" +
            "NOTE: this will overwrite any existing position list")
        # delete all:
        def _delete_all_positions():
            # empty the lists:
            self.focus_piezo_position_list = []
            self.XY_stage_position_list = []
            # clear the files:
            with open(
                self.session_folder + "focus_piezo_position_list.txt", "w"):
                pass
            with open(
                self.session_folder + "XY_stage_position_list.txt", "w"):
                pass
            # update gui:
            self.total_positions.update_and_validate(0)
            self.current_position.update_and_validate(0)
            return None
        delete_all_positions_button = tk.Button(
            frame,
            text="Delete all positions",
            command=_delete_all_positions,
            width=button_width,
            height=button_height)
        delete_all_positions_button.grid(row=1, column=0, padx=10, pady=10)
        delete_all_positions_tip = Hovertip(
            delete_all_positions_button,
            "The 'Delete all positions' button clears the current position\n" +
            "list in the GUI and updates the associated .txt files in the\n" +
            "'sols_gui_session' folder.\n" +
            "NOTE: this operation cannot be reversed.")
        # delete current:
        def _delete_current_position():
            if self.total_positions.value.get() == 0:
                return
            i = self.current_position.value.get() - 1
            self.focus_piezo_position_list.pop(i)
            self.XY_stage_position_list.pop(i)
            # update files:
            with open(self.session_folder +
                      "focus_piezo_position_list.txt", "w") as file:
                for i in range(len(self.focus_piezo_position_list)):
                    file.write(str(self.focus_piezo_position_list[i]) + ',\n')
            with open(self.session_folder +
                      "XY_stage_position_list.txt", "w") as file:
                for i in range(len(self.XY_stage_position_list)):
                    file.write(str(self.XY_stage_position_list[i]) + ',\n')
            # update gui:
            self.total_positions.update_and_validate(
                len(self.XY_stage_position_list))
            self.current_position.update_and_validate(i)
            return None
        delete_current_position_button = tk.Button(
            frame,
            text="Delete current position",
            command=_delete_current_position,
            width=button_width,
            height=button_height)
        delete_current_position_button.grid(row=2, column=0, padx=10, pady=10)
        delete_current_position_tip = Hovertip(
            delete_current_position_button,
            "The 'Delete current position' button clears the current \n" +
            "position from the position list in the GUI and updates \n" +
            "the associated .txt files in the 'gui_session' folder.\n" +
            "NOTE: this operation cannot be reversed.")
        # total positions:
        self.total_positions = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Total positions',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=0,
            max_value=1e6,
            default_value=0,
            row=3,
            width=spinbox_width)
        self.total_positions.spinbox.config(state='disabled')
        total_positions_spinbox_tip = Hovertip(
            self.total_positions,
            "The 'Total positions' displays the total number of positions\n" +
            "currently stored in the position list (both in the GUI and the\n" +
            "associated .txt files in the 'gui_session' folder.\n")
        # utility function:
        def _update_position(how):
            current_position = self.current_position.value.get()
            total_positions  = self.total_positions.value.get()
            if total_positions == 0:
                return
            # check which direction:
            if how == 'start':
                p = 1
            if how == 'back':
                p = current_position - 1
                if p < 1:
                    p = 1
            if how == 'forward':
                p = current_position + 1
                if p > total_positions:
                    p = total_positions
            if how == 'end':
                p = total_positions
            # record status of scout mode and switch off:
            self.scout_mode_status.set(self.running_scout_mode.get())
            self.running_scout_mode.set(0) # avoids snap from focus piezo                
            # move:
            if not self.autofocus_enabled.get():
                self.focus_piezo_z_um.update_and_validate(
                    self.focus_piezo_position_list[p - 1])
            self._update_XY_stage_position(
                self.XY_stage_position_list[p - 1])
            # update gui and snap:
            self.current_position.update_and_validate(p)
            self._snap_and_display()
            # re-apply scout mode:
            self.running_scout_mode.set(self.scout_mode_status.get())
            return None
        # move to start:
        move_to_start_button = tk.Button(
            frame,
            text="Move to start",
            command=lambda d='start': _update_position(d),
            width=button_width,
            height=button_height)
        move_to_start_button.grid(row=4, column=0, padx=10, pady=10)
        move_to_start_button_tip = Hovertip(
            move_to_start_button,
            "The 'Move to start' button will move the 'FOCUS PIEZO' and\n" +
            "'XY STAGE' to the first position in the position list.\n" +
            "NOTE: this is only active in 'Scout mode' and if the position\n" +
            "is not already at the start of the position list.")
        # move back:
        move_back_button = tk.Button(
            frame,
            text="Move back (-1)",
            command=lambda d='back': _update_position(d),
            width=button_width,
            height=button_height)
        move_back_button.grid(row=5, column=0, padx=10, pady=10)
        move_back_button_tip = Hovertip(
            move_back_button,
            "The 'Move back (-1)' button will move the 'FOCUS PIEZO' and\n" +
            "'XY STAGE' to the previous (n - 1) position in the position\n" +
            "list.")
        # current position:
        self.current_position = tkcw.CheckboxSliderSpinbox(
            frame,
            label='Current position',
            checkbox_enabled=False,
            slider_enabled=False,
            min_value=0,
            max_value=1e6,
            default_value=0,
            row=6,
            width=spinbox_width)
        self.current_position.spinbox.config(state='disabled')
        current_position_spinbox_tip = Hovertip(
            self.current_position,
            "The 'Current position' displays the current position in the\n" +
            "position list based on the last update to the position list\n" +
            "or move request in the 'POSITION LIST' panel.\n" +
            "NOTE: is not aware of XY moves made elsewhere (e.g. with the\n" +
            "joystick or 'XY STAGE' panel). Use one of the 'move' buttons\n" +
            "to update if needed.")
        # go forwards:
        move_forward_button = tk.Button(
            frame,
            text="Move forward (+1)",
            command=lambda d='forward': _update_position(d),
            width=button_width,
            height=button_height)
        move_forward_button.grid(row=7, column=0, padx=10, pady=10)
        move_forward_button_tip = Hovertip(
            move_forward_button,
            "The 'Move forward (+1)' button will move the 'FOCUS PIEZO'\n" +
            "and 'XY STAGE' to the next (n + 1) position in the position\n" +
            "list.")
        # move to end:
        move_to_end_button = tk.Button(
            frame,
            text="Move to end",
            command=lambda d='end': _update_position(d),
            width=button_width,
            height=button_height)
        move_to_end_button.grid(row=8, column=0, padx=10, pady=10)
        move_to_end_button_tip = Hovertip(
            move_to_end_button,
            "The 'Move to end' button will move the 'FOCUS PIEZO' and\n" +
            "'XY STAGE' to the last position in the position list.")
        return None

    def _update_position_list(self):
        # update list:
        self.focus_piezo_position_list.append(self.focus_piezo_z_um.value.get())
        self.XY_stage_position_list.append([self.X_stage_position_mm,
                                            self.Y_stage_position_mm])
        # update gui:
        positions = len(self.XY_stage_position_list)
        self.total_positions.update_and_validate(positions)
        self.current_position.update_and_validate(positions)
        # write to file:
        with open(self.session_folder +
                  "focus_piezo_position_list.txt", "a") as file:
            file.write(str(self.focus_piezo_position_list[-1]) + ',\n')
        with open(self.session_folder +
                  "XY_stage_position_list.txt", "a") as file:
            file.write(str(self.XY_stage_position_list[-1]) + ',\n')
        return None

    def init_acquire(self):
        frame = tk.LabelFrame(
            self.root, text='ACQUIRE', font=('Segoe UI', '10', 'bold'), bd=6)
        frame.grid(row=7, column=6, rowspan=4, padx=5, pady=5, sticky='n')
        frame.bind('<Enter>', lambda event: frame.focus_set()) # force update
        button_width, button_height = 25, 2
        bold_width_adjust = -3
        spinbox_width = 20
        # snap:
        snap_button = tk.Button(
            frame,
            text="Snap",
            command=self._snap_and_display,
            font=('Segoe UI', '10', 'bold'),
            width=button_width + bold_width_adjust,
            height=button_height)
        snap_button.grid(row=0, column=0, padx=10, pady=10)
        snap_button_tip = Hovertip(
            snap_button,
            "The 'Snap' button will apply the lastest microscope\n" +
            "settings and acquire an image. This is useful for refreshing\n" +
            "the display.\n" +
            "NOTE: this does not save any data or position information.")
        # live mode:
        def _live_mode():
            if self.running_live_mode.get():
                self._set_running_mode('live_mode')
            else:
                self._set_running_mode('None')
            def _run_live_mode():
                if self.running_live_mode.get():
                    if not self.last_acquire_task.is_alive():
                        self._snap_and_display()
                    self.root.after(int(1e3/30), _run_live_mode) # 30 fps
                return None
            _run_live_mode()
            return None
        self.running_live_mode = tk.BooleanVar()
        live_mode_button = tk.Checkbutton(
            frame,
            text='Live mode (On/Off)',
            variable=self.running_live_mode,
            command=_live_mode,
            indicatoron=0,
            font=('Segoe UI', '10', 'italic'),
            width=button_width,
            height=button_height)
        live_mode_button.grid(row=1, column=0, padx=10, pady=10)
        live_mode_button_tip = Hovertip(
            live_mode_button,
            "The 'Live mode (On/Off)' button will enable/disable 'Live \n" +
            "mode'. 'Live mode' will continously apply the lastest \n" +
            "microscope settings and acquire an image.\n" +
            "NOTE: this continously exposes the sample to light which \n" +
            "may cause photobleaching/phototoxicity. To reduce this \n" +
            "effect use 'Scout mode'.") 
        # scout mode:
        def _scout_mode():
            self._set_running_mode('scout_mode')
            if self.running_scout_mode.get():
                self._snap_and_display()
            return None
        self.running_scout_mode = tk.BooleanVar()
        scout_mode_button = tk.Checkbutton(
            frame,
            text='Scout mode (On/Off)',
            variable=self.running_scout_mode,
            command=_scout_mode,
            indicatoron=0,
            font=('Segoe UI', '10', 'bold', 'italic'),
            fg='green',
            width=button_width + bold_width_adjust,
            height=button_height)
        scout_mode_button.grid(row=2, column=0, padx=10, pady=10)
        scout_mode_button_tip = Hovertip(
            scout_mode_button,
            "The 'Scout mode (On/Off)' button will enable/disable \n" +
            "'Scout mode'. 'Scout mode' will only acquire an image\n" +
            "if XYZ motion is detected. This helps to reduce \n" +
            "photobleaching/phototoxicity.")
        # save image and position:
        def _save_image_and_position():
            if self.images_per_buffer.value.get() != 1:
                self.images_per_buffer.update_and_validate(1)
            self._update_position_list()
            folder_name = self._get_folder_name() + '_snap'
            self.last_acquire_task.get_result() # don't accumulate acquires
            self.scope.acquire(filename='snap.tif',
                               folder_name=folder_name,
                               description=self.description_textbox.text)
            return None
        save_image_and_position_button = tk.Button(
            frame,
            text="Save image and position",
            command=_save_image_and_position,
            font=('Segoe UI', '10', 'bold'),
            fg='blue',
            width=button_width + bold_width_adjust,
            height=button_height)
        save_image_and_position_button.grid(row=3, column=0, padx=10, pady=10)
        save_image_and_position_tip = Hovertip(
            save_image_and_position_button,
            "The 'Save image and position' button will apply the latest\n" +
            "microscope settings, save an image and add the current\n" +
            "position to the position list.")
        # run acquire:
        def _acquire():
            print('\nAcquire -> started')
            self._set_running_mode('acquire')
            self.folder_name = self._get_folder_name() + '_acquire'
            self.delay_saved = False
            self.acquire_count = 0
            self.acquire_position = 0
            def _run_acquire():
                if not self.running_acquire.get(): # check for cancel
                    return None
                # don't launch all tasks: either wait 1 buffer time or delay:
                wait_ms = int(round(1e3 * self.scope.buffer_time_s))
                # check mode -> either single position or loop over positions:
                if not self.loop_over_position_list.get():
                    if not self.z_stack.get():
                        self.scope.acquire(
                            filename='%06i.tif'%self.acquire_count,
                            folder_name=self.folder_name,
                            description=self.description_textbox.text)
                    else:
                        self.scope.acquire_stack(
                            z_range_um=self.z_range_um.value.get(),
                            z_step_um=self.z_step_um.value.get(),
                            bidirectional=self.z_bidirectional.get(),
                            filename='%06i'%self.acquire_count,
                            folder_name=self.folder_name,
                            description=self.description_textbox.text)
                    self.acquire_count += 1
                    if self.delay_s.value.get() > self.scope.buffer_time_s:
                        wait_ms = int(round(1e3 * self.delay_s.value.get()))                    
                else:
                    if self.acquire_position == 0:
                        self.loop_t0_s = time.perf_counter()
                    if not self.autofocus_enabled.get():
                        self.focus_piezo_z_um.update_and_validate(
                            self.focus_piezo_position_list[
                                self.acquire_position])
                    self._update_XY_stage_position(
                        self.XY_stage_position_list[self.acquire_position])
                    self.current_position.update_and_validate(
                        self.acquire_position + 1)
                    if not self.z_stack.get():
                        self.scope.acquire(
                            filename='%06i_p%04i.tif'%(
                                self.acquire_count, self.acquire_position),
                            folder_name=self.folder_name,
                            description=self.description_textbox.text)
                    else:
                        self.scope.acquire_stack(
                            z_range_um=self.z_range_um.value.get(),
                            z_step_um=self.z_step_um.value.get(),
                            bidirectional=self.z_bidirectional.get(),
                            filename='%06i_p%04i'%(
                                self.acquire_count, self.acquire_position),
                            folder_name=self.folder_name,
                            description=self.description_textbox.text)
                    if self.acquire_position < (
                        self.total_positions.value.get() - 1):
                        self.acquire_position +=1
                    else:
                        self.acquire_position = 0
                        self.acquire_count += 1
                        loop_time_s = time.perf_counter() - self.loop_t0_s
                        if self.delay_s.value.get() > loop_time_s:
                            wait_ms = int(round(1e3 * (
                                self.delay_s.value.get() - loop_time_s)))
                # record gui delay:
                if (not self.delay_saved and os.path.exists(
                    self.folder_name)):
                    with open(self.folder_name + '\\'  "gui_delay_s.txt",
                              "w") as file:
                        file.write(self.folder_name + '\n')
                        file.write(
                            'gui_delay_s: %i'%self.delay_s.value.get() + '\n')
                        self.delay_saved = True
                # check acquire count before re-run:
                if self.acquire_count < self.acquire_number.value.get():
                    self.root.after(wait_ms, _run_acquire)
                else:
                    self.scope.finish_all_tasks()
                    self._set_running_mode('None')
                    print('Acquire -> finished\n')
                if self.z_stack.get() and self.running_scout_mode.get():
                    self._snap_and_display()
                return None
            _run_acquire()
            return None
        self.running_acquire = tk.BooleanVar()
        acquire_button = tk.Checkbutton(
            frame,
            text="Run acquire",
            variable=self.running_acquire,
            command=_acquire,
            indicatoron=0,
            font=('Segoe UI', '10', 'bold'),
            fg='red',
            width=button_width + bold_width_adjust,
            height=button_height)
        acquire_button.grid(row=4, column=0, padx=10, pady=10)
        acquire_button_tip = Hovertip(
            acquire_button,
            "The 'Run acquire' button will run a full acquisition and may\n" +
            "include: \n" +
            "- multiple colors (enable with the 'TRANSMITTED LIGHT' and\n" +
            "'LASER BOX' panels).\n" +
            "- multiple positions (populate the 'POSITION LIST' and enable\n" +
            "'Loop over position list').\n" +
            "- multiple fast images per position (set 'Images per\n" +
            "acquire' > 1).\n" +
            "- multiple iterations of the above (set 'Acquire number' > 1).\n" +
            "- a time delay between successive iterations of the above \n" +
            "(set 'Inter-acquire delay (s)' > the time per iteration)")
        return None

    def init_running_mode(self):
        # define mode variable and dictionary:
        self.running_mode = tk.StringVar()
        self.mode_to_variable = {'grid_preview': self.running_grid_preview,
                                 'tile_preview': self.running_tile_preview,
                                 'live_mode':    self.running_live_mode,
                                 'scout_mode':   self.running_scout_mode,
                                 'acquire':      self.running_acquire}
        self.scout_mode_status = tk.BooleanVar()
        # cancel running mode popup:
        self.cancel_running_mode_popup = tk.Toplevel()
        self.cancel_running_mode_popup.title('Cancel current process')
        x, y = self.root.winfo_x(), self.root.winfo_y() # center popup
        self.cancel_running_mode_popup.geometry("+%d+%d" % (x + 1200, y + 600))
        self.cancel_running_mode_popup.withdraw()
        # cancel button:
        def _cancel():
            print('\n *** Canceled -> ' + self.running_mode.get() + ' *** \n')
            self._set_running_mode('None')
            return None
        self.cancel_running_mode_button = tk.Button(
            self.cancel_running_mode_popup,
            font=('Segoe UI', '10', 'bold'),
            bg='red',
            command=_cancel,
            width=25,
            height=2)
        self.cancel_running_mode_button.grid(row=8, column=0, padx=10, pady=10)
        cancel_running_mode_tip = Hovertip(
            self.cancel_running_mode_button,
            "Cancel the current process.\n" +
            "NOTE: this is not immediate since some processes must finish\n" +
            "once launched.")
        return None

    def _set_running_mode(self, mode):
        if mode != 'None':
            # record status of scout mode:
            self.scout_mode_status.set(self.running_scout_mode.get())
            # turn everything off except current mode:
            for v in self.mode_to_variable.values():
                if v != self.mode_to_variable[mode]:
                    v.set(0)
        if mode in ('grid_preview', 'tile_preview', 'acquire'):
            # update cancel text:
            self.running_mode.set(mode) # string for '_cancel' print
            self.cancel_running_mode_button.config(text=('Cancel: ' + mode))
            # display cancel popup and grab set:
            self.cancel_running_mode_popup.deiconify()
            self.cancel_running_mode_popup.grab_set()
        if mode == 'None':
            # turn everything off:
            for v in self.mode_to_variable.values():
                v.set(0)
            # hide cancel popup and release set:
            self.cancel_running_mode_popup.withdraw()
            self.cancel_running_mode_popup.grab_release()
            # re-apply scout mode:
            self.running_scout_mode.set(self.scout_mode_status.get())
        return None

if __name__ == '__main__':
    gui_microscope = GuiMicroscope(init_microscope=True)
