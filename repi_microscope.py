# Imports from the python standard library:
import atexit
import os
import queue
import time
from datetime import datetime

# Third party imports, installable via pip:
import napari
import numpy as np
from tifffile import imread, imwrite

# Our code, one .py file per module, copy files to your local directory:
try:
    import asi_MS_2000_500_CP       # github.com/amsikking/asi_MS_2000_500_CP
    import lumencor_Spectra_X       # github.com/amsikking/lumencor_Spectra_X
    import concurrency_tools as ct  # github.com/AndrewGYork/tools
    import ni_PCIe_6738             # github.com/amsikking/ni_PCIe_6738
    import pco_panda42_bi           # github.com/amsikking/pco_panda42_bi
    import pi_E_709_1C1L            # github.com/amsikking/pi_E_709_1C1L
    import prior_PureFocus850       # github.com/amsikking/prior_PureFocus850
    import sutter_Lambda_10_3       # github.com/amsikking/sutter_Lambda_10_3
    from napari_in_subprocess import display    # github.com/AndrewGYork/tools
except Exception as e:
    print('repi_microscope.py -> One or more imports failed')
    print('repi_microscope.py -> error =',e)

# Repi optical configuration (edit as needed):
camera_px_um = 6.5
dichroic_mirror_options = {'FF409/493/573/652-Di02' :0,
                           '(unused)'               :1}
emission_filter_options = {'Shutter'                :0,
                           'Open'                   :1,
                           'FF01-432/36'            :2,
                           'FF01-515/30'            :3,
                           'FF01-595/31'            :4,
                           'FF01-698/70'            :5,
                           'FF01-432/515/595/730'   :6,
                           '(unused)'               :7,
                           '(unused)'               :8,
                           '(unused)'               :9}
objective_options = {'name'  :('Nikon 10x0.45 air',
                               'Nikon 20x0.75 air',
                               'Nikon 40x0.95 air'),
                     # magnification:
                     'mag':(10, 20, 40),
                     # min working distance spec:
                     'WD_um' :(4000, 1000, 170)}

class Microscope:
    def __init__(self,
                 max_allocated_bytes,   # Limit of available RAM for machine
                 ao_rate,               # slow ~1e3, medium ~1e4, fast ~1e5
                 name='REPI v1.0',
                 verbose=True,
                 print_warnings=True):
        self.max_allocated_bytes = max_allocated_bytes
        self.name = name
        self.verbose = verbose
        self.print_warnings = print_warnings
        if self.verbose: print("%s: opening..."%self.name)
        self.unfinished_tasks = queue.Queue()
        # init hardware/software:
        slow_fw_init = ct.ResultThread(
            target=self._init_filter_wheel).start() #~5.3s
        slow_camera_init = ct.ResultThread(
            target=self._init_camera).start()       #~3.6s
        slow_XYZ_stage_init = ct.ResultThread(
            target=self._init_XYZ_stage).start()    #~0.65s
        slow_focus_init = ct.ResultThread(
            target=self._init_focus_piezo).start()  #~0.6s
        slow_leds_init = ct.ResultThread(
            target=self._init_leds).start()         #~0.25s
        slow_autofocus_init = ct.ResultThread(
            target=self._init_autofocus).start()    #~0.015s
        self._init_display()                        #~1.3s
        self._init_ao(ao_rate)                      #~0.2s
        slow_autofocus_init.get_result()
        slow_leds_init.get_result()
        slow_focus_init.get_result()
        slow_XYZ_stage_init.get_result()
        slow_camera_init.get_result()
        slow_fw_init.get_result()
        # configure autofocus: (focus_piezo and autofocus initialized)
        self.autofocus.set_piezo_range_um(400)
        self.autofocus.set_digipot_mode('Offset') # set for user convenience
        self.focus_piezo.set_analog_control_limits( # 0-10V is 400um
            v_min=0,
            v_max=10,
            z_min_ai=0,
            z_max_ai=self.autofocus.piezo_range_um)
        self.autofocus_offset_lens = (
            self.autofocus._get_offset_lens_position())
        self.autofocus_sample_flag = self.autofocus.get_sample_flag()
        self.autofocus_focus_flag  = self.autofocus.get_focus_flag()
        # set defaults:
        # -> apply_settings args
        self.timestamp_mode = "binary+ASCII"
        self.camera._set_timestamp_mode(self.timestamp_mode) # default on
        self.autofocus_enabled = False
        self.focus_piezo_z_um = self.focus_piezo.z
        self.XY_stage_position_mm = (1e-3 * self.XYZ_stage.position_um[0],
                                     1e-3 * self.XYZ_stage.position_um[1])
        self.Z_stage_position_mm = 1e-3 * self.XYZ_stage.position_um[2]
        self.camera_preframes = 1 # ditch some noisy frames before recording?
        self.max_bytes_per_buffer = (2**31) # legal tiff
        self.max_data_buffers = 3 # camera, display, filesave
        # -> additional
        self.num_active_data_buffers = 0
        self._settings_applied = False
        if self.verbose: print("\n%s: -> open and ready."%self.name)

    def _init_filter_wheel(self):
        if self.verbose: print("\n%s: opening filter wheel..."%self.name)
        self.filter_wheel = sutter_Lambda_10_3.Controller(
            which_port='COM10', verbose=False)
        if self.verbose: print("\n%s: -> filter wheel open."%self.name)
        atexit.register(self.filter_wheel.close)

    def _init_camera(self):
        if self.verbose: print("\n%s: opening camera..."%self.name)
        self.camera = ct.ObjectInSubprocess(
            pco_panda42_bi.Camera, verbose=False, close_method_name='close')
        if self.verbose: print("\n%s: -> camera open."%self.name)

    def _init_XYZ_stage(self):
        if self.verbose: print("\n%s: opening XYZ stage..."%self.name)
        self.XYZ_stage = asi_MS_2000_500_CP.Controller(
            which_port='COM3',
            axes=('X', 'Y', 'Z'),
            lead_screws=('S','S','F'),
            axes_min_mm=(-60,-40,-7), # recommended to check and use!
            axes_max_mm=( 60, 40, 0), # recommended to check and use!
            verbose=False)
        self.XYZ_stage.set_pwm_state('external') # setup 'TL_LED'
        if self.verbose: print("\n%s: -> XYZ stage open."%self.name)
        atexit.register(self.XYZ_stage.close)

    def _init_focus_piezo(self):
        if self.verbose: print("\n%s: opening focus piezo..."%self.name)
        self.focus_piezo = pi_E_709_1C1L.Controller(
            which_port='COM7', z_min_um=0, z_max_um=400, verbose=False)
        if self.verbose: print("\n%s: -> focus piezo open."%self.name)
        atexit.register(self.focus_piezo.close)

    def _init_leds(self):
        if self.verbose: print("\n%s: opening leds..."%self.name)
        self.led_names = (
            '395/25', '440/20', '470/24', '510/25', '550/15', '640/30')
        self.led_box = lumencor_Spectra_X.Controller(
            which_port='COM4', led_names=self.led_names, verbose=False)
        if self.verbose: print("\n%s: -> leds open."%self.name)
        atexit.register(self.led_box.close)

    def _init_autofocus(self):
        if self.verbose: print("\n%s: opening autofocus..."%self.name)
        self.autofocus = prior_PureFocus850.Controller(
            which_port='COM9', verbose=False)
        if self.verbose: print("\n%s: -> autofocus open."%self.name)
        atexit.register(self.autofocus.close)

    def _init_display(self):
        if self.verbose: print("\n%s: opening display..."%self.name)
        self.display = display(display_type=_CustomNapariDisplay)
        if self.verbose: print("\n%s: -> display open."%self.name)

    def _init_ao(self, ao_rate):
        self.illumination_sources = tuple( # controlled by ao
            ['TL_LED'] + [led for led in self.led_names])
        self.names_to_voltage_channels = {
            'camera_TTL': 0,
            'TL_LED_TTL': 1,
            '395/25_TTL': 2,
            '440/20_TTL': 3,
            '470/24_TTL': 4,
            '510/25_TTL': 5,
            '550/15_TTL': 6,
            '640/30_TTL': 7,
            }
        if self.verbose: print("\n%s: opening ao card..."%self.name)
        self.ao = ni_PCIe_6738.DAQ(
            num_channels=8, rate=ao_rate, verbose=False)
        if self.verbose: print("\n%s: -> ao card open."%self.name)
        atexit.register(self.ao.close)

    def _check_memory(self):        
        # Data:
        self.images = self.images_per_buffer * len(self.channels_per_image)
        self.bytes_per_data_buffer = (
            2 * self.images * self.height_px * self.width_px)
        self.data_buffer_exceeded = False
        if self.bytes_per_data_buffer > self.max_bytes_per_buffer:
            self.data_buffer_exceeded = True
            if self.print_warnings:
                print("\n%s: ***WARNING***: settings rejected"%self.name)
                print("%s: -> data_buffer_exceeded"%self.name)
                print("%s: -> reduce settings"%self.name +
                      " or increase 'max_bytes_per_buffer'")
        # Total:
        self.total_bytes = self.bytes_per_data_buffer * self.max_data_buffers
        self.total_bytes_exceeded = False
        if self.total_bytes > self.max_allocated_bytes:
            self.total_bytes_exceeded = True
            if self.print_warnings:
                print("\n%s: ***WARNING***: settings rejected"%self.name)
                print("%s: -> total_bytes_exceeded"%self.name)
                print("%s: -> reduce settings"%self.name +
                      " or increase 'max_allocated_bytes'")
        return None

    def _calculate_voltages(self):
        n2c = self.names_to_voltage_channels # nickname
        # Timing information:
        exposure_px = self.ao.s2p(1e-6 * self.camera.exposure_us)
        rolling_px =  self.ao.s2p(1e-6 * self.camera.rolling_time_us)
        jitter_px = max(self.ao.s2p(1000e-6), 1)
        period_px = max(exposure_px, rolling_px) + jitter_px
        # Calculate voltages:
        voltages = []
        # Add preframes (if any):
        for frames in range(self.camera_preframes):
            v = np.zeros((period_px, self.ao.num_channels), 'float64')
            v[:rolling_px, n2c['camera_TTL']] = 5 # falling edge-> light on!
            voltages.append(v)
        for images in range(self.images_per_buffer):
            for channel, power in zip(self.channels_per_image,
                                      self.power_per_channel):
                v = np.zeros((period_px, self.ao.num_channels), 'float64')
                v[:rolling_px, n2c['camera_TTL']] = 5 # falling edge-> light on!
                if channel == 'TL_LED':
                    ill_px = period_px - jitter_px - rolling_px
                    # need enough ttl on/off time for LED to respond:
                    ttl_px = self.ao.s2p(1e-3) # it needs about 1ms!
                    on_px = int((power / 100) * (ill_px - 2 * ttl_px))
                    if on_px >= ttl_px: # only on if there's enough time
                        v[rolling_px:rolling_px + ttl_px,
                          n2c[channel + '_TTL']] = 5 # turn on
                        v[rolling_px + ttl_px + on_px:
                          rolling_px + ttl_px + on_px + ttl_px,
                          n2c[channel + '_TTL']] = 5 # turn off
                if channel != 'TL_LED': # SpectraX
                    v[rolling_px:period_px - jitter_px,
                      n2c[channel + '_TTL']] = 5
                voltages.append(v)
        voltages = np.concatenate(voltages, axis=0)
        # Timing attributes:
        self.buffer_time_s = self.ao.p2s(voltages.shape[0])
        self.frames_per_s = self.images_per_buffer / self.buffer_time_s
        return voltages

    def _plot_voltages(self):
        import matplotlib.pyplot as plt
        # Reverse lookup table; channel numbers to names:
        c2n = {v:k for k, v in self.names_to_voltage_channels.items()}
        for c in range(self.voltages.shape[1]):
            plt.plot(self.voltages[:, c], label=c2n.get(c, f'ao-{c}'))
        plt.legend(loc='upper right')
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [self.ao.p2s(l) for l in xlocs])
        plt.ylabel('Volts')
        plt.xlabel('Seconds')
        plt.show()

    def _prepare_to_save(self, filename, folder_name, description, display):
        def make_folders(folder_name):
            os.makedirs(folder_name)
            os.makedirs(folder_name + '\\data')
            os.makedirs(folder_name + '\\metadata')
        assert type(filename) is str
        if folder_name is None:
            folder_index = 0
            dt = datetime.strftime(datetime.now(),'%Y-%m-%d_%H-%M-%S')
            folder_name = dt + '_%03i_repi'%folder_index
            while os.path.exists(folder_name): # check overwriting
                folder_index +=1
                folder_name = dt + '_%03i_repi'%folder_index
            make_folders(folder_name)
        else:
            if not os.path.exists(folder_name): make_folders(folder_name)
        data_path =     folder_name + '\\data\\'     + filename
        metadata_path = folder_name + '\\metadata\\' + filename
        # save metadata:
        to_save = {
            # date and time:
            'Date':datetime.strftime(datetime.now(),'%Y-%m-%d'),
            'Time':datetime.strftime(datetime.now(),'%H:%M:%S'),
            # args from 'acquire':
            'filename':filename,
            'folder_name':folder_name,
            'description':description,
            'display':display,
            # attributes from 'apply_settings':
            # -> args
            'channels_per_image':tuple(self.channels_per_image),
            'power_per_channel':tuple(self.power_per_channel),
            'dichroic_mirror':self.dichroic_mirror,
            'emission_filter':self.emission_filter,
            'illumination_time_us':self.illumination_time_us,
            'height_px':self.height_px,
            'width_px':self.width_px,
            'timestamp_mode':self.timestamp_mode,
            'images_per_buffer':self.images_per_buffer,
            'autofocus_enabled':self.autofocus_enabled,
            'focus_piezo_z_um':self.focus_piezo_z_um,
            'XY_stage_position_mm':self.XY_stage_position_mm,
            'camera_preframes':self.camera_preframes,
            'max_bytes_per_buffer':self.max_bytes_per_buffer,
            'max_data_buffers':self.max_data_buffers,
            # -> calculated
            'buffer_time_s':self.buffer_time_s,
            'frames_per_s':self.frames_per_s,
            # -> additional
            'autofocus_offset_lens':self.autofocus_offset_lens,
            'autofocus_sample_flag':self.autofocus_sample_flag,
            'autofocus_focus_flag':self.autofocus_focus_flag,
            'Z_stage_position_mm':self.Z_stage_position_mm,
            # optical configuration:
            'objective_name':self.objective_name,
            'objective_mag':self.objective_mag,
            'objective_WD_um':self.objective_WD_um,
            'camera_px_um':camera_px_um,
            'sample_px_um':self.sample_px_um,
            }
        with open(os.path.splitext(metadata_path)[0] + '.txt', 'w') as file:
            for k, v in to_save.items():
                file.write(k + ': ' + str(v) + '\n')
        return data_path

    def _get_data_buffer(self, shape, dtype):
        while self.num_active_data_buffers >= self.max_data_buffers:
            time.sleep(1e-3) # 1.7ms min
        # Note: this does not actually allocate the memory. Allocation happens
        # during the first 'write' process inside camera.record_to_memory
        data_buffer = ct.SharedNDArray(shape, dtype)
        self.num_active_data_buffers += 1
        return data_buffer

    def _release_data_buffer(self, shared_numpy_array):
        assert isinstance(shared_numpy_array, ct.SharedNDArray)
        self.num_active_data_buffers -= 1

    def apply_settings( # Must call before .acquire()
        self,
        objective_name=None,        # String
        channels_per_image=None,    # Tuple of strings
        power_per_channel=None,     # Tuple of floats
        dichroic_mirror=None,       # String
        emission_filter=None,       # String
        illumination_time_us=None,  # Float
        height_px=None,             # Int
        width_px=None,              # Int
        timestamp_mode=None,        # "off" or "binary" or "binary+ASCII"
        images_per_buffer=None,     # Int
        autofocus_enabled=None,     # Bool
        focus_piezo_z_um=None,      # (Float, "relative" or "absolute")
        XY_stage_position_mm=None,  # (Float, Float, "relative" or "absolute")
        camera_preframes=None,      # Int
        max_bytes_per_buffer=None,  # Int
        max_data_buffers=None,      # Int
        ):
        args = locals()
        args.pop('self')
        def settings_task(custody):
            custody.switch_from(None, to=self.camera) # Safe to change settings
            self._settings_applied = False # In case the thread crashes
            # Attributes must be set previously or currently:
            for k, v in args.items(): 
                if v is not None:
                    setattr(self, k, v) # A lot like self.x = x
                assert hasattr(self, k), (
                    "%s: attribute %s must be set at least once"%(self.name, k))
            if objective_name is not None:
                assert objective_name in objective_options['name']
                i = objective_options['name'].index(objective_name)
                self.autofocus.set_current_objective(i + 1)
                self.objective_mag  = objective_options['mag'][i]
                self.objective_WD_um = objective_options['WD_um'][i]
                self.sample_px_um = camera_px_um / self.objective_mag
            if dichroic_mirror is not None:
                assert dichroic_mirror in dichroic_mirror_options.keys()
            if height_px is not None or width_px is not None: # legalize first
                h_px, w_px = height_px, width_px
                if height_px is None: h_px = self.height_px
                if width_px is None:  w_px = self.width_px
                self.height_px, self.width_px, self.roi_px = ( 
                    pco_panda42_bi.legalize_image_size(
                        h_px, w_px, verbose=False))
            self._check_memory()
            if self.data_buffer_exceeded or self.total_bytes_exceeded:
                custody.switch_from(self.camera, to=None)
                return None
            # Send hardware commands, slowest to fastest:
            if XY_stage_position_mm is not None:
                assert XY_stage_position_mm[2] in ('relative', 'absolute')
                x, y = (1e3 * XY_stage_position_mm[0],
                        1e3 * XY_stage_position_mm[1])
                if XY_stage_position_mm[2] == 'relative':
                    self.XYZ_stage.move_um((x, y, None), block=False)
                if XY_stage_position_mm[2] == 'absolute':
                    self.XYZ_stage.move_um(
                        (x, y, None), relative=False, block=False)
            else: # must update XYZ stage attributes if joystick was used
                update_XYZ_stage_position_thread = ct.ResultThread(
                    target=self.XYZ_stage._get_position).start()
            if emission_filter is not None:
                self.filter_wheel.move(
                    emission_filter_options[emission_filter], block=False)
            if autofocus_enabled is not None:
                assert isinstance(autofocus_enabled, bool)
                if autofocus_enabled:
                    sample_flags = []
                    for flags in range(3):
                        sample_flags.append(self.autofocus.get_sample_flag())
                    if any(sample_flags): # sample detected?
                        self.autofocus.set_servo_enable(False)
                        self.autofocus.set_piezo_voltage( # ~zero motion volts
                            self.focus_piezo.get_voltage_for_move_um(0))
                        self.focus_piezo.set_analog_control_enable(True)
                        self.autofocus.set_servo_enable(True)
                    else: # no sample detected, don't enable autofocus servo
                        self.autofocus_enabled = False
                        if self.print_warnings:
                            print("\n%s: ***WARNING***: "%self.name +
                                  "autofocus_sample_flag=FALSE")
                            print("\n%s: ***WARNING***: "%self.name +
                                  "autofocus_enabled=FALSE")
                else:
                    self.focus_piezo.set_analog_control_enable(False)
                    self.focus_piezo_z_um = self.focus_piezo.z # update attr
                    self.autofocus.set_servo_enable(False)                
            if focus_piezo_z_um is not None:
                if not self.autofocus_enabled:
                    assert focus_piezo_z_um[1] in ('relative', 'absolute')
                    z = focus_piezo_z_um[0]
                    if focus_piezo_z_um[1] == 'relative':
                        self.focus_piezo.move_um(z, block=False)
                    if focus_piezo_z_um[1] == 'absolute':
                        self.focus_piezo.move_um(z, relative=False, block=False)
                else:
                    if focus_piezo_z_um != (0,'relative'):
                        raise Exception(
                            'cannot move focus piezo with autofocus enabled')
            if (height_px is not None or
                width_px is not None or
                illumination_time_us is not None):
                self.camera._disarm()
                self.camera._set_roi(self.roi_px) # height_px updated first
                self.camera._set_exposure_time_us(int(
                    self.illumination_time_us + self.camera.rolling_time_us))
                self.camera._arm(self.camera._num_buffers)
            if timestamp_mode is not None:
                self.camera._set_timestamp_mode(timestamp_mode)
            check_write_voltages_thread = False
            if (channels_per_image is not None or
                power_per_channel is not None or
                height_px is not None or
                illumination_time_us is not None or
                images_per_buffer is not None or
                camera_preframes is not None):
                for channel in self.channels_per_image:
                    assert channel in self.illumination_sources
                assert len(self.power_per_channel) == (
                    len(self.channels_per_image))
                for i, p in enumerate(self.power_per_channel):
                    assert 0 <= p <= 100
                    if self.channels_per_image[i] != 'TL_LED': # SpectraX
                        self.led_box.set_power(p, self.channels_per_image[i])
                assert type(self.images_per_buffer) is int
                assert self.images_per_buffer > 0
                assert type(self.camera_preframes) is int
                self.camera.num_images = ( # update attribute
                    self.images + self.camera_preframes)
                self.voltages = self._calculate_voltages()
                write_voltages_thread = ct.ResultThread(
                    target=self.ao._write_voltages,
                    args=(self.voltages,)).start()
                check_write_voltages_thread = True
            # Finalize hardware commands, fastest to slowest:
            if focus_piezo_z_um is not None:
                self.focus_piezo._finish_moving()
                self.focus_piezo_z_um = self.focus_piezo.z
            if emission_filter is not None:
                self.filter_wheel._finish_moving()
            if XY_stage_position_mm is not None:
                self.XYZ_stage._finish_moving()
            else:
                update_XYZ_stage_position_thread.get_result()
            self.XY_stage_position_mm = (1e-3 * self.XYZ_stage.position_um[0],
                                         1e-3 * self.XYZ_stage.position_um[1])
            self.Z_stage_position_mm = 1e-3 * self.XYZ_stage.position_um[2]
            if check_write_voltages_thread:
                write_voltages_thread.get_result()
            self._settings_applied = True
            custody.switch_from(self.camera, to=None) # Release camera
        settings_thread = ct.CustodyThread(
            target=settings_task, first_resource=self.camera).start()
        self.unfinished_tasks.put(settings_thread)
        return settings_thread

    def acquire(self,               # 'tcyx' format
                filename=None,      # None = no save, same string = overwrite
                folder_name=None,   # None = new folder, same string = re-use
                description=None,   # Optional metadata description
                display=True):      # Optional turn off
        def acquire_task(custody):
            custody.switch_from(None, to=self.camera) # get camera
            if not self._settings_applied:
                if self.print_warnings:
                    print("\n%s: ***WARNING***: settings not applied"%self.name)
                    print("%s: -> please apply legal settings"%self.name)
                    print("%s: (all arguments must be specified at least once)")
                custody.switch_from(self.camera, to=None)
                return
            if self.autofocus_enabled: # update attributes:
                self.focus_piezo_z_um = self.focus_piezo.get_position(
                    verbose=False)
                self.autofocus_offset_lens = (
                    self.autofocus._get_offset_lens_position())
                self.autofocus_sample_flag = self.autofocus.get_sample_flag()
                self.autofocus_focus_flag  = self.autofocus.get_focus_flag()
                if self.print_warnings:
                    if not self.autofocus_sample_flag:
                        print("\n%s: ***WARNING***: "%self.name +
                              "self.autofocus_sample_flag=FALSE")
                    if not self.autofocus_focus_flag:
                        print("\n%s: ***WARNING***: "%self.name +
                              "autofocus_focus_flag=FALSE")
            # must update XYZ stage positions in case joystick/Z drive was used
            # no thread (blocking) so metatdata in _prepare_to_save is current
            x_um, y_um, z_um = self.XYZ_stage._get_position()
            self.XY_stage_position_mm = (1e-3 * x_um, 1e-3 * y_um)
            self.Z_stage_position_mm = 1e-3 * z_um
            if filename is not None:
                prepare_to_save_thread = ct.ResultThread(
                    target=self._prepare_to_save,
                    args=(filename, folder_name, description, display)).start()
            # We have custody of the camera so attribute access is safe:
            im   = self.images_per_buffer
            ch   = len(self.channels_per_image)
            h_px = self.height_px
            w_px = self.width_px
            ti   = self.images + self.camera_preframes
            data_buffer = self._get_data_buffer((ti, h_px, w_px), 'uint16')
            # camera.record_to_memory() blocks, so we use a thread:
            camera_thread = ct.ResultThread(
                target=self.camera.record_to_memory,
                kwargs={'allocated_memory': data_buffer,
                        'software_trigger': False},).start()
            # Race condition: the camera starts with (typically 16) single
            # frame buffers, which are filled by triggers from
            # ao.play_voltages(). The camera_thread empties them, hopefully
            # fast enough that we never run out. So far, the camera_thread
            # seems to both start on time, and keep up reliably once it starts,
            # but this could be fragile. The camera thread (effectively)
            # acquires shared memory as it writes to the allocated buffer.
            # On this machine the memory acquisition is faster than the camera
            # (~4GB/s vs ~1GB/s) but this could also be fragile if another
            # process interferes.
            self.ao.play_voltages(block=False)
            camera_thread.get_result()
            # Acquisition is 3D, but display and filesaving are 4D:
            data_buffer = data_buffer[ # ditch preframes
                self.camera_preframes:, :, :].reshape(im, ch, h_px, w_px)
            if display:
                custody.switch_from(self.camera, to=self.display)
                if self.timestamp_mode == "binary+ASCII":
                    self.display.show_image(data_buffer[:,:,8:,:])
                else:
                    self.display.show_image(data_buffer)
                custody.switch_from(self.display, to=None)
            else:
                custody.switch_from(self.camera, to=None)
            if filename is not None:
                data_path = prepare_to_save_thread.get_result()
                if self.verbose:
                    print("%s: saving '%s'"%(self.name, data_path))
                # TODO: consider puting FileSaving in a SubProcess
                imwrite(data_path, data_buffer[:,np.newaxis,:,:,:], imagej=True)
                if self.verbose:
                    print("%s: done saving."%self.name)
            self._release_data_buffer(data_buffer)
            del data_buffer
        acquire_thread = ct.CustodyThread(
            target=acquire_task, first_resource=self.camera).start()
        self.unfinished_tasks.put(acquire_thread)
        return acquire_thread

    def acquire_stack(self,                 # get z stack with the focus piezo
                      z_range_um,           # what z range in um
                      z_step_um,            # what z step size in um
                      bidirectional=False,  # True = -z_range_um to +z_range_um
                      filename=None,        # None = no save, same = overwrite
                      folder_name=None,     # None = new folder, same = re-use
                      description=None,     # Optional metadata description
                      display=True):        # Optional turn off
        assert z_step_um <= 99.9, (
            'z_step_um (%s) rejected -> max 99.9um allowed'%z_step_um)
        assert (z_step_um * 10) % 1 == 0, (
            'z_step_um (%s) rejected -> max 1 decimal place allowed'%z_step_um)
        # get initial state:
        z_zero_um = self.focus_piezo_z_um
        re_enable_autofocus = False
        if self.autofocus_enabled:
            self.apply_settings(autofocus_enabled=False)
            re_enable_autofocus = True
        # work out steps:
        num_steps = int(round(z_range_um / z_step_um)) + 1
        if bidirectional:
            self.apply_settings(focus_piezo_z_um=(-z_range_um,'relative'))
            num_steps = 2 * num_steps - 1
        assert num_steps <= 999, (
            'num_steps (%s) rejected '%num_steps +
            '(reduce z_range_um or increase z_step_um)')
        # move piezo and acquire:
        for step in range(num_steps):
            self.apply_settings(focus_piezo_z_um=(z_step_um,'relative'))
            self.acquire(
                filename=filename + '_%04.1fum_%03i.tif'%(z_step_um, step),
                folder_name=folder_name,
                description=description,
                display=display,
                )
        # return to initial state:
        self.apply_settings(focus_piezo_z_um=(z_zero_um,'absolute'))
        if re_enable_autofocus:
            self.apply_settings(autofocus_enabled=True)
        return None

    def finish_all_tasks(self):
        collected_tasks = []
        while True:
            try:
                th = self.unfinished_tasks.get_nowait()
            except queue.Empty:
                break
            th.get_result()
            collected_tasks.append(th)
        return collected_tasks

    def close(self):
        if self.verbose: print("%s: closing..."%self.name)
        self.finish_all_tasks()
        self.filter_wheel.close()
        self.camera.close()
        self.XYZ_stage.close()
        self.focus_piezo.close()
        self.led_box.close()
        self.autofocus.close()
        self.display.close()
        self.ao.close()
        if self.verbose: print("%s: done closing."%self.name)

class _CustomNapariDisplay:
    def __init__(self, auto_contrast=False):
        self.auto_contrast = auto_contrast
        self.viewer = napari.Viewer()

    def _legalize_slider(self, image):
        for ax in range(len(image.shape) - 2): # slider axes other than X, Y
            # if the current viewer slider steps > corresponding image shape:
            if self.viewer.dims.nsteps[ax] > image.shape[ax]:
                # set the slider position to the max legal value:
                self.viewer.dims.set_point(ax, image.shape[ax] - 1)

    def _reset_contrast(self, image): # 4D image min to max
        for layer in self.viewer.layers: # image, grid, tile
            layer.contrast_limits = (image.min(), image.max())

    def show_image(self, image):
        self._legalize_slider(image)
        if self.auto_contrast:
            self._reset_contrast(image)
        if not hasattr(self, 'image'):
            self.image = self.viewer.add_image(image)
        else:
            self.image.data = image

    def show_grid_image(self, grid_image):
        if not hasattr(self, 'grid_image'):
            self.grid_image = self.viewer.add_image(grid_image)
        else:
            self.grid_image.data = grid_image

    def show_tile_image(self, tile_image):
        if not hasattr(self, 'tile_image'):
            self.tile_image = self.viewer.add_image(tile_image)
        else:
            self.tile_image.data = tile_image

    def close(self):
        self.viewer.close()

if __name__ == '__main__':
    t0 = time.perf_counter()

    # Create scope object:
    scope = Microscope(max_allocated_bytes=10e9, ao_rate=1e5)
    scope.apply_settings(       # Mandatory call
        objective_name='Nikon 20x0.75 air',
        channels_per_image=("TL_LED",),
        power_per_channel=(15,),
        dichroic_mirror='FF409/493/573/652-Di02',
        emission_filter='Open',
        illumination_time_us=10000,
        height_px=2048,
        width_px=2048,
        images_per_buffer=1,
##        autofocus_enabled=True, # optional test
        focus_piezo_z_um=(0,'relative'),
        XY_stage_position_mm=(0,0,'relative'),
        ).get_result()

    # Acquire:
    folder_label = 'repi_test_data'
    dt = datetime.strftime(datetime.now(),'%Y-%m-%d_%H-%M-%S_000_')
    folder_name = dt + folder_label
    for i in range(3):
        scope.acquire(
            filename='%06i.tif'%i,
            folder_name=folder_name,
            description='something...',
            display=True,
            )

    # Acquire stack:
    folder_label = 'repi_test_stack'
    dt = datetime.strftime(datetime.now(),'%Y-%m-%d_%H-%M-%S_000_')
    folder_name = dt + folder_label
    scope.acquire_stack(
        z_range_um=10,
        z_step_um=1,
        bidirectional=False,
        filename='stack',
        folder_name=folder_name,
        description='something...',
        display=True,
        )
    scope.close()

    t1 = time.perf_counter()
    print('time_s', t1 - t0) # ~ 6.3s
