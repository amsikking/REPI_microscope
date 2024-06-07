import time
import os
import numpy as np
from datetime import datetime
from tifffile import imread, imwrite

import repi_microscope as repi

if __name__ == '__main__': # required block for repi_microscope
    # Create scope:
    scope = repi.Microscope(max_allocated_bytes=10e9, ao_rate=1e5)

    # Apply settings at least once: (required)
    scope.apply_settings(
        objective_name='Nikon 20x0.75 air',
        # ('TL_LED','395/25','440/20','470/24','510/25','550/15','640/30')
        channels_per_image=("TL_LED",),
        # match channels 0-100% i.e. (0,5,0,20,0,30,100)
        power_per_channel=(10,),
        dichroic_mirror='ZT405/488/561/640rpc',
        emission_filter='Shutter',  # reset later, options are:
        # 'Shutter', 'Open'
        # '(unused)', '(unused)', '(unused)', '(unused)'
        # '(unused)', '(unused)', '(unused)', '(unused)'
        illumination_time_us=1*1e3, # reset later
        height_px=2048,                         # 10 -> 2048 (typical range)
        width_px=2048,                          # 64 -> 2048 (typical range)
        images_per_buffer=1,                    # usually 1, can be more...
##        autofocus_enabled=True,                # set 'True' for autofocus
        focus_piezo_z_um=(0,'relative'),        # = don't move
        XY_stage_position_mm=(0,0,'relative'),  # = don't move
        ).get_result()

    # Get current XY position for moving back at the end of the script:
    x_mm_0, y_mm_0 = scope.XY_stage_position_mm
    
    # Setup minimal positions for no moving (current FOV only):
    XY_stage_positions      = ((0, 0, 'relative'),)
    focus_piezo_positions   = [[0,'relative'],]

    # Optional XYZ moves from position lists collected by GUI:
    # -> uncomment and copy paste lists to use...
    # ***CAUTION WHEN MOVING XY STAGE -> DOUBLE CHECK POSITIONS***
##    XY_stage_positions      = [ # copy past lists in here:
##        # e.g. here's 2 XY positions:
##        [-0.412, -4.9643],
##        [-0.528025, -4.9643],
##        ]
##    focus_piezo_positions   = [ # copy past lists in here:
##        # e.g. here's 2 focus positions:
##        48,
##        48,
##        ]
##    # convert to correct format for .apply_settings():
##    for xy in XY_stage_positions:
##        xy.append('absolute')
##    for i, z in enumerate(focus_piezo_positions):
##            focus_piezo_positions[i] = [z, 'absolute']

    # Get number of positions:
    assert len(focus_piezo_positions) == len(XY_stage_positions)
    positions = len(XY_stage_positions)

    # Make folder name for data:
    folder_label = 'repi_acquisition_template'  # edit name to preference
    dt = datetime.strftime(datetime.now(),'%Y-%m-%d_%H-%M-%S_000_')
    folder_name = dt + folder_label

    # Decide parameters for acquisition:
    iterations = 2      # how many time points?
    time_delay_s = None # delay between acquisitions in seconds (or None)

    # Run acquisition: (tcyx)
    current_time_point = 0
    for i in range(iterations):
        print('\nRunning iteration %i:'%i)
        # start timer:
        t0 = time.perf_counter()
        for p in range(positions):
            # Move to XYZ position:
            # -> remove focus piezo if 'autofocus_enabled=True'
            scope.apply_settings(focus_piezo_z_um=focus_piezo_positions[p],
                                 XY_stage_position_mm=XY_stage_positions[p])
            print('-> position:%i'%p)
            # 470 example:
            filename470 = '470_%06i_%06i.tif'%(i, p)
            scope.apply_settings(
                channels_per_image=('470/24',),
                power_per_channel=(5,),
                emission_filter='Open',
                illumination_time_us=1*1e3,
                images_per_buffer=1,
                )
            # (optional z stack with 'scope.acquire_stack')
            scope.acquire(filename=filename470,
                          folder_name=folder_name,
                          description='470 something...')
            # 550 example:
            filename550 = '550_%06i_%06i.tif'%(i, p)
            scope.apply_settings(
                channels_per_image=('550/15',),
                power_per_channel=(5,),
                emission_filter='Open',
                illumination_time_us=1*1e3,
                images_per_buffer=1,
                )
            # (optional z stack with 'scope.acquire_stack')
            scope.acquire(filename=filename550,
                          folder_name=folder_name,
                          description='550 something...')
        # finish timing:
        loop_time_s = time.perf_counter() - t0
        if i == iterations:
            break # avoid last delay
        # Apply time delay if applicable:
        if time_delay_s is not None:
            if time_delay_s > loop_time_s:
                print('\nApplying time_delay_s: %0.2f'%time_delay_s)
                time.sleep(time_delay_s - loop_time_s)
            else:
                print('\n***WARNING***')
                print('time_delay_s not applied (loop_time_s > time_delay_s)')

    # return to 'zero' starting position for user convenience
    scope.apply_settings(focus_piezo_z_um=focus_piezo_positions[0],
                         XY_stage_position_mm=(x_mm_0, y_mm_0, 'absolute'))
    scope.close()
