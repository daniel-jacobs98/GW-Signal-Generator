import numpy as np
from pycbc.types.timeseries import TimeSeries
from scipy.signal.windows import tukey
import h5py

#Not my function - Ref: Gebhard, Kilbertus https://github.com/timothygebhard/ggwd
def fade_on(timeseries,
            alpha=0.25):
    """
    Take a PyCBC time series and use a one-sided Tukey window to "fade
    on" the waveform (to reduce discontinuities in the amplitude).

    Args:
        timeseries (pycbc.types.timeseries.TimeSeries): The PyCBC
            TimeSeries object to be faded on.
        alpha (float): The alpha parameter for the Tukey window.

    Returns:
        The `timeseries` which has been faded on.
    """

    # Save the parameters from the time series we are about to fade on
    delta_t = timeseries.delta_t
    epoch = timeseries.start_time
    duration = timeseries.duration
    sample_rate = timeseries.sample_rate

    # Create a one-sided Tukey window for the turn on
    window = tukey(M=int(duration * sample_rate), alpha=alpha)
    window[int(0.5*len(window)):] = 1

    # Apply the one-sided Tukey window for the fade-on
    ts = window * np.array(timeseries)

    # Create and return a TimeSeries object again from the resulting array
    # using the original parameters (delta_t and epoch) of the time series
    return TimeSeries(initial_array=ts,
                      delta_t=delta_t,
                      epoch=epoch)


#fixes signals all to the same length so they can be saved together in a HDF file
#extends shorter signals with 0s
def convert_cbc_array_to_np(signals, target_length):
    cbc_arrs=[]

    if type(signals)==dict:
        for det, cbc in signals.items():
            cbc_arr = np.array(cbc)
            cbc_arrs.append(cbc_arr)
    elif type(signals)==list:
        for cbc in signals:
            cbc_arr = np.array(cbc)
            cbc_arrs.append(cbc_arr)

    if cbc_arrs[0].shape==cbc_arrs[1].shape==cbc_arrs[2].shape:
        return cbc_arrs

    corrected_cbc = []
    for cbc in cbc_arrs:
        zero_arr = np.zeros((target_length,))
        slice_idx = target_length if cbc.shape[0]>=target_length else cbc.shape[0]
        zero_arr[:cbc.shape[0]] = cbc[:slice_idx]
        corrected_cbc.append(zero_arr)

    return corrected_cbc


#Take signal series and add them to the dataset hdf file
#If this is the first set, create groups and file structure
def to_hdf(file_path, sim_params, data, sig_params, signal_len):
    #Initialise arrays to hold strain, sample_times, and parameters
    shape = (len(data), signal_len+68)
    h1_strain = np.zeros(shape)
    l1_strain = np.zeros(shape)
    v1_strain = np.zeros(shape)
    h1_time = np.zeros(shape)
    l1_time = np.zeros(shape)
    v1_time = np.zeros(shape)
    sig_param_arr = np.empty(shape=(len(data), 12, 2), dtype='|S16')

    for i in range(0, len(data)):

        #Join signals as one Numpy array
        sig_set = convert_cbc_array_to_np(data[i], signal_len+68)
        h1_strain[i,:sig_set[0].shape[0]] = sig_set[0]
        l1_strain[i,:sig_set[1].shape[0]] = sig_set[1]
        v1_strain[i,:sig_set[2].shape[0]] = sig_set[2]

        #Join time as one numpy array
        time_set = convert_cbc_array_to_np([data[i]['H1'].sample_times, data[i]['L1'].sample_times,
                                            data[i]['V1'].sample_times], signal_len+68)
        h1_time[i, :time_set[0].shape[0]] = time_set[0]
        l1_time[i, :time_set[1].shape[0]] = time_set[1]
        v1_time[i, :time_set[2].shape[0]] = time_set[2]

        #Convert simulation parameters of this signal to Numpy array
        param_set = sig_params[i]
        param_set_arr = np.array(list(param_set.items()))
        sig_param_arr[i, :, :] = param_set_arr

    signal_dict = {
        'H1':(h1_strain, h1_time),
        'L1':(l1_strain, l1_time),
        'V1':(v1_strain, v1_time)
    }
    with h5py.File(file_path, 'a') as hdf:
        append = True if 'signals' in hdf.keys() else False
        if not append:
            sig_group = hdf.create_group('signals')
            for det, (sig_strain, sig_time) in signal_dict.items():
                #First dimension of maxshape needs to be None so that the datasets can be resized
                sig_group.create_dataset(name='{0} strain'.format(det),
                                            dtype='float32',
                                            shape=sig_strain.shape,
                                            data=sig_strain,
                                            maxshape=(None, signal_len+68))
                sig_group.create_dataset(name='{0} times'.format(det),
                                            dtype='float32',
                                            shape=sig_time.shape,
                                            data=sig_time,
                                            maxshape=(None, signal_len+68))
            sig_group.create_dataset(name='Signal parameters',
                                        dtype='|S16',
                                        shape=sig_param_arr.shape,
                                        data=sig_param_arr,
                                        maxshape=(None, 12, 2))

            #store ranges used in this simulation run
            sim_ranges_group = hdf.create_group('simulation_ranges')
            for key, val in sim_params.items():
                sim_ranges_group.attrs[key]=val
        else:
            sig_group = hdf['signals']
            for det, (sig_strain, sig_time) in signal_dict.items():
                det_strain = sig_group['{0} strain'.format(det)]
                det_strain.resize((det_strain.shape[0]+sig_strain.shape[0], signal_len+68))
                det_strain[-sig_strain.shape[0]:] = sig_strain

                det_times = sig_group['{0} times'.format(det)]
                det_times.resize((det_times.shape[0]+sig_time.shape[0], signal_len+68))
                det_times[-sig_time.shape[0]:] = sig_time

            sig_param_group = sig_group['Signal parameters']
            sig_param_group.resize((sig_param_group.shape[0] + sig_param_arr.shape[0], 12, 2))
            sig_param_group[-sig_param_arr.shape[0]::] = sig_param_arr

