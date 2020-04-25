'''
	Daniel Jacobs 2020
'''

#TODO: Write in chunks (append) to HDF
#TODO: Make all waveforms the same size so they can be appended to HDF
#TODO: Add Gaussian noise
#TODO: Fix mass (frequency) error
#TODO:

from pycbc.waveform import td_approximants
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import pycbc.noise
import pycbc.psd
from pycbc.filter import sigma
import matplotlib.pyplot as plt 
import random as rand 
import math
import argparse
import json
import h5py 
import numpy as np
from generation_utils import fade_on, to_hdf
import time
import sys
import os.path
from lal import LIGOTimeGPS
from plot_signal import plot_sigs
import copy
global signal_len
signal_len=10000

global sim_params
#Defines ranges that parameters are sampled from
sim_params = {
	'mass_range':(1.4,80),
	'spin_range':(-1,1),
	'num_signals':20,
	'inclination_range':(0, math.pi),
	'coa_phase_range':(0, 2*math.pi),
	'right_asc_range':(0, 2*math.pi),
	'declination_range':(0, 1),
	'polarisation_range':(0, 2*math.pi),
	'distance_range':(40, 3000),
	'snr_range':(15, 20),

	#Increased sampleing rate to account for LalSim error: ringdown frequency>nyquist frequency.
	#I think this is because if the masses are very small (neutron star level) they have too high a 
	#ringdown frequency to be fully described by a sampling rate of 4096. I questioned why a paper used
	#8096Hz in my proposal, this is probably why.'''
	'sample_freq':16384 
}

#Yield a parameter set describing a signal uniformly samples from the sim_param ranges
def get_param_set(sim_params):
	param_set = {}
	for i in range(0, sim_params['num_signals']):
		param_set['m1'] = rand.uniform(sim_params['mass_range'][0], sim_params['mass_range'][1])
		param_set['m2'] = rand.uniform(sim_params['mass_range'][0], sim_params['mass_range'][1])

		#Stick to convention, make the larger body mass1
		if param_set['m2']>param_set['m1']:
			m_lesser = param_set['m1']
			param_set['m1']=param_set['m2']
			param_set['m2']=m_lesser

		param_set['x1'] = rand.uniform(sim_params['spin_range'][0], sim_params['spin_range'][1])
		param_set['x2'] = rand.uniform(sim_params['spin_range'][0], sim_params['spin_range'][1])
		param_set['inc'] = rand.uniform(sim_params['inclination_range'][0], sim_params['inclination_range'][1])
		param_set['coa'] = rand.uniform(sim_params['coa_phase_range'][0], sim_params['coa_phase_range'][1])
		param_set['ra'] = rand.uniform(sim_params['right_asc_range'][0], sim_params['right_asc_range'][1])
		param_set['dec'] = math.asin(1-(2*rand.uniform(sim_params['declination_range'][0], sim_params['declination_range'][1])))
		param_set['pol'] = rand.uniform(sim_params['polarisation_range'][0], sim_params['polarisation_range'][1])
		param_set['dist'] = rand.randint(sim_params['distance_range'][0], sim_params['distance_range'][1])
		param_set['f'] = sim_params['sample_freq']
		param_set['snr'] = rand.uniform(sim_params['snr_range'][0], sim_params['snr_range'][1])
		yield param_set


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

#Generate and return projections of a signal described by param_set onto the Hanford, Livingston, Virgo detectors
def generate_signal(param_set):
	hp, hc = get_td_waveform(approximant='SEOBNRv4',
							mass1=param_set['m1'],
							mass2=param_set['m2'],
							spin1z=param_set['x1'],
							spin2z=param_set['x2'],
							inclination_range=param_set['inc'],
							coa_phase=param_set['coa'],
							distance=param_set['dist'],
							delta_t=1.0/param_set['f'],
							f_lower=30)

	det_h1 = Detector('H1')
	det_l1 = Detector('L1')
	det_v1 = Detector('V1')

	sig_h1 = det_h1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
	sig_l1 = det_l1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
	sig_v1 = det_v1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])

	return {'H1':sig_h1,
			'L1':sig_l1,
			'V1':sig_v1}

def cut_sigs(signal_dict):
	cut_sigs = dict()
	# rs_sig_dict = copy.deepcopy(signal_dict)
	# resize_works = True
	zeroIdxs = {
		'H1':find_nearest(signal_dict['H1'].sample_times, 0),
		'L1':find_nearest(signal_dict['L1'].sample_times, 0),
		'V1':find_nearest(signal_dict['V1'].sample_times, 0)
	}
	# for det in ['H1', 'L1', 'V1']:
	# 	zIdx = zeroIdxs[det]
	# 	res = rs_sig_dict[det]
	# 	res.resize(signal_len)
	# 	if zIdx>res.shape[0]:
	# 		resize_works=False
	# 		break
	# 	else:
	# 		cut_sigs[det]=res

	# if resize_works:
	# 	print('Resize')
	# 	print('H1: {0}\nL1: {1}\nV1: {2}'.format(cut_sigs['H1'].shape[0], cut_sigs['L1'].shape[0], cut_sigs['V1'].shape[0]))
	# 	return cut_sigs

	print('After res attempt: {0}, {1}, {2}'.format(signal_dict['H1'].shape[0], signal_dict['L1'].shape[0], signal_dict['V1'].shape[0]))
	for det in ['H1','L1','V1']:
		zIdx = zeroIdxs[det]
		sig = signal_dict[det]
		print(zIdx)
		startIdx = int(zIdx-(signal_len*0.9))
		prep_zeros = 0
		print(startIdx)
		endIdx = int(zIdx+(signal_len*0.1))
		ap_zeros = 0
		print(endIdx)
		if startIdx<0:
			prep_zeros = int(startIdx*-1)
			startIdx = 0
		if endIdx>sig.shape[0]:
			ap_zeros = endIdx-sig.shape[0]
			endIdx = sig.shape[0]-1
		print('sIdx: {0}, eIdx:  {1}'.format(startIdx, endIdx))
		print(sig.shape[0])
		res = sig[startIdx:endIdx]
		if res.shape[0]!=signal_len:
			res.prepend_zeros(prep_zeros)
			res.append_zeros(ap_zeros)

		new_zidx = find_nearest(res.sample_times, 0)
		print('In' if (new_zidx<endIdx and new_zidx>zIdx) else 'Not in')
		cut_sigs[det]=res
	print('Cut')
	print('H1: {0}\nL1: {1}\nV1: {2}'.format(cut_sigs['H1'].shape[0], cut_sigs['L1'].shape[0], cut_sigs['V1'].shape[0]))
	return cut_sigs

#Inject a set of signals into Gaussian noise with the given SNR
def inject_signals_gaussian(signal_dict, inj_snr):
	resized_sigs = cut_sigs(signal_dict)
	noise = dict()
	global sim_params
	for i, det in enumerate(('H1', 'L1', 'V1')):
		flow = 30.0
		delta_f = resized_sigs[det].delta_f
		flen = int(sim_params['sample_freq'] / delta_f) + 1
		psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
		noise[det] = pycbc.noise.gaussian.noise_from_psd(length=resized_sigs[det].sample_times.shape[0],
														delta_t=1.0/sim_params['sample_freq'], psd=psd)
		start_time = resized_sigs[det].start_time
		noise[det]._epoch = LIGOTimeGPS(start_time)

	psds = dict()
	dummy_strain = dict()
	snrs = dict()

	#using dummy strain and psds from the noise, calculate the snr of each signal+noise injection to find the 
	#network optimal SNR, used for injecting the real signal
	for det in ('H1', 'L1', 'V1'):
		delta_f = resized_sigs[det].delta_f
		noise_inj = noise[det].add_into(resized_sigs[det])
		dummy_strain[det] = noise_inj
		psds[det] = noise_inj.psd(0.6)
		psds[det] = pycbc.psd.interpolate(psds[det], delta_f=delta_f)
		snrs[det] = sigma(htilde=resized_sigs[det],
							psd=psds[det],
							low_frequency_cutoff=flow)
	nomf_snr = np.sqrt(snrs['H1']**2 + snrs['L1']**2 + snrs['V1']**2)
	scale_factor = inj_snr/float(nomf_snr)
	noisy_signals = dict()

	#inject signals with the correct scaling factor for the target SNR
	for det in ('H1', 'L1', 'V1'):
		noisy_signals[det] = noise[det].add_into(scale_factor*resized_sigs[det])
	return noisy_signals

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''Generate a GW signal dataset 
									with Gaussian noise''')
	parser.add_argument('-cf','--config-file', help='JSON config file')
	parser.add_argument('-o', '--output', help='Specify output HDF file name')
	args = vars(parser.parse_args())

	#If config file provided, override default ranges with those in the config
	if args['config_file']!=None:
		with open(args['config_file']) as config_file:
			config_data = json.load(config_file)
			sim_params['mass_range'] = (config_data['min_mass'], config_data['max_mass'])
			sim_params['spin_range'] = (config_data['min_spin'], config_data['max_spin'])
			sim_params['num_signals'] = config_data['num_signals']
			sim_params['distance_range'] = (config_data['min_distance'], config_data['max_distance'])
			sim_params['snr_range'] = (config_data['min_snr'], config_data['max_snr'])

	output_path = args['output'] if args['output']!=None else 'output.hdf'
	if os.path.isfile(output_path):
		print('Output file already exists. Please remove this file or use a different file name.')
		sys.exit(1)

	start=time.time()
	print('Starting generation of {0} signals...'.format(sim_params['num_signals']))

	sig_list = []
	param_list = []
	noisy_sig_list = []

	# test_params = {
	# 	'm1':38.81,
	# 	'm2':8.30,
	# 	'x1':0.36,
	# 	'x2':-0.81,
	# 	'snr':18.20,
	# 	'dist':1839,
	# 	'ra':2.64,
	# 	'dec':-0.24,
	# 	'f':16384,
	# 	'coa':rand.uniform(sim_params['coa_phase_range'][0], sim_params['coa_phase_range'][1]),
	# 	'pol':rand.uniform(sim_params['polarisation_range'][0], sim_params['polarisation_range'][1]),
	# 	'inc':rand.uniform(sim_params['inclination_range'][0], sim_params['inclination_range'][1])
	# }
	# sig = generate_signal(test_params)
	# print(sig)
	# noisy_sig = inject_signals_gaussian(sig, test_params['snr'])
	# print(noisy_sig['H1'].shape[0])
	# print(noisy_sig['L1'].shape[0])	
	# print(noisy_sig['V1'].shape[0])
	# fig, axes = plt.subplots(nrows=6)
	# for i, det in enumerate(['H1', 'L1', 'V1']):
	# 	axes[i].plot(sig[det].sample_times, sig[det], label=det)
	# 	axes[i+3].plot(noisy_sig[det].sample_times, noisy_sig[det], label=det)
	# plt.show()
	# sys.exit()

	for i in range(0, sim_params['num_signals']):
		param_list.append(next(get_param_set(sim_params)))
		sig_list.append(generate_signal(param_list[-1]))
		noisy_sig_list.append(inject_signals_gaussian(sig_list[-1], param_list[-1]['snr']))

		#Every x loops, save the samples generated, stops memory errors when generating large datasets
		#Recommend value ~1000 (~400Mb)
		x=1000
		if i%x==0 and i!=0:
			to_hdf(output_path, sim_params, noisy_sig_list, param_list, signal_len)
			sig_list=[]
			param_list=[]
			noisy_sig_list=[]
	to_hdf(output_path, sim_params, noisy_sig_list, param_list, signal_len)

	end = time.time()
	print('\nFinished! Took {0} seconds to generate and save {1} samples.\n'.format(float(end-start), sim_params['num_signals']))
