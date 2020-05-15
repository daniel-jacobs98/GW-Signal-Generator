'''
	Daniel Jacobs 2020
	OzGrav - University of Western Australia
'''

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
from plot_signal import plot_sigs, plot_with_pure, find_nearest
import copy


global sim_params
#Defines ranges that parameters are sampled from
sim_params = {
	# Black hole mass range
	'mass_range':(5,80),
	'spin_range':(-1,1),
	'num_signals':10000,
	'inclination_range':(0, math.pi),
	'coa_phase_range':(0, 2*math.pi),
	'right_asc_range':(0, 2*math.pi),
	'declination_range':(0, 1),
	'polarisation_range':(0, 2*math.pi),
	'distance_range':(40, 3000),
	'snr_range':(10,20),
	'sample_freq':4096 
}

global signal_len
signal_len=0.25*sim_params['sample_freq']

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


#Generate and return projections of a signal described by param_set onto the Hanford, Livingston, Virgo detectors
def generate_signal(param_set):
	hp, hc = get_td_waveform(approximant='SEOBNRv4', #This approximant is only appropriate for BBH mergers
							mass1=param_set['m1'],
							mass2=param_set['m2'],
							spin1z=param_set['x1'],
							spin2z=param_set['x2'],
							inclination_range=param_set['inc'],
							coa_phase=param_set['coa'],
							distance=param_set['dist'],
							delta_t=1.0/param_set['f'],
							f_lower=30)

	time = 100000000

	det_h1 = Detector('H1')
	det_l1 = Detector('L1')
	det_v1 = Detector('V1')

	hp = fade_on(hp,0.25)
	hc = fade_on(hc,0.25)

	sig_h1 = det_h1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
	sig_l1 = det_l1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
	sig_v1 = det_v1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])

	return {'H1':sig_h1,
			'L1':sig_l1,
			'V1':sig_v1}


#Reshape signals to desired length by appending and prepending zeros if necessary
def cut_sigs(signal_dict):
	cut_sigs = dict()
	zeroIdxs = {
		'H1':find_nearest(signal_dict['H1'].sample_times, 0),
		'L1':find_nearest(signal_dict['L1'].sample_times, 0),
		'V1':find_nearest(signal_dict['V1'].sample_times, 0)
	}

	for det in ['H1','L1','V1']:
		zIdx = zeroIdxs[det]
		sig = signal_dict[det]
		startIdx = int(zIdx-(math.floor(signal_len*0.8)))
		prep_zeros = 0
		endIdx = int(zIdx+(math.ceil(signal_len*0.2)))
		ap_zeros = 0
		if startIdx<0:
			prep_zeros = int(startIdx*-1)
			startIdx = 0
		if endIdx>sig.shape[0]:
			ap_zeros = endIdx-sig.shape[0]
			endIdx = sig.shape[0]-1
		res = sig[startIdx:endIdx]
		if res.shape[0]!=signal_len:
			res.prepend_zeros(prep_zeros)
			res.append_zeros(ap_zeros)
		cut_sigs[det]=res
	return cut_sigs

#Inject a set of signals into Gaussian noise with the given SNR
def inject_signals_gaussian(signal_dict, inj_snr, sig_params):
	resized_sigs = cut_sigs(signal_dict)
	noise = dict()
	global sim_params
	for i, det in enumerate(('H1', 'L1', 'V1')):
		flow = 30.0
		delta_f = resized_sigs[det].delta_f
		flen = int(sim_params['sample_freq'] / delta_f) + 1
		psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
		noise[det] = pycbc.noise.gaussian.noise_from_psd(length=sim_params['sample_freq']*16,
														delta_t=1.0/sim_params['sample_freq'], psd=psd)
		start_time = resized_sigs[det].start_time-8
		noise[det]._epoch = LIGOTimeGPS(start_time)

	psds = dict()
	dummy_strain = dict()
	snrs = dict()

	#using dummy strain and psds from the noise, calculate the snr of each signal+noise injection to find the 
	#network optimal SNR, used for injecting the real signal
	for det in ('H1', 'L1', 'V1'):
		delta_f = resized_sigs[det].delta_f
		dummy_strain[det] = noise[det].add_into(resized_sigs[det])
		
		psds[det] = dummy_strain[det].psd(0.2)
		psds[det] = pycbc.psd.interpolate(psds[det], delta_f=delta_f)
		snrs[det] = sigma(htilde=resized_sigs[det],
							psd=psds[det],
							low_frequency_cutoff=flow)
	nomf_snr = np.sqrt(2*((snrs['H1']+snrs['L1']+snrs['V1'])/3)**2)
	scale_factor = 1.0* inj_snr/nomf_snr
	noisy_signals = dict()

	#inject signals with the correct scaling factor for the target SNR
	for det in ('H1', 'L1', 'V1'):
		noisy_signals[det] = noise[det].add_into(resized_sigs[det]*scale_factor)

		#Whiten signal
		noisy_signals[det] = noisy_signals[det].whiten(segment_duration=1,
														max_filter_duration=1, 
														remove_corrupted=False,
														low_frequency_cutoff=30.0)

		#Cut down to desired length and cut off corrupted tails of signal
		noisy_signals[det] = noisy_signals[det].time_slice(-0.2, 0.05)
		
		# fig, axes = plt.subplots(nrows=3)

		# axes[0].plot(resized_sigs[det].sample_times, resized_sigs[det], 'r')
		# axes[0].set_title('Pure signal (at {0})'.format(det))
		# axes[0].set_ylabel('Strain')

		# plot_noise = noise[det].time_slice(-0.2,0.05)
		# axes[1].plot(plot_noise.sample_times, plot_noise, 'b')
		# axes[1].set_title('Gaussian noise')
		# axes[1].set_ylabel('Strain')

		# axes[2].plot(noisy_signals[det].sample_times, noisy_signals[det], 'b')
		# axes[2].set_ylabel('Strain')

		# axes[2].set_xlabel('Time from merger (seconds')
		# axes[2].set_title('Injected signal (at {0})'.format(det))
		# plt.subplots_adjust(hspace=0.45)
		# fig.suptitle('Masses: [{0:.2f}, {1:.2f}], SNR: {2:.2f}'.format(sig_params['m1'], sig_params['m2'], inj_snr))
		# plt.show()
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
	offset_list = []
	param_list = []
	noisy_sig_list = []

	for i in range(0, sim_params['num_signals']):
		param_list.append(next(get_param_set(sim_params)))
		sig_list.append(generate_signal(param_list[-1]))
		noisy_sig_list.append(inject_signals_gaussian(sig_list[-1], param_list[-1]['snr'], param_list[-1]))
	
		#Every x loops, save the samples generated, stops memory errors when generating large datasets
		#Recommend value ~1000 (~400Mb)
		x=1000
		if i%x==0 and i!=0:
			print('Finished {0}...'.format(i))
			to_hdf(output_path, sim_params, noisy_sig_list, param_list, signal_len)
			sig_list=[]
			param_list=[]
			noisy_sig_list=[]
			offset_list=[]
	to_hdf(output_path, sim_params, noisy_sig_list, param_list, signal_len)

	end = time.time()
	print('\nFinished! Took {0} seconds to generate and save {1} samples.\n'.format(float(end-start), sim_params['num_signals']))
