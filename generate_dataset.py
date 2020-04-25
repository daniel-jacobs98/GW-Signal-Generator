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

global signal_len
signal_len=10000

global sim_params
#Defines ranges that parameters are sampled from
sim_params = {
	'mass_range':(1.4,80), # sometimes getting input domain error something to do with low mass
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

#Return a parameter set describing a signal uniformly samples from the sim_param ranges
def get_param_set(sim_params):
	param_set = {}
	for i in range(0, sim_params['num_signals']):
		param_set['m1'] = rand.uniform(sim_params['mass_range'][0], sim_params['mass_range'][1])
		param_set['m2'] = rand.uniform(sim_params['mass_range'][0], sim_params['mass_range'][1])

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


#Generate and return projections of a signal described by param_set onto the Hanford, Livingston, Virgo
# detectors
def generate_signal(param_set):
	print('Masses: ({0}, {1})'.format(param_set['m1'], param_set['m2']))
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
	end_time = 1192529720
	hp = fade_on(hp)
	hc = fade_on(hc)
	hp.resize(signal_len)
	hc.resize(signal_len)


	det_h1 = Detector('H1')
	det_l1 = Detector('L1')
	det_v1 = Detector('V1')

	sig_h1 = det_h1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
	sig_l1 = det_l1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
	sig_v1 = det_v1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])

	return {'H1':sig_h1,
		'L1':sig_l1,
		'V1':sig_v1
	}


def inject_signals_gaussian(signal_dict, inj_snr):
	noise = dict()
	for i, det in enumerate(('H1', 'L1', 'V1')):
		flow = 30.0
		delta_f = signal_dict[det].delta_f
		flen = int(16384 / delta_f) + 1
		psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
		noise[det] = pycbc.noise.gaussian.noise_from_psd(length=signal_dict[det].sample_times.shape[0],delta_t=1.0/16384, psd=psd)
		start_time = signal_dict[det].start_time
		noise[det]._epoch = LIGOTimeGPS(start_time)

	psds = dict()
	dummy_strain = dict()
	snrs = dict()
	for det in ('H1', 'L1', 'V1'):
		delta_f = signal_dict[det].delta_f
		noise_inj = noise[det].add_into(signal_dict[det])
		dummy_strain[det] = noise_inj
		# print('Signal len (seconds): {0}'.format(signal_dict[det].sample_times.shape[0]))
		# print('Noise inj len (seconds): {0}'.format(noise_inj.sample_times.shape[0]))
		psds[det] = noise_inj.psd(0.5)
		psds[det] = pycbc.psd.interpolate(psds[det], delta_f=delta_f)
		# print('PSD delta_f: {0}'.format(psds[det].delta_f))
		# print('Waveform delta_f: {0}'.format(signal_dict[det].delta_f))
		# print('Noise delta_f: {0}'.format(noise[det].delta_f))
		snrs[det] = sigma(htilde=signal_dict[det],
							psd=psds[det],
							low_frequency_cutoff=flow)
	nomf_snr = np.sqrt(snrs['H1']**2 + snrs['L1']**2 + snrs['V1']**2)
	scale_factor = inj_snr/float(nomf_snr)
	noisy_signals = dict()

	for det in ('H1', 'L1', 'V1'):
		noisy_signals[det] = noise[det].add_into(scale_factor*signal_dict[det])
	return noisy_signals

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''Generate a GW signal dataset 
									with Gaussian noise''')
	parser.add_argument('-cf','--config-file', help='JSON config file')
	parser.add_argument('-o', '--output-dir', help='Directory to store output HDF files (num_signals/10)')
	args = vars(parser.parse_args())

	start=time.time()

	if args['config_file']!=None:
		with open(args['config_file']) as config_file:
			config_data = json.load(config_file)
			sim_params['mass_range'] = (config_data['min_mass'], config_data['max_mass'])
			sim_params['spin_range'] = (config_data['min_spin'], config_data['max_spin'])
			sim_params['num_signals'] = config_data['num_signals']
			sim_params['distance_range'] = (config_data['min_distance'], config_data['max_distance'])
			sim_params['snr_range'] = (config_data['min_snr'], config_data['max_snr'])

	output_path = args['output_dir'] if args['output_dir']!=None else 'output.hdf'
	if os.path.isfile(output_path):
		print('Output file already exists. Please remove this file or use a different file name.')
		sys.exit(1)

	sig_list = []
	param_list = []
	noisy_sig_list = []
	for i in range(0, sim_params['num_signals']):
		param_list.append(next(get_param_set(sim_params)))
		sig_list.append(generate_signal(param_list[-1]))
		noisy_sig_list.append(inject_signals_gaussian(sig_list[-1], param_list[-1]['snr']))
		#h1 = sig_list[i]['H1']
		#l1 = sig_list[i]['L1']
		#v1 = sig_list[i]['V1']
		
		#fig = plot_sigs(h1, l1, v1)
		#plt.show()
		#fig.savefig('test_sample_plot_{0}.png'.format(i))
		#plt.close(fig)
		if i%10==0 and i!=0:
			#change output path to user defined name, doesnt change
			to_hdf(output_path, sim_params, noisy_sig_list, param_list, signal_len)
			sig_list=[]
			param_list=[]
			noisy_sig_list=[]
	to_hdf(output_path, sim_params, noisy_sig_list, param_list, signal_len)

	end = time.time()
	print('\nFinished! Took {0} seconds to generate and save {1} samples.\n'.format(float(end-start), sim_params['num_signals']))
	'''TODO: Add Gaussian noise'''

