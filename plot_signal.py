'''
	Daniel Jacobs 2020
'''

import numpy as np
import h5py
from matplotlib import pyplot as plt
import argparse
import random as rand
from generation_utils import find_nearest

def plot_with_pure(pure_signals, noisy_signals, signal_parameters):
	print(signal_parameters)
	fig, axes = plt.subplots(nrows=3)
	plt.subplots_adjust(hspace=0.45, top=0.93, bottom=0.1)


	for i, det in enumerate(['H1', 'L1', 'V1']):
		axes[i].plot(noisy_signals[det].sample_times, noisy_signals[det], 'b')
		axes[i].plot(pure_signals[det].sample_times, pure_signals[det], 'r')


	for i, detname in enumerate(['Hanford', 'Livingston', 'Virgo']):
		axes[i].set_ylabel('{0} strain'.format(detname))
		#add line at merge point
		axes[i].axvline(x=0, color='black', ls='--', lw=1)

	axes[2].set_xlabel('Seconds from merger')
	parameters = 'Mass1={:.2f}, Mass2={:.2f}, SNR={:.2f}, Spin1={:.2f}, Spin2={:.2f}, Distance={:.2f}, RA={:.2f}, Dec={:.2f}'.format(
						signal_parameters['m1'], signal_parameters['m2'], signal_parameters['snr'], 
						signal_parameters['x1'], signal_parameters['x2'], signal_parameters['dist'], 
						signal_parameters['ra'], signal_parameters['dec'])
	plt.figtext(0.5, 0.95, parameters, fontsize=8, ha='center')
	plt.gcf().set_size_inches(12, 6, forward=True)
	return fig

def plot_sigs((h1_strain, h1_time), (l1_strain, l1_time), (v1_strain, v1_time), signal_parameters):
	param_dict = {param[0]: float(param[1]) for param in signal_parameters}
	fig, axes = plt.subplots(nrows=3)
	plt.subplots_adjust(hspace=0.45, top=0.93, bottom=0.1)

	zIdx = find_nearest(h1_time,0)
	axes[0].plot(h1_time[zIdx:], h1_strain[zIdx:], label='Hanford')
	zIdx = find_nearest(l1_time,0)
	axes[1].plot(l1_time[zIdx:], l1_strain[zIdx:], label='Livingston')
	zIdx = find_nearest(v1_time,0)
	axes[2].plot(v1_time[zIdx:], v1_strain[zIdx:], label='Virgo')

	for i, detname in enumerate(['Hanford', 'Livingston', 'Virgo']):
		axes[i].set_ylabel('{0} strain'.format(detname))
		#add line at merge point
		axes[i].axvline(x=0, color='black', ls='--', lw=1)

	axes[2].set_xlabel('Seconds from merger')
	parameters = 'Mass1={:.2f}, Mass2={:.2f}, SNR={:.2f}, Spin1={:.2f}, Spin2={:.2f}, Distance={:.2f}, RA={:.2f}, Dec={:.2f}'.format(
						param_dict['m1'], param_dict['m2'], param_dict['snr'], 
						param_dict['x1'], param_dict['x2'], param_dict['dist'], param_dict['ra'], param_dict['dec'])
	plt.figtext(0.5, 0.95, parameters, fontsize=8, ha='center')
	plt.gcf().set_size_inches(12, 6, forward=True)
	return fig

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Plot a signal given in hdf format')
	parser.add_argument('-i', '--input')
	args = vars(parser.parse_args())

	with h5py.File(args['input'], 'r') as f:
		print(f.keys())
		print(f['signals']['Signal parameters'])
		print(f['signals']['H1 strain'])
		print(f['signals']['H1 times'])


		h1_strain = f['signals']['H1 strain']
		h1_times = f['signals']['H1 times']
		l1_strain = f['signals']['L1 strain']
		l1_times = f['signals']['L1 times']
		v1_strain = f['signals']['V1 strain']
		v1_times = f['signals']['V1 times']
		signal_parameters = f['signals']['Signal parameters']

		masses = []
		snrs = []
		spins = []
		for i in range(signal_parameters.shape[0]):
			params = {d[0]: float(d[1].replace('e-','')) for d in signal_parameters[i,:,:]}
			masses.append(params['m1'])
			masses.append(params['m2'])

			spins.append(params['x1'])
			spins.append(params['x2'])
			snrs.append(params['snr'])
		masses = np.array(masses)
		spins = np.array(spins)
		snrs = np.array(snrs)

		fig, axes = plt.subplots(nrows=3)
		axes[0].hist(masses, bins=80)
		axes[0].set_xlabel('Component masses (M-Sun)')
		axes[1].hist(spins, bins=100)
		axes[1].set_xlabel('Component spins')
		axes[2].hist(snrs, bins=20, label='SNRs')
		axes[2].set_xlabel('Injected signal SNR')
		fig.suptitle('Distributions of Masses, Spins and SNRs')
		plt.subplots_adjust(hspace=0.45)
		plt.show()
		# waves_to_plot = rand.sample(range(f['signals']['H1 strain'].shape[0]), k=20)
		# waves_to_plot=[1,9]
		# for i in waves_to_plot:
		# 	fig = plot_sigs((h1_strain[i,:], h1_times[i,:]), (l1_strain[i,:], l1_times[i,:]), (v1_strain[i,:], v1_times[i,:]),
		# 					signal_parameters[i,:])
		# 	fig.suptitle(i)
		# 	plt.show()