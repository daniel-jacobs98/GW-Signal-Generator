'''
	Daniel Jacobs 2020
'''

import numpy as np
import h5py
from matplotlib import pyplot as plt
import argparse
import random as rand

#TODO: fix figure dimensions
#TODO: Some waves have merge points not in the signals: When cutting signals shift merger to middle (hanford)

def plot_sigs((h1_strain, h1_time), (l1_strain, l1_time), (v1_strain, v1_time), signal_parameters):
	param_dict = {param[0]: float(param[1]) for param in signal_parameters}
	fig, axes = plt.subplots(nrows=3)
	plt.subplots_adjust(hspace=0.45, top=0.93, bottom=0.1)
	axes[0].plot(h1_time, h1_strain, label='Hanford')
	axes[1].plot(l1_time, l1_strain, label='Livingston')
	axes[2].plot(v1_time, v1_strain, label='Virgo')

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

		waves_to_plot = rand.sample(range(f['signals']['H1 strain'].shape[0]), k=20)
		for i in waves_to_plot:
			fig = plot_sigs((h1_strain[i,:], h1_times[i,:]), (l1_strain[i,:], l1_times[i,:]), (v1_strain[i,:], v1_times[i,:]),
							signal_parameters[i,:])
			plt.show()