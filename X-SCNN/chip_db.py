#!/usr/bin/python3.6 -tt

import h5py, pandas as pd, argparse, gzip, numpy as np, sys


def gopen(file):
	if file[-3:] == '.gz':
		return gzip.open(file, 'rb')
	return open(file, 'rU')


def z_score(in_array, means, stds):
	# takes in an array of size (features / length), means and stds of size (features), 
	# and returns a z-score
	return (in_array - np.expand_dims(means, 1)) / np.expand_dims(stds, 1)


def get_start(chromosome, chr_names, chr_starts):
	# returns the starting position (in index-space) of the passed chromosome
	idx = int(chr_starts[chr_names == chromosome])
	return idx


def bg_to_wig(infile, gen_len, chr_names, chr_starts, step_size):
	# Jump straight to array generation
	out_array = np.zeros(gen_len)
	with gopen(infile) as openfile:
		for line in openfile:
			# omit all commented lines
			if not line[0] == '#':
				field = line.split()
				# Find current position and values
				curr_chr = field[0]
				bg_start = int(field[1]) + 1
				bg_end = int(field[2]) + 1
				bg_value = float(field[3])
				# Find window indexes which overlap this interval
				chr_start_idx = int(bg_start / step_size)
				chr_end_idx = int(bg_end / step_size)
				window_start_idx = get_start(curr_chr, chr_names, chr_starts) + chr_start_idx
				window_end_idx = get_start(curr_chr, chr_names, chr_starts) + chr_end_idx
				# For each window that is covered by the bg interval, add values
				for chr_idx, window_idx in zip(list(range(chr_start_idx, chr_end_idx+1)), 
							list(range(window_start_idx, window_end_idx+1))):
					out_array[window_idx] += (min((chr_idx+1) * step_size, bg_end) - \
					max(chr_idx * step_size, bg_start)) * bg_value
	#
	return out_array


def make_chr_breakpoints(size_file, resolution):
	# returns two lists, one with chromosome names, the other with starting positions in the array
	# Also keep track of the size of each chromosome (in index length)
	chr_names = []
	chr_sizes = []
	chr_starts = [0]
	#
	with open(size_file,'rU') as chromfile:
		for line in chromfile:
			field = line.split()
			# Chromosome name, length of chromosome (in index length), and chromosome start position
			chr_names.append(field[0])
			length = int(field[1])
			chr_sizes.append(length/resolution + bool(length%resolution) )
			chr_starts.append(chr_starts[-1] + chr_sizes[-1])
	# return the lists, along with the total genome length (in indexes)
	gen_len = chr_starts[-1]
	chr_starts = chr_starts[:-1]
	return chr_names, chr_sizes, chr_starts, gen_len


def read_wig(chip_file):
	#
	# return numpy array of wig file
	df = pd.read_table(chip_file, header=None, low_memory=False)
	#
	# any non-numeric fields (headers) become NAN, and each row has counted above it 
	# how many times NAN has been seen
	mask = pd.to_numeric(df.iloc[:,0], errors='coerce').isnull().cumsum()
	#
	# makes a list of dfs, where each one is renamed as the corresponding chromosome
	# resets the index for each df so it's correctly accessible
	dfs = [g[1:].rename(
		columns={0:(g.iloc[0].values[0]).split()[1].split('=')[1]}) \
		for i, g in df.groupby(mask)]
	chr_set = set(k.columns[0] for k in dfs) # set of all chromosomes
	chr_dict = {k:[] for k in chr_set}
	for idx, df_ in enumerate(dfs):
		chr_dict[df_.columns[0]].append(idx)
	#
	# returns a dictionary mapping the chromosome name to the array in float32 and flattened
	return {chrom: np.array(pd.concat([dfs[k] for k in chr_dict[chrom]]).reset_index(drop=True) \
						.reset_index(drop=True)).astype('float32').flatten() for chrom in chr_set}


class chip_data:
	def __init__(self, hdf5_file, group, dataset, cell_type, tracks=None):
		# Open up the file for reading upon initialization of object
		self.file = hdf5_file
		# group
		self.group = group
		#
		self.cell_type = cell_type
		# dataset
		self.dataset = dataset
		# resolution of data tracks
		self.resolution = int(self.dataset.attrs['resolution'])
		# the length of the genome in indices
		self.gen_len = int(self.dataset.attrs['gen_len'])
		# the names, sizes, and starting positions of all chromosomes
		self.chr_names = self.dataset.attrs['chr_names']
		self.chr_sizes = self.dataset.attrs['chr_sizes']
		self.chr_starts = self.dataset.attrs['chr_starts']
		# avail_tracks is a list of strings, each corresponding to a track in the dataset
		self.avail_tracks = self.dataset.attrs['tracks']
		# tracks is the ones actually used
		self.curr_tracks = tracks
		# find indexes which correspond to given tracks
		if tracks:
			self.parse_tracks(tracks)
		# the mean and std of the signal and log(signal+1)
		self.mean_sig = self.dataset.attrs['mean_sig']
		self.std_sig = self.dataset.attrs['std_sig']
		self.mean_log_sig = self.dataset.attrs['mean_log_sig']
		self.std_log_sig = self.dataset.attrs['std_log_sig']
	#
	def get_start(self, chromosome):
		# returns the starting position (in index-space) of the passed chromosome
		idx = int(self.chr_starts[self.chr_names == chromosome])
		return idx
	#
	def parse_tracks(self, tracks):
		# makes sure tracks is a list
		tracks = [k.lower() for k in tracks.split(',')]
		primary_marks = ['h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27ac']
		secondary_marks = ['h3k4me2', 'h3k9ac', 'h4k20me1', 'h3k79me2', 'h2az', 'dnase']
		self.track_idxs = []
		# if all tracks are asked for, track indexes are all 1s
		if self.cell_type == 'K562' and tracks == ['imputed']:
			cell_type = 'E123'
		else:
			cell_type = self.cell_type
		if 'all' in tracks:
			self.track_idxs = np.arange(len(self.avail_tracks))
		if 'all_hm' in tracks:
			for idx, avail_track in enumerate(self.avail_tracks):
				if ('histone'+cell_type+'h').lower().encode() in avail_track.lower():
					self.track_idxs.append(idx)
				if ('dnase'+cell_type).lower().encode() in avail_track.lower():
					self.track_idxs.append(idx)
		if 'primary_hm' in tracks:
			for idx, avail_track in enumerate(self.avail_tracks):
				for mark in primary_marks:
					if ('histone'+cell_type+mark).lower().encode() in avail_track.lower():
						self.track_idxs.append(idx)
		if 'secondary_hm' in tracks:
			for idx, avail_track in enumerate(self.avail_tracks):
				for mark in primary_marks + secondary_marks:
					if ('histone'+cell_type+mark).lower().encode() in avail_track.lower():
						self.track_idxs.append(idx)
				if ('dnase'+cell_type).lower().encode() in avail_track.lower():
					self.track_idxs.append(idx)
		if 'ctcf' in [k.lower() for k in tracks]:
			for idx, avail_track in enumerate(self.avail_tracks):
				if (cell_type+'ctcf').lower().encode() in avail_track.lower():
					self.track_idxs.append(idx)
		if 'imputed' in [k.lower() for k in tracks]:
			for idx, avail_track in enumerate(self.avail_tracks):
				if cell_type.lower().encode() in avail_track.lower() and b'imputed' in avail_track.lower():
					self.track_idxs.append(idx)
		if 'observed' in [k.lower() for k in tracks]:
			for idx, avail_track in enumerate(self.avail_tracks):
				if cell_type.lower().encode() in avail_track.lower() and not b'imputed' in avail_track.lower():
					self.track_idxs.append(idx)
		# for each track, find index in available tracks
		for curr_track in self.curr_tracks:
			curr_id = curr_track.split('/')[-1].split('.')[0]
			for idx, avail_track in enumerate(self.avail_tracks):
				if curr_id.encode() == avail_track:
					self.track_idxs.append(idx)
		#
		self.track_idxs = sorted(list(set(self.track_idxs)))
	#
	def bin_vals(self, arr, dim, step, method='mean', overhang=True):
		# Takes in an array, and averages values over some dimension with a given step size
		# Ex: bin_vals([1, 2, 3, 4, 5, 6], 0, 2, 'mean') will return [1.5, 3.5, 5.5]
		# Check that the given axis is divisible by the number of observations in a bin
		shape = list(np.shape(arr))
		# find new shape of intermediate array
		intermediate_shape = shape[:]
		intermediate_shape.insert(dim+1, step)
		if shape[dim] % step == 0:
			intermediate_shape[dim] = shape[dim] // step
		elif overhang:
			intermediate_shape[dim] = int(float(shape[dim]) / step + 1)
			arr = np.append(arr, np.zeros(intermediate_shape[dim] * step - shape[dim]))
		else:
			intermediate_shape[dim] = int(shape[dim] / step)
		# insert a new axis into the array, and reshape the array to split the given dimension into two
		arr = np.expand_dims(arr, dim+1)
		arr = arr.reshape(intermediate_shape)
		#
		if method == 'mean':
			return np.mean(arr, dim+1)
		if method == 'std':
			return np.std(arr, dim+1)
		if method == 'max':
			return np.max(arr, dim+1)
		if method == 'min':
			return np.min(arr, dim+1)
		if method == 'range':
			return np.ptp(arr, dim+1)
	#
	def z_score(self, in_array, means, stds):
		# takes in an array of size (features / length), means and stds of size (features), 
		# and returns a z-score
		# if a single position
		if len(np.shape(in_array)) == 1:
			return (in_array - means) / stds
		# If 2D array (not just a single vector)
		return (in_array - np.expand_dims(means, 1)) / np.expand_dims(stds, 1)
	#
	def get_data(self, chromosome, start_pos, end_pos, res=None, bin_method='mean', transform=None):
		chromosome = chromosome.encode('UTF-8')
		# Check if the chromosome exists
		if not chromosome in self.chr_names:
			raise NameError('chromosome ' + str(chromosome) + ' not found in database')
		# Check if start and end positons are valid by seeing if the chromosome's end position
		# is before the next chromosome starts
		# Calculate how many chromosomes start at or before the current one starts
		num_less = sum(self.chr_starts <= self.get_start(chromosome) )
		# Calculate how many chromosomes start at or before the position in question
		num_more = sum(self.chr_starts <= self.get_start(chromosome) + 
			(end_pos / self.resolution) )
		# If they're different, the position doesn't make sense
		if not num_less == num_more:
			print(chromosome, start_pos, end_pos)
			print("chromosome start idx:", self.get_start(chromosome))
			print("region end idx:", self.get_start(chromosome) + (end_pos / self.resolution))
		assert num_less == num_more
		# indexes in whole array. Rounds the positions down by resolution.
		start_idx = int(self.get_start(chromosome) + (start_pos / self.resolution))
		end_idx = int(self.get_start(chromosome) + (end_pos / self.resolution))
		# Check if new resolution is asked for. If so, change size of matrix.
		if not res == None and not res == self.resolution:
			out_array = self.bin_vals(self.dataset[start_idx:end_idx, self.track_idxs],
				dim=0, 
				step = int(res / self.resolution),
				method=bin_method)
		else:
			out_array = self.dataset[start_idx:end_idx, self.track_idxs]
		# return matching positions only for available tracks
		if transform == None or transform.lower() == 'none':
			return np.transpose(out_array)
		elif transform == "log":
			return np.log2(np.transpose(out_array) + 1)
		elif transform == "z":
			return self.z_score(np.transpose(out_array),
				self.mean_sig[self.track_idxs], 
				self.std_sig[self.track_idxs])
		elif transform == "log_z":
			try:
				np.log2(np.transpose(out_array)+1)
			except:
				print(out_array, start_idx, end_idx, self.track_idxs)
				sys.exit()
			return self.z_score(np.log2(np.transpose(out_array)+1),
				self.mean_log_sig[self.track_idxs], 
				self.std_log_sig[self.track_idxs])
		else:
			raise NameError("Unrecognized transformation for signal")
	#
	def add_track(self, track):
		# If wig, read wig and create dictionary of chromosome values
		print("Adding track " + track + " to hdf5 database")
		if '.wig' in track:
			# Check wig resolution and make sure it's compatible with db resolution
			with gopen(track) as infile:
				wig_resolution = int(infile.readline().split()[3].split('=')[1])
				assert self.resolution % wig_resolution == 0 # wig resolution must evenly divide db resolution
				bin_size = self.resolution // wig_resolution
			# read_wig returns a dictionary of arrays
			data_dict = read_wig(track)
			if bin_size != 1:
				for chrom in data_dict:
					data_dict[chrom] = self.bin_vals(data_dict[chrom], 0, bin_size, overhang=True)
			new_array = np.zeros(self.gen_len)
			for chrom, arr in data_dict.items():
				try:
					new_array[self.get_start(chrom):self.get_start(chrom)+len(arr)] = arr
				except:
					print(chrom, len(arr))
					sys.exit()
		# If bedgraph, convert to wig-type
		elif '.bedgraph' in track:
			new_array = bg_to_wig(track, self.gen_len, self.chr_names, 
				self.chr_starts, self.resolution)
		# Set all infinite values to the max value found
		max_val = np.max(new_array[new_array < np.inf])
		new_array[new_array == np.inf] = max_val
		# add array to data
		idx = len(self.avail_tracks)
		self.dataset.resize((self.gen_len, idx+1))
		self.dataset[:, idx] = new_array
		# add track to available tracks
		track_name = track.split('/')[-1]
		self.dataset.attrs['tracks'] = np.append(self.dataset.attrs['tracks'], track_name)
		self.avail_tracks = np.append(self.avail_tracks, track_name)
		# Calculate mean and std for signal
		mean_sig = np.mean(new_array[new_array > 0])
		std_sig = np.std(new_array[new_array > 0])
		# and log(signal+1)
		mean_log_sig = np.mean(np.log2(new_array[new_array > 0]+1))
		std_log_sig = np.std(np.log2(new_array[new_array > 0]+1))
		# add to attributes of dataset and to object
		self.mean_sig = np.append(self.mean_sig, mean_sig)
		self.dataset.attrs['mean_sig'] = np.append(self.dataset.attrs['mean_sig'], 
				mean_sig)
		self.std_sig = np.append(self.std_sig, std_sig)
		self.dataset.attrs['std_sig'] = np.append(self.dataset.attrs['std_sig'], 
				std_sig)
		self.mean_log_sig = np.append(self.mean_log_sig, mean_log_sig)
		self.dataset.attrs['mean_log_sig'] = np.append(self.dataset.attrs['mean_log_sig'], 
				mean_log_sig)
		self.std_log_sig = np.append(self.std_log_sig, std_log_sig)
		self.dataset.attrs['std_log_sig'] = np.append(self.dataset.attrs['std_log_sig'], 
				std_log_sig)


def main():
	'''
	This script takes in a list of ChIP-seq tracks (in bedgraph or fixed-step wig format) and 
	a resolution (in bp) and generates a hdf5-format file of the tracks. It writes each track
	sequentially to save memory. This file can then be accessed pseudo-randomly to pull out
	specific regions of the genome.

	The signal tracks will be averaged over the whole window. Additional attributes, such as
	mean, stdev, min and max of the signal, will also be saved, to allow for different output.
	Each track will also have attributes for the lab that created the data and the factor.

	A 'data' object can be then created, which retains all meta data about the file, including 
	the names of the tracks, the positions at which chromosomes start, and resolution. The 'data'
	object also has a method to retrieve a certain genomic locus for a specified number of tracks.
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('cell_type', help='Name of cell type')
	parser.add_argument('data_res', help='resolution desired for data (in bp)', type=int)
	parser.add_argument('genome', help='chromosome size file')
	parser.add_argument('data', help='tracks to add into database in either bedgraph\
		or wig format (either may be gzipped)', nargs='+')
	args = parser.parse_args()
	#
	# Open the db file, creating if necessary.
	hdfile = h5py.File('ChIP_db.hdf5', 'a')
	# If the required group does not exist, create. Otherwise, open.
	if not '/'+args.cell_type+'/'+str(args.data_res) in hdfile:
		grp = hdfile.create_group('/'+args.cell_type+'/'+str(args.data_res))
	else:
		grp = hdfile['/'+args.cell_type+'/'+str(args.data_res)]
	#
	# Determine number of tracks if initializing and genome size, creating a genome breakpoint file
	num_tracks = len(args.data)
	chr_names, chr_sizes, chr_starts, gen_len = make_chr_breakpoints(args.genome, args.data_res)
	#
	# Create dataset
	if 'chip_tracks' not in grp:
		dset = grp.create_dataset('chip_tracks', 
			shape=(gen_len, num_tracks), 
			chunks=(100, num_tracks),
			maxshape=(gen_len, None),
			dtype='float32',
			compression='gzip')
		dset.attrs['resolution'] = args.data_res
		dset.attrs['chr_names'] = chr_names
		dset.attrs['chr_sizes'] = chr_sizes
		dset.attrs['chr_starts'] = chr_starts
		dset.attrs['gen_len'] = gen_len
		dset.attrs['tracks'] = []
		dset.attrs['mean_sig'] = []
		dset.attrs['std_sig'] = []
		dset.attrs['mean_log_sig'] = []
		dset.attrs['std_log_sig'] = []
	# In this scenario, we will be adding tracks to an existing dataset
	else:
		dset = grp[u'chip_tracks']
	#
	# Create the dataset object for easy manipulation
	my_data = chip_data(hdfile, grp, dset, args.cell_type)
	# Go through each track given in arguments, and add to dataset if not already there
	for track in args.data:
		track_name = track.split('/')[-1].split('.')[0]
		if track_name not in list(my_data.avail_tracks):
			my_data.add_track(track)
		else:
			print('Skipping track '+ track)

if __name__ == '__main__':
	main()