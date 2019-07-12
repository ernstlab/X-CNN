#!/usr/bin/python3.6 -tt

import os, sys, argparse, random, gzip, h5py, keras, time, datetime, itertools
import pickle, numpy as np, pandas as pd
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate, concatenate
from keras.layers.core import Dense
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras import backend as K, regularizers
from chip_db import chip_data

def bool_parse(x):
	# Useful for parsing arguments to boolean variables
	if x.lower() in ['f','false','no','0']:
		return False
	elif x.lower() in ['t','true','yes','1']:
		return True
	raise ValueError('Need to provide boolean argument: '+str(x))


def timer(start, end):
	# Returns the total amount of time elapsed from start to end
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	sys.stdout.write("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)+'\n')


def gopen(file):
	if file[-3:] == '.gz':
		return gzip.open(file, 'rb')
	return open(file, 'rU')


def rounddown(val, step=1):
	# Round value down to closest step
	return int(val / step) * step


def roundup(val, step=1):
	# Round value up to closest step
	if val % step != 0:
		return (int(val / step) + 1) * step
	return val


def get_data(args, curr_db, chromosome, start_pos, end_pos):
	# Round start and end positions to resolution of data
	start_pos = rounddown(start_pos, args.data_res)
	end_pos = roundup(end_pos, args.data_res)
	if args.intn_len:
		return curr_db.get_data(
			chromosome=chromosome,
			start_pos=(start_pos+end_pos)//2 - (args.intn_len//2),
			end_pos=(start_pos+end_pos)//2 + (args.intn_len//2),
			res=args.data_res,
			transform=args.transform)
	# shape of data is num_tracks, length
	return curr_db.get_data(
			chromosome=chromosome,
			start_pos=start_pos,
			end_pos=end_pos,
			res=args.data_res,
			transform=args.transform)


def pad(matrix, axis=0, left_pad=None, right_pad=None, total_pad=None, neg=False):
	# Pads a matrix along a given axis with zeros (or whatever)
	input_shape = list(np.shape(matrix))
	if total_pad:
		left_pad = int(total_pad) // 2
		right_pad = total_pad - left_pad
	left_shape = input_shape[:]
	left_shape[axis] = left_pad
	right_shape = input_shape[:]
	right_shape[axis] = right_pad
	if not neg:
		return np.concatenate([np.zeros(left_shape), 
								matrix, 
								np.zeros(right_shape)], 
								axis=axis)
	elif neg:
		return np.concatenate([np.zeros(left_shape), 
								np.zeros(input_shape), 
								np.zeros(right_shape)], 
								axis=axis)


def scale(idxs, multiplier):
	out_idxs = np.array([])
	for m in range(multiplier):
		out_idxs = np.concatenate([out_idxs, np.array(idxs) * multiplier + m])
	return out_idxs


def get_idxs(final_model, chrom_idxs):
	# Returns indexes for train, validation, and test sets
	[train_idxs, val_idxs, test_idxs] = [[]] * 3
	if final_model:
		train_idxs = list(np.concatenate([chrom_idxs[k] for k in \
			['chr'+str(l) for l in list(range(1,8)) + list(range(10,23))]]).astype(int))
		val_idxs = list(np.concatenate([chrom_idxs['chr8'], chrom_idxs['chr9']]).astype(int))
	else:
		train_idxs = list(np.concatenate([chrom_idxs[k] for k in \
			['chr'+str(l) for l in list(range(2,8)) + list(range(10,23))]]).astype(int))
		val_idxs = list(np.concatenate([chrom_idxs['chr8'], chrom_idxs['chr9']]).astype(int))
		test_idxs = chrom_idxs['chr1']
	return [train_idxs, val_idxs, test_idxs]


def AUC(pts):
	AUC_sum = 0
	### This uses the trapezoidal rule
	for pos in range(len(pts[0]) - 1):
		AUC_sum += (pts[0,pos+1] - [pts[0,pos]]) * ((pts[1,pos+1] + [pts[1,pos]])/2)
	return AUC_sum


def generate_ROC(pos_stats, neg_stats):
	### pos_stats and neg_stats are both sorted from largest to smallest
	if not all([pos_stats[i] >= pos_stats[i+1] for i in range(len(pos_stats)-1)]):
		pos_stats = sorted(pos_stats, reverse=True)
	if not all([neg_stats[i] >= neg_stats[i+1] for i in range(len(neg_stats)-1)]):
		neg_stats = sorted(neg_stats, reverse=True)
	total_pos = float(len(pos_stats))
	total_neg = float(len(neg_stats))
	pts = [(0.,0.)]
	pos_idx=0
	neg_idx=0
	while pos_idx < total_pos and neg_idx < total_neg:
		if pos_stats[pos_idx] > neg_stats[neg_idx]:
			pos_idx += 1
		elif pos_stats[pos_idx] < neg_stats[neg_idx]:
			neg_idx += 1
		else:
			tied_value = pos_stats[pos_idx]
			while pos_idx < total_pos and pos_stats[pos_idx] == tied_value:
				pos_idx += 1
			while neg_idx < total_neg and neg_stats[neg_idx] == tied_value:
				neg_idx += 1
		pts.append((neg_idx/total_neg, pos_idx/total_pos))
	pts.append((1.,1.))
	pts = np.transpose(np.array(pts))
	return pts, AUC(pts)


def generate_PR(pos_stats, neg_stats):
	### pos_stats and neg_stats are both sorted from largest to smallest
	if not all([pos_stats[i] >= pos_stats[i+1] for i in range(len(pos_stats)-1)]):
		pos_stats = sorted(pos_stats, reverse=True)
	if not all([neg_stats[i] >= neg_stats[i+1] for i in range(len(neg_stats)-1)]):
		neg_stats = sorted(neg_stats, reverse=True)
	total_pos = float(len(pos_stats))
	total_neg = float(len(neg_stats))
	pts = [(0,0)]

	# Collapspe positive statistics to resolve any ties
	pos_stats_sorted = sorted(list(set(np.array(pos_stats).flatten())), reverse=True)
	for pos_stat in pos_stats_sorted:
		TP = np.sum(pos_stats >= pos_stat)
		FP = np.sum(neg_stats >= pos_stat)
		if np.sum(pos_stats == pos_stat) > 1:
			# Follow Davis & Goadrich (2006) to interpolate points
			prev_TP = np.sum(pos_stats > pos_stat)
			prev_FP = np.sum(neg_stats > pos_stat)
			skew = (FP - prev_FP) / (TP - prev_TP)
			for x in range(1, TP - prev_TP+1):
				recall = (prev_TP + x) / total_pos
				precision = (prev_TP + x) / (prev_TP + x + prev_FP + (skew*x)) 
				pts.append((recall, precision))
		else:
			# Calculate precision and recall directly
			recall = TP / total_pos
			precision = TP / (TP + FP)
			pts.append((recall, precision))
		
	pts.append((1., total_pos/(total_pos+total_neg)))
	pts = np.transpose(np.array(pts))
	return pts, AUC(pts)


def flip(input_array, neg_motif=False):
	# returns an array with identical size, but the orientation of which is flipped.
	# takes chip seq tracks and reverses order (all features but the last one)
	# takes CTCF motif track and reverses order and switches sign
	# shape of array is (num_features x position)
	if neg_motif:
		return np.concatenate([input_array[:-1, ::-1], 
			-np.expand_dims(input_array[-1, ::-1], 0)], axis=0)
	return input_array[:, ::-1]


def generate_negative_samples(args, interactions, interaction_length, pos_chrom_list, positions):
	# Generate negative samples with a specified ratio to the number of positive samples
	# Record distance between regions
	input_length = interaction_length // args.data_res
	distance_distr = []
	for idx, row in interactions.iterrows():
		distance_distr.append(row[4] - row[1])
		start_pos=(row[1]+row[2])//2 - interaction_length//2
		end_pos=(row[1]+row[2])//2 + interaction_length//2
	distance_distr = np.array(distance_distr)
	# Set up instance of database object
	hdfile = h5py.File(args.database, 'r')
	grp = hdfile['/'+args.cell_type+'/50']
	dset = grp[u'chip_tracks']
	curr_db = chip_data(hdfile, grp, dset, args.cell_type, args.tracks)
	data_neg = []
	outfile = open(args.out_dir + args.cell_type+'.interactions.neg.txt', 'w')
	num_interactions = len(interactions)
	neg_chrom_list = []
	# Take distribution of peak distances and draw random positions genome-wide
	for idx in range(len(interactions)):
		# Sample negative interactions to make a neg_ratio ratio
		while len(data_neg) <= args.neg_ratio * idx:
			# Choose distance first to guarantee following distribution
			while True:
				distance = np.random.choice(distance_distr)//args.data_res
				chromosome = pos_chrom_list[idx]
				distance = np.random.choice(distance_distr)//args.data_res
				try:
					start_left = np.random.choice(len(positions[chromosome]) \
												  - distance - input_length)
					break
				except ValueError:
					pass
			neg_chrom_list.append(chromosome)
			start_right = start_left + distance
			# Add to data
			data_neg.append( \
					[get_data(args, curr_db, chromosome, start_left*args.data_res, \
						(start_left+(interaction_length//args.data_res))*args.data_res),
					get_data(args, curr_db, chromosome, start_right*args.data_res, \
						(start_right+(interaction_length//args.data_res))*args.data_res)])
				# Write interaction out to file
			outfile.write('\t'.join([str(k) for k in \
				[chromosome, start_left*args.data_res, 
				(start_left+input_length)*args.data_res,
				chromosome, start_right*args.data_res, 
				(start_right+input_length)*args.data_res]]) + '\n')
			sys.stdout.write("\r{:5.1f}".format(float(idx) / (num_interactions) \
					* 100) + "% complete.")
	sys.stdout.write('\r100.0% complete!\n')
	outfile.close()
	data_neg = np.array(data_neg)
	for i in np.argwhere(np.isnan(data_neg)):
		data_neg[tuple(i)] = 0.
	# data array is of shape (num_samples, 2, num_tracks, input_length)
	fname = ':'.join([args.out_dir + args.cell_type, args.tracks, args.transform, str(args.data_res)+'bp', 
					  'neg.npy'])
	np.save(fname, data_neg)
	if args.low_mem:
		del data_neg
		data_neg = np.load(fname, mmap_mode='r')

	return data_neg, neg_chrom_list


def add_sample(left_array, right_array, data_left, data_right):
	left_array.append(data_left)
	right_array.append(data_right)
	left_array.append(data_right)
	right_array.append(data_left)
	return left_array, right_array


def generate_samples(args, input_length, data_pos=None, pos_idxs=None, data_neg=None, neg_idxs=None,
		ret_single=False, ret_neg_shuffle=False, side='both'):
	# This function takes in a list of data matrices, iterates over them and generates positve 
	# and negative samples by flipping and shuffling
	#
	# At each iteration, return a positive sample and its mirror
	# Also return negative samples and their mirrors
	while True:
		# Reset label and weight vectors
		label_vect = []
		weight_vect = []
		#
		# Need to return array of samples
		left_output_array = []
		right_output_array = []
		# Go over every index in the fold
		neg_list_idx = 0
		for list_idx, data_idx in enumerate(pos_idxs):

			if hasattr(data_pos, 'shape'):
				# Pull out interaction data
				data_left = data_pos[data_idx, 0]
				data_right = flip(data_pos[data_idx, 1])
				if ret_single:
					# Need to reshape to have length 1 (1 sample)
					output_shape = (1, input_length, np.shape(data_pos)[2])
					# yield data, label, weight, move to next sample
					if side == 'left' or side == 'both':
						yield (np.reshape([np.swapaxes(data_left, 1, 0)], output_shape),
								np.reshape([np.swapaxes(data_left, 1, 0)], output_shape))
					if side == 'right' or side == 'both':
						yield (np.reshape([np.swapaxes(data_right, 1, 0)], output_shape),
								np.reshape([np.swapaxes(data_right, 1, 0)], output_shape))
					continue
				else:
					left_output_array, right_output_array = add_sample(left_output_array, right_output_array,
																	   data_left, data_right)
					label_vect.extend([1] * 2)
					weight_vect.extend([1] * 2)

			# Negative samples are generated by shifting positive samples, 
			# using the given negative, and all their mirror images
			# Take the previous positive line as the "shuffled" interaction
			if ret_neg_shuffle:
				prev_data_left = data_pos[data_idx-1, 0]
				prev_data_right = flip(data_pos[data_idx-1, 1])
				left_output_array, right_output_array = add_sample(left_output_array, right_output_array, 
																   data_left, prev_data_right)
				left_output_array, right_output_array = add_sample(left_output_array, right_output_array, 
																   prev_data_left, data_right)
				label_vect.extend([0] * 4)
				weight_vect.extend([0.5] * 4)  # Weigh half as much because 2 ways to swap
			# 
			if hasattr(data_neg, 'shape'):
				# Add negative samples until proper balance is achieved
				while neg_list_idx <= list_idx * args.neg_ratio:
					data_left_neg = data_neg[neg_list_idx, 0]
					data_right_neg = flip(data_neg[neg_list_idx, 1])
					left_output_array, right_output_array = add_sample(left_output_array, right_output_array, 
																	   data_left_neg, data_right_neg)
					label_vect.extend([0] * 2)
					weight_vect.extend([1./args.neg_ratio] * 2)
					neg_list_idx += 1
			#
			# Yield entire matrix
			if (list_idx+1) % args.batch_size == 0 or list_idx == len(pos_idxs)-1:
				# pad matrices with zeros so all matrices are the same size
				if args.pad:
					left_output_array = pad(np.array(left_output_array), 
											axis=2, 
											total_pad=2*args.filter_len)
					right_output_array = pad(np.array(right_output_array), 
											axis=2, 
											total_pad=2*args.filter_len)
				yield ([np.swapaxes(left_output_array, 1, 2), \
						np.swapaxes(right_output_array, 1, 2)], \
						np.array(label_vect), \
						np.array(weight_vect))
				#
				# Reset label and weight vectors, array of samples
				left_output_array = []
				right_output_array = []
				label_vect = []
				weight_vect = []

		# After passing through all samples, shuffle indexes for next epoch
		# This will not apply when doing validation
		if pos_idxs: np.random.shuffle(pos_idxs)
		if neg_idxs: np.random.shuffle(neg_idxs)


def make_subnetwork(args, input_length, num_tracks, encoder=None):
	# Create input. Will have to be shape (1, input_length, num_tracks)
	if args.pad:
		chip_input = Input(shape=(input_length + 2*args.filter_len, num_tracks))
	else:
		chip_input = Input(shape=(input_length, num_tracks))
	if encoder:
		x = encoder(chip_input)
	else:
		x = chip_input
	x = Conv1D(
		filters=args.conv_kernel,
		kernel_size=args.filter_len,
		activation='relu',
		kernel_regularizer=regularizers.l1_l2(args.regularizer),
		use_bias=args.bias)(x)
	x = GlobalMaxPooling1D()(x)
	conv_model = Model(chip_input, x)
	return conv_model


def make_model(args, input_length, num_tracks, encoder_1=None, encoder_2=None):
	# Builds a model with required specifications. Takes optional input of pre-trained layers.

	# Create input. Will have to be shape (1, input_length, num_tracks)
	if args.pad:
		to_pad = 2*args.filter_len
	else:
		to_pad = 0
	chip_input = Input(shape=(input_length + to_pad, num_tracks))

	# Add encoder before convolution if available
	if args.shared_weights:
		conv_model = make_subnetwork(args, input_length, num_tracks, encoder_1)

		# Create left and right inputs
		input_left = Input(shape=(input_length + to_pad, num_tracks))
		input_right = Input(shape=(input_length + to_pad, num_tracks))

		# Create outputs
		output_left = conv_model(input_left)
		output_right = conv_model(input_right)
	else:
		conv_model1 = make_subnetwork(args, input_length, num_tracks, encoder_1)
		conv_model2 = make_subnetwork(args, input_length, num_tracks, encoder_2)

		# Create left and right inputs
		input_left = Input(shape=(input_length + to_pad, num_tracks))
		input_right = Input(shape=(input_length + to_pad, num_tracks))

		# Create outputs
		output_left = conv_model1(input_left)
		output_right = conv_model2(input_right)

	# Merge the two layers into one, concatenating them
	concatenated = concatenate([output_left, output_right])
	out = Dense(args.dense_kernel, 
		activation='relu', 
		kernel_regularizer=regularizers.l1_l2(args.regularizer),
		use_bias=args.bias)(concatenated)
	if args.extra_dense:
		out = Dense(args.dense_kernel,
			activation='relu',
			kernel_regularizer=regularizers.l1_l2(args.regularizer),
			use_bias=args.bias)(out)
	out = Dropout(args.dense_dropout)(out)
	out = Dense(1, 
		activation=K.sigmoid,
		kernel_regularizer=regularizers.l1_l2(args.regularizer),
		use_bias=args.bias)(out)

	# Make the final classifier and return it
	classification_model = Model([input_left, input_right], out)
	return classification_model


def train_autoencoder(args, data_pos, pos_idxs, side='both'):

	#	This autoencoder's purpose is to reduce the dimensionality of the data by finding the most
	#	common patterns in ChIP seq patterns. This does not look at spatial patterns; rather, it
	#	looks at individual windows, so those are what is fed to the autoencoder. It should
	#	work similarly to PCA, where strongly correlated features are joined in the same filter.
	num_samples, x, num_tracks, input_length = np.shape(data_pos)

	# input data placeholder
	input_data = Input(shape=(input_length, num_tracks))
	# "encoded" is the encoded representation of the input
	encoded = Conv1D(
		filters=args.autoencoder,
		kernel_regularizer=regularizers.l1_l2(args.regularizer),
		kernel_size=1,
		activation='relu',
		use_bias=args.bias)(input_data)
	# "decoded" is the lossy reconstruction of the input
	decoded = Conv1D(
		filters=num_tracks,
		kernel_regularizer=regularizers.l1_l2(args.regularizer),
		kernel_size=1,
		activation='relu',
		use_bias=args.bias)(encoded)

	# This takes an input to its reconstruction
	autoencoder = Model(inputs=input_data, outputs=decoded)
	# this model maps an input to its encoded representation
	encoder = Model(inputs=input_data, outputs=encoded)
	# compile the model using adadelta as an optimizer for best performance
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_logarithmic_error')
	if args.test:
		num_samples = 1000
		args.ae_epochs = 0
	# Check if we're sharing weights
	if side == 'both':
		num_steps = num_samples*2
	else:
		num_steps = num_samples

	# Train the model using a generator
	sys.stdout.write("*** Training autoencoder ***\n")
	autoencoder.fit_generator(
		generate_samples(args,
			data_pos=data_pos,
			pos_idxs=pos_idxs,
			input_length=input_length,
			ret_single=True,
			side=side),
		steps_per_epoch=min(20000,len(data_pos)),
		epochs=args.ae_epochs,
		verbose=int(args.verbose))

	return encoder


def train_and_test_model(args, 
		data_pos, pos_train_idxs=[], pos_val_idxs=[], pos_test_idxs=[], 
		data_neg=None, neg_train_idxs=[], neg_val_idxs=[], neg_test_idxs=[]):

	# Set length of input and number of training samples
	num_samples, x, num_tracks, input_length = np.shape(data_pos)
	interaction_length = input_length * args.data_res
	if args.neg_shuffle:
		data_neg = data_pos
		[neg_train_idxs, neg_val_idxs, neg_test_idxs] = [pos_train_idxs, pos_val_idxs, pos_test_idxs]

	# Train autoencoder on all samples (training, validation, and test) and build model
	encoder = None
	if args.autoencoder:
		if args.shared_weights:
			encoder = train_autoencoder(args, 
				data_pos=data_pos,
				pos_idxs=pos_train_idxs,
				side='both')
			final_model = make_model(args, input_length, num_tracks, encoder_1=encoder)
		else:
			encoder_1 = train_autoencoder(args, 
				data_pos=data_pos,
				pos_idxs=pos_train_idxs,
				side='left')
			encoder_2 = train_autoencoder(args, 
				data_pos=data_pos,
				pos_idxs=pos_train_idxs,
				side='right')
			final_model = make_model(args, input_length, num_tracks, encoder_1=encoder_1, encoder_2=encoder_2)

	# Use adadelta as an optimizer
	final_model.compile(optimizer='adadelta', loss='binary_crossentropy')

	# Early stopping and saving of the best model
	modelpath = args.out_dir+'final_model'+args.out_suff+'hdf5'
	modelCheckpoint = keras.callbacks.ModelCheckpoint(modelpath, save_best_only=True)
	earlyStopping = keras.callbacks.EarlyStopping(patience=args.early_stop)
	if args.test:
		args.num_epochs = 1
	# Fit model with a generator
	sys.stdout.write("*** Training CNN ***\n")
	final_model.fit_generator(
		generate_samples(args,
			data_pos=data_pos,
			pos_idxs=pos_train_idxs,
			data_neg=data_neg,
			neg_idxs=neg_train_idxs,
			input_length=input_length,
			ret_neg_shuffle=args.neg_shuffle,),
		validation_data=generate_samples(args,
			data_pos=data_pos,
			pos_idxs=pos_val_idxs,
			data_neg=data_neg,
			neg_idxs=neg_val_idxs,
			input_length=input_length,
			ret_neg_shuffle=args.neg_shuffle),
		steps_per_epoch=roundup(len(pos_train_idxs) / args.batch_size),
		validation_steps=roundup(len(pos_val_idxs) / args.batch_size),
		epochs=args.num_epochs,
		verbose=int(args.verbose),
		callbacks=[earlyStopping, modelCheckpoint],
		max_queue_size=96)

	# Generate predictions on test data to evaluate performance on held-out set
	sys.stdout.write('Predicting on validation data...\n')
	pos_prob_val = final_model.predict_generator(
		generate_samples(args,
			data_pos=data_pos,
			pos_idxs=pos_val_idxs,
			input_length=input_length),
		steps=roundup(len(pos_val_idxs) / args.batch_size),
		max_queue_size=96)
	neg_prob_val = final_model.predict_generator(
		generate_samples(args,
			data_pos=data_neg,
			pos_idxs=neg_val_idxs,
			input_length=input_length, 
			ret_neg_shuffle=args.neg_shuffle),
		steps=roundup(len(neg_val_idxs) / args.batch_size),
		max_queue_size=96)

	with open(args.out_dir+'pos_prob_val'+args.out_suff+'txt','w') as outfile:
		for prob in pos_prob_val:
			outfile.write(str(prob[0]) + '\n')
	with open(args.out_dir+'neg_prob_val'+args.out_suff+'txt','w') as outfile:
		for prob in neg_prob_val:
			outfile.write(str(prob[0]) + '\n')

	ROC_pts_val, AUROC_val = generate_ROC(pos_prob_val, neg_prob_val)
	PR_pts_val, AUPR_val = generate_PR(pos_prob_val, neg_prob_val)
	np.savetxt(args.out_dir + 'ROC_val' + args.out_suff + 'txt.gz', np.transpose(ROC_pts_val))
	np.savetxt(args.out_dir + 'AUROC_val' + args.out_suff + 'txt', np.transpose(AUROC_val))
	np.savetxt(args.out_dir + 'PR_val' + args.out_suff + 'txt.gz', np.transpose(PR_pts_val))
	np.savetxt(args.out_dir + 'AUPR_val' + args.out_suff + 'txt', np.transpose(AUPR_val))
	sys.stderr.write('Final val AUROC: ' + str(float(AUROC_val))+ '\n')

	# If we train for final results, save the model
	if not args.final:
		# Generate predictions on test data to evaluate performance on held-out set
		sys.stdout.write('Predicting on test data...\n')
		pos_prob_test = final_model.predict_generator(
			generate_samples(args,
				data_pos=data_pos,
				pos_idxs=pos_test_idxs,
				input_length=input_length),
			steps=roundup(len(pos_test_idxs) / args.batch_size),
			max_queue_size=96)
		neg_prob_test = final_model.predict_generator(
			generate_samples(args,
				data_pos=data_neg,
				pos_idxs=neg_test_idxs,
				input_length=input_length, 
				ret_neg_shuffle=args.neg_shuffle),
			steps=roundup(len(neg_test_idxs) / args.batch_size),
			max_queue_size=96)
		

		with open(args.out_dir+'pos_prob_test'+args.out_suff+'txt','w') as outfile:
			for prob in pos_prob_test:
				outfile.write(str(prob[0]) + '\n')
		with open(args.out_dir+'neg_prob_test'+args.out_suff+'txt','w') as outfile:
			for prob in neg_prob_test:
				outfile.write(str(prob[0]) + '\n')

		ROC_pts_test, AUROC_test = generate_ROC(pos_prob_test, neg_prob_test)
		PR_pts_test, AUPR_test = generate_PR(pos_prob_test, neg_prob_test)
		np.savetxt(args.out_dir + 'ROC_test' + args.out_suff + 'txt.gz', np.transpose(ROC_pts_test))
		np.savetxt(args.out_dir + 'AUROC_test' + args.out_suff + 'txt', np.transpose(AUROC_test))
		np.savetxt(args.out_dir + 'PR_test' + args.out_suff + 'txt.gz', np.transpose(PR_pts_test))
		np.savetxt(args.out_dir + 'AUPR_test' + args.out_suff + 'txt', np.transpose(AUPR_test))
		sys.stderr.write('Final test AUROC: ' + str(float(AUROC_test))+ '\n')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('cell_type', help='name of cell type')
	parser.add_argument('data_res', help='resolution of input data (in bp)', type=int)
	parser.add_argument('interaction', help='name of interaction file')
	parser.add_argument('--pos_data', help='name of positive data file. Exclusive with --database', default=None)
	parser.add_argument('--neg_shuffle', help='shuffle positive data for negative set', default=False, type=bool_parse)
	parser.add_argument('--neg_ratio', help='number of negative samples to generate per positive sample', default=1., type=float)
	parser.add_argument('--neg_data_file', help='name of negative data file', default=None)
	parser.add_argument('--neg_intn_file', help='name of negative interaction file', default=None)
	parser.add_argument('--database', help='name of database', default=None)
	parser.add_argument('--transform', help='data transform [none, log, z, log_z]', default='log')
	parser.add_argument('--low_mem', help='use low-memory mode to handle data', default=False, action='store_true')
	parser.add_argument('--pad', help='pad data with zeros of width of convolution', default=False, type=bool_parse)
	parser.add_argument('--tracks', help='tracks to learn on, comma separated (can include primary_hm, secondary_hm, ctcf)', default='observed')
	parser.add_argument('--chr_size', help='chromosome size file')
	parser.add_argument('--autoencoder', help='use an autoencoder with this number of kernels (0 for no autoencoder)', default=16, type=int)
	parser.add_argument('--shared_weights', help='use shared weights for autoencoder and convolutional layers', default=True, type=bool_parse)
	parser.add_argument('--ae_epochs', help='epochs to train autoencoder', default=1, type=int)
	parser.add_argument('--filter_len', help='length of filter in convolutional layer', default=10, type=int)
	parser.add_argument('--conv_kernel', help='number of kernels to use in convolutional layer', default=16, type=int)
	parser.add_argument('--dense_kernel', help='numer of kernels to use in dense layer', default=16, type=int)
	parser.add_argument('--dense_dropout', help='probability with which to set dense layer kernel to 0', default=0, type=float)
	parser.add_argument('--extra_dense', help='use extra dense layer in model', default=False, action='store_true')
	parser.add_argument('--bias', help='include bias in learning of dense and logistic layer', default=True, type=bool_parse)
	parser.add_argument('--regularizer', help='strength of regularizer', default=0.0001, type=float)
	parser.add_argument('--num_epochs', help='number of epochs to train', default=500, type=int)
	parser.add_argument('--early_stop', help='number of epochs to wait once validation loss starts increasing', default=5)
	parser.add_argument('--final', help='do not withhold data for testing and save final model', default=False, action='store_true')
	parser.add_argument('--out_suff', help='additional suffix to use in naming files', default='.')
	parser.add_argument('--out_dir', help='directory in which to save files', default='.')
	parser.add_argument('--verbose', help='verbose training status', default=True, type=bool_parse)
	parser.add_argument('--test', help='test code by training for a shorter time', default=False, action='store_true')
	parser.add_argument('--seed', help='random seed', default=1, type=int)
	parser.add_argument('--intn_len', help='extend peaks to this size', default=None, type=int)
	parser.add_argument('--batch_size', help='batch size', default=24, type=int)

	args = parser.parse_args()
	if args.out_dir[-1] != '/':
		args.out_dir += '/'
	if args.out_suff[0] != '.':
		args.out_suff = '.' + args.out_suff
	if args.out_suff[-1] != '.':
		args.out_suff += '.'
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)
	# start timer
	start_time = time.time()
	seed = args.seed
	np.random.seed(seed)

	# Read in interaction file
	sys.stderr.write('Reading and analyzing interacting regions.\n')
	with gopen(args.interaction) as infile:
		toskip = 0
		if str(infile.readline()).startswith('chr1\tx1\tx2'):
			toskip = 1
	interactions = pd.read_csv(args.interaction, usecols=range(6), sep='\t', 
		names=['chrA', 'startA','endA','chrB','startB','endB'], skiprows=toskip)
	# If the chromosome name does not start with 'chr', add it in front
	if interactions.iloc[0]['chrA'] == '1':
		interactions[['chrA']] = 'chr' + interactions[['chrA']]
		interactions[['chrB']] = 'chr' + interactions[['chrB']]
	num_interactions = len(interactions)
	if args.intn_len:
		interaction_length = args.intn_len
	else:
		interaction_length = interactions.iloc[0]['endA'] - interactions.iloc[0]['startA']
	input_length = interaction_length // args.data_res

	# Find positions of single interacting loci
	if args.neg_ratio:
		with open(args.chr_size, 'rU') as infile:
			# initialize vectors
			positions = {}
			for line in infile:
				field = line.split()
				if not '_' in field[0]:
					positions[field[0]] = np.zeros(int(field[1])//args.data_res + 1)
		for idx, row in interactions.iterrows():
			# Left interacting region
			start_pos = (row[1]+row[2])//2 - interaction_length//2
			end_pos = start_pos + interaction_length
			positions[row[0]][start_pos//args.data_res:end_pos//args.data_res] = 1
			# Right interacting region
			start_pos = (row[4]+row[5])//2 - interaction_length//2
			end_pos = start_pos + interaction_length
			positions[row[3]][start_pos//args.data_res:end_pos//args.data_res] = 1
		# Save chromosome lengths
		chr_lengths = {}
		for chrom in ['chr' + str(k) for k in list(range(1,23)) + ['X']]:
			chr_lengths[chrom] = len(positions[chrom])

	# Save each chromosome's idxs to make it easier to split data
	pos_chrom_idxs = {'chr'+str(k):[] for k in list(range(1,23)) + ['X']}
	pos_chrom_list = []
	for idx, row in interactions.iterrows():
		pos_chrom_idxs[row[0]].append(int(idx))
		pos_chrom_list.append(row[0])

	if args.pos_data:
		sys.stderr.write('Reading positive data.\n')
		if args.low_mem:
			data_pos = np.load(args.pos_data, mmap_mode='r')
		else:
			data_pos = np.load(args.pos_data)
	elif os.path.isfile(args.out_dir + args.cell_type + ':' + args.tracks + ':' + args.transform + \
						':' + str(args.data_res)+'bp.npy'):
		sys.stderr.write('Reading existing positive data.\n')
		data_pos = np.load(args.out_dir + args.cell_type + ':' + args.tracks + ':' + args.transform + \
						':' + str(args.data_res)+'bp.npy')
	else:
		sys.stdout.write("Generating positive samples.\n")
		# Initialize chip seq data
		hdfile = h5py.File(args.database, 'r')
		grp = hdfile['/'+args.cell_type+'/50']
		dset = grp[u'chip_tracks']
		# Set up instance of database object
		curr_db = chip_data(hdfile, grp, dset, args.cell_type, args.tracks)
		# Pull out all relevant regions from the database and save for reuse
		data_pos = []
		for idx, row in interactions.iterrows():
			data_pos.append([get_data(args, curr_db, row[0], row[1], row[2]),
							get_data(args, curr_db, row[3], row[4], row[5])])
			# Print progress
			sys.stdout.write("\r{:5.1f}".format(float(idx) / num_interactions * 100) \
					+ "% complete")
		sys.stdout.write('\r100.0% complete!\n')
		data_pos = np.array(data_pos)
		# Check for nan
		for i in np.argwhere(np.isnan(data_pos)):
			data_pos[tuple(i)] = 0.
		# data array is of shape (num_samples, 2, num_tracks, input_length)
		np.save(args.out_dir + args.cell_type+':'+args.tracks+':'+args.transform+':'+\
				str(args.data_res)+'bp.npy', data_pos)
		if args.low_mem:
			del data_pos
			data_pos = np.load(args.out_dir + args.cell_type+':'+args.tracks+':'+\
				args.transform+':'+str(args.data_res)+'bp.npy', mmap_mode='r')

	if args.neg_shuffle:
		data_neg = None

	elif args.neg_data_file:
		sys.stderr.write('Reading negative data.\n')
		if args.low_mem:
			data_neg = np.load(args.neg_data_file, mmap_mode='r')
		else:
			data_neg = np.load(args.neg_data_file)
		args.neg_ratio = len(data_neg) / len(data_pos)
		# Read in interactions to get chromosomes
		neg_interactions = pd.read_csv(args.neg_intn_file, usecols=range(6), sep='\t', 
			names=['chrA', 'startA','endA','chrB','startB','endB'])
		neg_chrom_list = list(neg_interactions['chrA'])
		neg_chrom_idxs = {'chr'+str(k):[] for k in list(range(1,23)) + ['X']}
		for idx, chrom in enumerate(neg_chrom_list):
			neg_chrom_idxs[chrom].append(idx)

	else:
		sys.stdout.write("Generating negative samples.\n")
		data_neg, neg_chrom_list = generate_negative_samples(args, interactions, interaction_length, 
															 pos_chrom_list, positions)

		neg_chrom_idxs = {'chr'+str(k):[] for k in list(range(1,23)) + ['X']}
		for idx, chrom in enumerate(neg_chrom_list):
			neg_chrom_idxs[chrom].append(idx)

	[pos_train_idxs, pos_val_idxs, pos_test_idxs] = get_idxs(args.final, pos_chrom_idxs)
	[neg_train_idxs, neg_val_idxs, neg_test_idxs] = get_idxs(args.final, neg_chrom_idxs)

	np.random.shuffle(pos_train_idxs)

	# Train and test model
	sys.stdout.write("Starting training.\n")
	sys.stdout.flush()
	train_and_test_model(args, data_pos, data_neg=data_neg, 
			pos_train_idxs=pos_train_idxs, pos_val_idxs=pos_val_idxs, pos_test_idxs=pos_test_idxs,
			neg_train_idxs=neg_train_idxs, neg_val_idxs=neg_val_idxs, neg_test_idxs=neg_test_idxs)

	sys.stdout.write('Finished in ')
	sys.stdout.flush()
	timer(start_time, time.time())

if __name__ == '__main__':
	main()