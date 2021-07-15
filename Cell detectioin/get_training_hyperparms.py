# -*- coding: utf-8 -*-
"""
Created on 30/03/2020

@author: yhagos
"""
import os
import argparse


def get_hyper_params():
	args = get_args_()

	training_params = dict()
	if args.cluster:
		const_params = {'valid_image_dir': os.path.join(args.data_dir, 'val_images.npy'),
				  'valid_label_dir': os.path.join(args.data_dir, 'val_label_numcells.npz'),
				  'train_image_dir': os.path.join(args.data_dir, 'train_images_aug.npy'),
				  'train_label_dir': os.path.join(args.data_dir, 'train_label_numcells.npz'),
				  'output_dir': args.output,
				  'cell_counter_model_dir': args.counter_dir,
				  'pretrained': True,
						'input_shape':(224, 224, 3),
				  'epochs': 800,
				  }
	else:
		data_dir = r'D:\Projects\Vectra_deep_learning\Data\20200710\data\train\Rect_Cell\polyscope-rectangles\npy_files'
		output = r'D:\Projects\Vectra_deep_learning\Data\20200710\data\train\Rect_Cell\polyscope-rectangles\models'
		counter_dir= r'PretrainedCellCounter\bestweights.h5'
		const_params = {'valid_image_dir': os.path.join(data_dir, 'val_images.npy'),
				  'valid_label_dir': os.path.join(data_dir, 'val_label_numcells.npz'),
				  'train_image_dir': os.path.join(data_dir, 'train_images_aug.npy'),
				  'train_label_dir': os.path.join(data_dir, 'train_label_numcells.npz'),
				  'output_dir': output,
				  'cell_counter_model_dir': counter_dir,
				  'pretrained': True,
						'input_shape': (224, 224, 3),
				  'epochs': 800,
				  }

	learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
	backends = ['inception']  # 'vgg'
	optimizer_types = ['adam']
	loss_types = ['dice_and_count']  # ['dice', 'dice_and_count']
	bases = [16]
	batch_sizes = [32]
	depths = [3, 4, 5]
	n = 0
	for loss_type in loss_types:
		for optimizer_type in optimizer_types:
			for lr in learning_rates:
				for base in bases:
					for backend in backends:
						for batch_size in batch_sizes:
							for depth in depths:
								params = dict()
								params['loss_type'] = loss_type
								params['learning_rate'] = lr
								params['base'] = base
								params['batch_size'] = batch_size
								params['optimizer_type'] = optimizer_type
								params['depth'] = depth
								params['backend'] = backend

								training_params[n] = {**params, **const_params}

								n += 1

	return training_params

def get_args_():

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', dest='data_dir', default='', help='directory of training csv file')
	parser.add_argument('-o', '--output', dest='output', default='', help='directory to save training output')
	parser.add_argument('-c', '--counter_dir', dest='counter_dir', default='', help='directory to save training output')
	parser.add_argument('--cluster', dest='cluster', default=False, action='store_true', help='boolean to indicate ')

	args = parser.parse_args()

	return args
