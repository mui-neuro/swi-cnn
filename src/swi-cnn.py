#! /usr/bin/env python

import os
import argparse
import inspect

import numpy as np

from os.path import join, abspath
from importlib import reload
from sklearn.model_selection import KFold

try:
    import utils as ut
    import generator as gn
except:
    import src.utils as ut
    import src.generator as gn

# Actively reload in case changes were made
reload(ut)
reload(gn)

""" Define global parameters """

config = dict()

config['overwrite'] = False
config['verbose'] = 1

# Models and general training params
models = ['unet', 'vnet', 'unetpp']

config['n_epochs'] = 1000
config['batch_size'] = 5
n_folds = 5

# Patches parameters
config['conform_shape'] = [240, 320, 223]
config['downsample_shape'] = [64, 64, 64]
config['down_res'] = [3, 3, 3]
config['patch_shape'] = [64, 64, 64]

# Labels
config['labels'] = {'rn_r': 1, 'rn_l': 2,
                    'sn_r': 3, 'sn_l': 4,
                    'den_r': 5, 'den_l': 6,
                    'stn_r': 7, 'stn_l': 8}


if __name__ == '__main__':

    # Define main directory as parent directory relative to current file
    config['main_dir'] = os.path.dirname(abspath(join(
                            inspect.getfile(inspect.currentframe()),
                            os.pardir)))

    parser = argparse.ArgumentParser(description="Perform segmentation.")
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument("-cwd", "--cwd", type=str, default=config['main_dir'],
                       help="Current (Root) Working directory.")
    mutex.add_argument("-s", "--subject", type=str, default=None,
                       help="Subject to be processed.")
    mutex.add_argument("-sl", "--subjects_list", type=str, default=None,
                       help="List of all subject to be processed.")
    parser.add_argument("-p", "--preproc", action="store_true",
                        help='Perform preprocessing of SWI data')
    parser.add_argument("-ir", "--iso_resolution", type=float, default=0.6875,
                        help='Isotropic resolution for preprocessing.')
    parser.add_argument("-lr", "--label_regions", action="store_true",
                        help='Label regions using final segmentation models.')
    parser.add_argument("-a", "--augment", action="store_true",
                        help='Perform data augmentation.')
    parser.add_argument("-d", "--downsample", action="store_true",
                        help='Perform downsampling of augmented data.')
    parser.add_argument("-pp", "--preallocate_patch", action="store_true",
                        help='Preallocate batches for patch models.')
    parser.add_argument("-pcv", "--patch_cv", action="store_true",
                        help='Perform cross-validation of patch models.')
    parser.add_argument("-pf", "--patch_final", action="store_true",
                        help='Train the final patch models using all available data.')
    parser.add_argument("-ps", "--preallocate_segmentation", action="store_true",
                        help='Preallocate batches for segmentation models.')
    parser.add_argument("-scv", "--segmentation_cv", action="store_true",
                        help='Perform cross-validation of segmentaion models.')
    parser.add_argument("-ecv", "--ensemble_cv", action="store_true",
                        help='Compute the ensemble segmentation for the CV test data.')
    parser.add_argument("-sf", "--segmentation_final", action="store_true",
                    help='Train the final segmentation models using all available data.')
    parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1,
                        help="Number of parallel jobs. Default 1.")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Level of verbosity. 0: silent, 1: minimal (default), 2: detailed.")
    args = parser.parse_args()


# Assign main directories
config['swi_dir'] = join(config['main_dir'], 'swi')
config['preproc_dir'] = join(config['main_dir'], 'preproc')
config['augment_dir'] = join(config['main_dir'], 'augment')
config['downsample_dir'] = join(config['main_dir'], 'downsample')

config['n_jobs'] = args.n_jobs

# Post process sujetcs arguments
if args.subject:
    subjects = np.array([args.subject])


if args.subjects_list:
    with open(args.subjects_list, 'r') as f:
        subjects = np.array([subject.strip() for subject in f.readlines()])


if args.preproc:

    if args.verbose > 0:
        print('Preprocessing data...')

    config['dataset'] = {'subjects': subjects}

    ut.conform_dir_to_iso(config['swi_dir'], config['preproc_dir'], subjects,
                          args.iso_resolution,
                          conform_shape=config['conform_shape'],
                          conform_strides=False)

    if args.verbose > 0:
        print('Creating brainmasks...')
    ut.create_swi_brainmask(config)

    if args.verbose > 0:
        print('Performing N4 bias field correction...')
    ut.N4_bias_correction(config)

    if args.verbose > 0:
        print('Normalizing data (using SRS)...')
    ut.srs_normalize(config)


if args.augment:

    if args.verbose > 0:
        print('Performing data augmentation...')

    config['dataset'] = {'subjects': subjects,
                         'data': [join('preproc', '%s_brain_norm.nii.gz') % s
                                  for s in subjects],
                         'truth': [join(config['main_dir'], 'manual_labels',
                                        '%s.nii.gz') % s for s in subjects]}

    if args.verbose > 0:
        print('Allocating test data')
    config['augment'] = False
    gn.augment_dataset(config)

    if args.verbose > 0:
        print('Augmenting training data %i times.' % config['n_augment'])
    config['augment'] = True
    gn.augment_dataset(config)


if args.downsample:

    config['dataset'] = {'subjects': subjects}

    if args.verbose > 0:
        print('Downsampling test data...')
    config['augment'] = False
    gn.downsample_dataset(config)

    if args.verbose > 0:
        print('Downsampling training data...')
    config['augment'] = True
    gn.downsample_dataset(config)


if args.preallocate_patch:

    if args.verbose > 0:
        print('Preallocating batches for patch models...')

    config['batches_dir'] = join(config['main_dir'], 'batches', 'patch')

    config['dataset'] = {'subjects': subjects}

    if args.verbose > 0:
        print('Creating test data...')
    config['augment'] = False
    gn.preallocate_patch_batches(config)

    if args.verbose > 0:
        print('Preallocating batches for patch models...')
    config['augment'] = True
    gn.preallocate_patch_batches(config)


if args.patch_cv:

    if args.verbose > 0:
        print('Performing cross-validation of patch models...')

    # Generate splits for CV
    skf = KFold(n_splits=n_folds, shuffle=False)
    folds = [folds for folds in skf.split(subjects)]

    for n_fold, (training, test) in enumerate(folds):

        if args.verbose > 0:
            print('Processing fold %i' % n_fold)

        print('Training data')
        print(subjects[training])
        print('Test data')
        print(subjects[test])

        # Batch params
        config['training'] = subjects[training]
        config['test'] = subjects[test]
        config['test_batch_random'] = False

        # Model parameters
        config['input_shape'] = [1] + config['downsample_shape']
        config['batch_norm'] = True

        # Optimization parameters
        config['initial_learning_rate'] = 1e-2
        config['end_learning_rate'] = 1e-5
        config['decay_steps'] = 125

        for model in models:
            model_dir = join(config['main_dir'], 'models', 'patch', model)
            fold_dir = join(model_dir, 'n_fold-%i' % n_fold)
            ut.assert_dir(model_dir)
            ut.assert_dir(fold_dir)
            for region in config['labels'].keys():
                config['batches_x_dir'] = config['downsample_dir']
                config['batches_y_dir'] = join(config['main_dir'],
                                               'batches', 'patch', region)
                model_path = join(fold_dir, region)
                ut.train_model(config, region, model, model_path)


if args.patch_final:

    if args.verbose > 0:
        print('Training final patch models using all available data...')

    # Batch params
    config['training'] = subjects
    config['test'] = subjects
    config['test_batch_random'] = True

    # Model parameters
    config['input_shape'] = [1] + config['downsample_shape']
    config['batch_norm'] = True

    # Optimization parameters
    config['initial_learning_rate'] = 1e-2
    config['end_learning_rate'] = 1e-5
    config['decay_steps'] = 125

    for model in models:
        model_dir = join(config['main_dir'], 'models', 'patch', model)
        final_dir = join(model_dir, 'final')
        ut.assert_dir(model_dir)
        ut.assert_dir(final_dir)
        for region in config['labels'].keys():
            config['batches_x_dir'] = config['downsample_dir']
            config['batches_y_dir'] = join(config['main_dir'], 'batches',
                                           'patch', region)
            model_path = join(final_dir, region)
            ut.train_model(config, region, model, model_path)


if args.preallocate_segmentation:

    for model in models:

        config['batches_dir'] = join(config['main_dir'], 'batches', 'region',
                                     model)
        ut.assert_dir(config['batches_dir'])

        config['patch_models_dir'] = join(config['main_dir'], 'models',
                                          'patch', model, 'final')

        config['dataset'] = {'subjects': subjects}

        print('Creating test data...')
        config['augment'] = False
        gn.preallocate_region_batches(config)

        print('Processing augmented data for regions...')
        config['augment'] = True
        gn.preallocate_region_batches(config)


if args.segmentation_cv:

    # Generate splits for CV
    skf = KFold(n_splits=n_folds, shuffle=False)
    folds = [folds for folds in skf.split(subjects)]

    for n_fold, (training, test) in enumerate(folds):

        print('Training data')
        print(subjects[training])
        print('Test data')
        print(subjects[test])

        # Batch params
        config['training'] = subjects[training]
        config['test'] = subjects[test]
        config['test_batch_random'] = False

        # Model parameters
        config['input_shape'] = [1] + config['patch_shape']
        config['batch_norm'] = True

        # Optimization parameters
        config['initial_learning_rate'] = 1e-2
        config['end_learning_rate'] = 1e-5
        config['decay_steps'] = 125

        for model in models:

            # Train models
            model_dir = join(config['main_dir'], 'models', 'region', model)
            fold_dir = join(model_dir, 'n_fold-%i' % n_fold)
            ut.assert_dir(model_dir)
            ut.assert_dir(fold_dir)

            for region in config['labels'].keys():
                config['batches_x_dir'] = join(config['main_dir'],
                                               'batches', 'region', model,
                                               region)
                config['batches_y_dir'] = join(config['main_dir'],
                                               'batches', 'region', model,
                                               region)
                model_path = join(fold_dir, region)
                ut.train_model(config, region, model, model_path)

            # Labeling test data
            config['dataset'] = {'subjects': subjects[test],
                                 'data': [join('preproc',
                                          '%s_brain_norm.nii.gz') % s
                                          for s in subjects[test]],
                                 'truth': [join(config['main_dir'],
                                           'manual_labels', '%s.nii.gz') % s
                                           for s in subjects[test]]}

            config['labels_dir'] = join(config['main_dir'], 'models', 'region',
                                        model, 'test_labels')
            config['patch_models_dir'] = join(config['main_dir'],
                                              'models', 'patch', model,
                                              'n_fold-%i' % n_fold)
            config['region_models_dir'] = join(config['main_dir'],
                                               'models', 'region', model,
                                               'n_fold-%i' % n_fold)
            ut.label_regions(config, unite=True)


if args.ensemble_cv:

    config['dataset'] = {'subjects': subjects}
    config['models_dir'] = [join(config['main_dir'], 'models', 'region',
                                 model, 'test_labels') for model in models]
    config['ensemble_dir'] = join(config['main_dir'], 'models', 'region',
                                  'ensemble', 'test_labels')
    ut.ensemble_regions(config)


if args.segmentation_final:

    # Batch params
    config['training'] = subjects
    config['test'] = subjects
    config['test_batch_random'] = True

    # Model parameters
    config['input_shape'] = [1] + config['patch_shape']
    config['batch_norm'] = True

    # Optimization parameters
    config['initial_learning_rate'] = 1e-2
    config['end_learning_rate'] = 1e-5
    config['decay_steps'] = 125

    for model in models:

        model_dir = join(config['main_dir'], 'models', 'region', model)
        final_dir = join(model_dir, 'final')
        ut.assert_dir(model_dir)
        ut.assert_dir(final_dir)

        for region in config['labels'].keys():
            config['batches_x_dir'] = join(config['main_dir'],
                                           'batches', 'region', model, region)
            config['batches_y_dir'] = join(config['main_dir'],
                                           'batches', 'region', model, region)
            model_path = join(final_dir, region)
            ut.train_model(config, region, model, model_path)


if args.label_regions:

    config['dataset'] = {'subjects': subjects,
                         'data': [join('preproc', '%s_brain_norm.nii.gz') % s
                                  for s in subjects]}

    # Label dataset
    for model in models:
        config['labels_dir'] = join(config['main_dir'], 'labels', model)
        config['patch_models_dir'] = join(config['main_dir'], 'models',
                                          'patch', model, 'final')
        config['region_models_dir'] = join(config['main_dir'], 'models',
                                           'region', model, 'final')
        ut.label_regions(config, unite=True)

    # Create ensemble
    config['labels_dir'] = join(config['main_dir'], 'labels')
    config['models_dir'] = [join(config['main_dir'], 'labels', model)
                            for model in models]
    config['ensemble_dir'] = join(config['main_dir'], 'labels', 'ensemble')
    ut.ensemble_regions(config)


""" Data augmentation training """
total_batches = config['n_epochs']*config['batch_size']
batches_per_augment = len(subjects)*2*(n_folds-1)/n_folds
config['n_augment'] = np.ceil(total_batches/batches_per_augment).astype(int)
