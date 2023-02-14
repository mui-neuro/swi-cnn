#! /usr/bin/env python

import ants
import argparse
import os

import numpy as np
import scipy as sp

from os.path import join
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

patch_models = ['unet', 'vnet', 'unetpp', 'fc_dense_net', 'dilated_fc_dense_net']

models = ['unet', 'vnet', 'unetpp', 'fc_dense_net', 'dilated_fc_dense_net']

config['n_epochs'] = 100
config['batch_size'] = 4
n_folds = 5

# Patches parameters
config['conform_shape'] = [240, 320, 223]
config['downsample_shape'] = [64, 64, 64]
config['down_res'] = [3, 3, 3]
config['patch_shape'] = [64, 64, 64]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument("-s", "--subject", type=str, default=None,
                       help="Subject to be processed.")
    mutex.add_argument("-sl", "--subjects_list", type=str, default=None,
                       help="List of all subject to be processed.")
    parser.add_argument("-p", "--preproc", action="store_true",
                        help='Perform preprocessing of SWI data')
    parser.add_argument("-fres", "--full_resolution", type=float,
                        default=0.6875,
                        help='Isotropic resolution for preprocessing.')
    parser.add_argument("-lp", "--label_patches", action="store_true",
                        help='Label patches to be used for patch extraction.')
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
    parser.add_argument("-pcv_mixed", "--patch_cv_mixed", action="store_true",
                        help='Perform cross-validation of patch models.')
    parser.add_argument("-pf", "--patch_final", action="store_true",
                        help='Train the final patch models using all available data.')
    parser.add_argument("-ps", "--preallocate_segmentation", action="store_true",
                        help='Preallocate batches for segmentation models.')
    parser.add_argument("-pms", "--preallocate_mixed_segmentation", action="store_true",
                        help='Preallocate batches for segmentation models.')
    parser.add_argument("-scv", "--segmentation_cv", action="store_true",
                        help='Perform cross-validation of segmentaion models.')
    parser.add_argument("-scv_mixed", "--segmentation_cv_mixed", action="store_true",
                        help='Perform cross-validation of segmentaion models.')
    parser.add_argument("-ecv", "--ensemble_cv", action="store_true",
                        help='Compute the ensemble segmentation for the CV test data.')
    parser.add_argument("-sf", "--segmentation_final", action="store_true",
                    help='Train the final segmentation models using all available data.')
    parser.add_argument("-t", "--test", action="store_true",
                    help='Test')
    parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1,
                        help="Number of parallel jobs. Default 1.")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Level of verbosity. 0: silent, 1: minimal (default), 2: detailed.")
    args = parser.parse_args()


# Assign main directories
data_dir = os.getcwd()
config['swi_dir'] = join(data_dir, 'swi')
config['preproc_dir'] = join(data_dir, 'preproc')
config['augment_dir'] = join(data_dir, 'augment')
config['downsample_dir'] = join(data_dir, 'downsample')
model_dir =  join(data_dir, 'models')

config['n_jobs'] = args.n_jobs
config['full_res'] = args.full_resolution

# Post process sujetcs arguments
if args.subject:
    subjects = np.array([args.subject])


if args.subjects_list:
    # with open('training_subjects', 'r') as f:
    with open(args.subjects_list, 'r') as f:
        subjects = np.array([subject.strip() for subject in f.readlines()])


if args.preproc:

    if args.verbose > 0:
        print('Preprocessing data...')

    config['dataset'] = {'subjects': subjects}

    ut.conform_dir_to_iso(config['swi_dir'],
                          config['preproc_dir'],
                          subjects,
                          config['full_res'],
                          conform_shape=config['conform_shape'],
                          conform_strides=False)

    if args.verbose > 0:
        print('Creating brainmasks...')
    ut.create_swi_brainmask(config, config['preproc_dir'])

    if args.verbose > 0:
        print('Performing N4 bias field correction...')
    ut.N4_bias_correction(config, config['preproc_dir'])

    if args.verbose > 0:
        print('Normalizing data (using SRS)...')
    ut.srs_normalize(config, config['preproc_dir'])


if args.augment:

    # For L/R flip, every label needs to be defined individually
    labels = {'rn_r': 1, 'rn_l': 2,
              'sn_r': 3, 'sn_l': 4,
              'den_r': 5, 'den_l': 6,
              'stn_r': 7, 'stn_l': 8}

    if args.verbose > 0:
        print('Performing data augmentation...')

    config['dataset'] = {'subjects': subjects,
                         'data': [join('preproc', '%s_brain_norm.nii.gz') % s
                                  for s in subjects],
                         'truth': [join(data_dir, 'manual_labels',
                                        '%s.nii.gz') % s for s in subjects]}

    if args.verbose > 0:
        print('Allocating test data')
    gn.augment_dataset(config, labels)

    if args.verbose > 0:
        print('Augmenting training data')
    gn.augment_dataset(config, labels, n_augment=config['n_epochs'])


if args.downsample:

    if args.verbose > 0:
        print('Downsampling test data...')
    gn.downsample_training_dataset(config, subjects)

    if args.verbose > 0:
        print('Downsampling training data...')
    gn.downsample_training_dataset(
        config,
        subjects,
        n_augment=config['n_epochs'])


if args.preallocate_patch:

    labels = {'rn_r': 1, 'rn_l': 2,
              'sn_r': 3, 'sn_l': 4,
              'den_r': 5, 'den_l': 6,
              'stn_r': 7, 'stn_l': 8,
              'rn_sn_stn': [1, 2, 3, 4, 7, 8]}

    if args.verbose > 0:
        print('Preallocating batches for patch models...')

    config['batches_dir'] = join(data_dir, 'batches', 'patch')

    if args.verbose > 0:
        print('Creating test data...')
    gn.preallocate_patch_batches(config,
                                 subjects,
                                 labels)

    if args.verbose > 0:
        print('Preallocating batches for patch models...')
    gn.preallocate_patch_batches(config,
                                 subjects,
                                 labels,
                                 n_augment=config['n_epochs'])


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

        # Model parameters
        config['input_shape'] = [1] + config['downsample_shape']
        config['batch_norm'] = True

        regions = ['rn_r', 'rn_l',
                   'sn_r', 'sn_l',
                   'den_r', 'den_l',
                   'stn_r', 'stn_l',
                   'rn_sn_stn']

        for model in patch_models:

            # Optimization parameters
            if model.endswith('fc_dense_net'):
                config['n_epochs'] = 90
                config['initial_learning_rate'] = 1e-3
                config['end_learning_rate'] = 1e-5
                config['decay_steps'] = 30
            else:
                config['n_epochs'] = 90
                config['initial_learning_rate'] = 1e-2
                config['end_learning_rate'] = 1e-4
                config['decay_steps'] = 30

            model_dir = join(data_dir, 'models', 'patch', model,
                             'n_fold-%i' % n_fold)
            ut.assert_dir(model_dir)

            for region in regions:
                config['batches_x_dir'] = config['downsample_dir']
                config['batches_y_dir'] = join(data_dir, 'batches', 'patch', region)

                model_path = join(model_dir, region)

                ut.train_model(config,
                               model,
                               model_path)

                # Label patches in test subjects
                fdown = [join(config['downsample_dir'], subject, 'x.nii.gz')
                         for subject in subjects[test]]
                out_dir = join(data_dir, 'models', 'patch', model, 'test_labels', 'individual', region)
                ut.assert_dir(out_dir)
                ut.label_patches(subjects[test],
                                 model_path + '.h5',
                                 out_dir,
                                 config['preproc_dir'],
                                 fdown=fdown)


if args.patch_cv_mixed:

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

        # Model parameters
        config['input_shape'] = [1] + config['patch_shape']
        config['batch_norm'] = True

        for mixed_type in ['sigmoid', 'softmax']:

            if mixed_type == 'sigmoid':
                regions = ['rn_r', 'rn_l',
                           'sn_r', 'sn_l',
                           'den_r', 'den_l',
                           'stn_r', 'stn_l']
            elif mixed_type == 'softmax':
                regions = ['rn_sn_stn', 'den_r', 'den_l']

            for model in patch_models:

                # Optimization parameters
                if model.endswith('fc_dense_net'):
                    config['n_epochs'] = 90
                    config['initial_learning_rate'] = 1e-3
                    config['end_learning_rate'] = 1e-5
                    config['decay_steps'] = 30
                else:
                    config['n_epochs'] = 90
                    config['initial_learning_rate'] = 1e-2
                    config['end_learning_rate'] = 1e-4
                    config['decay_steps'] = 30

                # Assert output directory
                model_dir = join(
                    data_dir, 'models', 'patch', model,
                    'n_fold-%i' % n_fold)
                ut.assert_dir(model_dir)

                # Train mixed models
                config['batches_x_dir'] = config['downsample_dir']
                config['batches_y_dir'] = {}
                for region in regions:
                    config['batches_y_dir'][region] = join(
                        data_dir, 'batches', 'patch', region)

                model_path = join(model_dir, 'mixed_' + mixed_type)
                ut.train_mixed_model(config,
                                     model,
                                     model_path,
                                     regions)

                # Label patches in test subjects
                out_dir = join(data_dir, 'models', 'patch', model, 'test_labels', 'mixed_' + mixed_type)

                full_images = [join(config['preproc_dir'], subject + '.nii.gz')
                               for subject in subjects[test]]
                down_images = [
                    join(config['downsample_dir'], subject, 'x.nii.gz')
                    for subject in subjects[test]]

                ut.label_mixed_patches(config,
                                       model_path,
                                       regions,
                                       subjects[test],
                                       full_images,
                                       down_images,
                                       out_dir)


if args.patch_final:

    if args.verbose > 0:
        print('Training final patch models using all available data...')

    # Batch params
    config['training'] = subjects
    config['test'] = subjects

    # Model parameters
    config['input_shape'] = [1] + config['downsample_shape']
    config['batch_norm'] = True

    for mixed_type in ['sigmoid', 'softmax']:

        if mixed_type == 'sigmoid':
            regions = ['rn_r', 'rn_l',
                       'sn_r', 'sn_l',
                       'den_r', 'den_l',
                       'stn_r', 'stn_l']
        elif mixed_type == 'softmax':
            regions = ['rn_sn_stn', 'den_r', 'den_l']

        for model in ['unetpp']:

            # Optimization parameters
            if model.endswith('fc_dense_net'):
                config['n_epochs'] = 90
                config['initial_learning_rate'] = 1e-3
                config['end_learning_rate'] = 1e-5
                config['decay_steps'] = 30
            else:
                config['n_epochs'] = 90
                config['initial_learning_rate'] = 1e-2
                config['end_learning_rate'] = 1e-4
                config['decay_steps'] = 30

            # Assert output directory
            model_dir = join(
                data_dir, 'models', 'patch', model, 'final')
            ut.assert_dir(model_dir)

            # Train mixed models
            config['batches_x_dir'] = config['downsample_dir']
            config['batches_y_dir'] = {}
            for region in regions:
                config['batches_y_dir'][region] = join(
                    data_dir, 'batches', 'patch', region)

            model_path = join(model_dir, 'mixed_' + mixed_type)
            ut.train_mixed_model(config,
                                 model,
                                 model_path,
                                 regions)


if args.preallocate_mixed_segmentation:

    # Generate splits for CV
    skf = KFold(n_splits=n_folds, shuffle=False)
    folds = [folds for folds in skf.split(subjects)]

    for mixed_type in ['mixed_sigmoid', 'mixed_softmax']:

        if mixed_type.endswith('sigmoid'):
            labels = {'rn_r': 1, 'rn_l': 2,
                      'sn_r': 3, 'sn_l': 4,
                      'den_r': 5, 'den_l': 6,
                      'stn_r': 7, 'stn_l': 8}
        elif mixed_type.endswith('softmax'):
            labels = {'rn_sn_stn': [1, 2, 3, 4, 7, 8],
                      'den_r': 5, 'den_l': 6}

        for n_fold, (training, test) in enumerate(folds):

            print('Training data')
            print(subjects[training])
            print('Test data')
            print(subjects[test])

            model = 'unetpp'

            config['batches_dir'] = join(
                data_dir, 'batches', 'region', model, mixed_type)
            ut.assert_dir(config['batches_dir'])

            patch_model = join(
                data_dir, 'models', 'patch', model,
                'n_fold-%i' % n_fold, mixed_type + '.h5')

            print('Creating test data...')
            gn.preallocate_mixed_region_batches(
                config,
                patch_model,
                subjects[test],
                labels)

            print('Processing augmented data for regions...')
            gn.preallocate_mixed_region_batches(
                config,
                patch_model,
                subjects[test],
                labels,
                n_augment=config['n_epochs'])


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

        # Model parameters
        config['input_shape'] = [1] + config['patch_shape']
        config['batch_norm'] = True

        for model in models:

            labels = {'rn_r': 1, 'rn_l': 2,
                      'sn_r': 3, 'sn_l': 4,
                      'den_r': 5, 'den_l': 6,
                      'stn_r': 7, 'stn_l': 8,
                      'rn_sn_stn': [1, 2, 3, 4, 7, 8]}

            # Train models
            model_dir = join(data_dir, 'models', 'region', model,
                             'n_fold-%i' % n_fold)
            ut.assert_dir(model_dir)

            for region in labels:

                if region in ['rn_r', 'rn_l',
                              'sn_r', 'sn_l',
                              'stn_r', 'stn_l']:
                    config['batches_x_dir'] = join(
                        data_dir, 'batches', 'region', 'unetpp',
                        'mixed_sigmoid', region)
                    config['batches_y_dir'] = join(
                        data_dir, 'batches', 'region', 'unetpp',
                        'mixed_sigmoid', region)
                else:
                    config['batches_x_dir'] = join(
                        data_dir, 'batches', 'region', 'unetpp',
                        'mixed_softmax', region)
                    config['batches_y_dir'] = join(
                        data_dir, 'batches', 'region', 'unetpp',
                        'mixed_softmax', region)

                model_path = join(model_dir, region)

                if model.endswith('fc_dense_net'):
                    config['n_epochs'] = 75
                    config['initial_learning_rate'] = 1e-3
                    config['end_learning_rate'] = 1e-5
                    config['decay_steps'] = 25
                else:
                    config['n_epochs'] = 75
                    config['initial_learning_rate'] = 1e-2
                    config['end_learning_rate'] = 1e-4
                    config['decay_steps'] = 25

                ut.train_model(config,
                               model,
                               model_path,
                               labels=labels[region],
                               max_n_train=1)

            # Labeling test data
            config['dataset'] = {'subjects': subjects[test],
                                 'data': [join('preproc',
                                          '%s_brain_norm.nii.gz') % s
                                          for s in subjects[test]],
                                 'truth': [join(data_dir,
                                           'manual_labels', '%s.nii.gz') % s
                                           for s in subjects[test]]}

            # Label using model with sigmoid layer
            labels_dct = {'rn_r': 1, 'rn_l': 2,
                          'sn_r': 3, 'sn_l': 4,
                          'den_r': 5, 'den_l': 6,
                          'stn_r': 7, 'stn_l': 8}

            labels_dir = join(data_dir, 'models', 'region',
                              model, 'test_labels', 'sigmoid')

            patches_dir = join(data_dir, 'models', 'patch',
                               'unetpp', 'test_labels', 'mixed_sigmoid')

            ut.label_regions(config,
                             model_dir,
                             labels_dct,
                             labels_dir,
                             patches_dir,
                             unite=True)

            # Label using model with softmax layer
            labels_dct = {'rn_sn_stn': [1, 2, 3, 4, 7, 8],
                          'den_r': 5, 'den_l': 6}

            labels_dir = join(data_dir, 'models', 'region',
                              model, 'test_labels', 'softmax')

            patches_dir = join(data_dir, 'models', 'patch',
                               'unetpp', 'test_labels', 'mixed_softmax')

            ut.label_regions(config,
                             model_dir,
                             labels_dct,
                             labels_dir,
                             patches_dir,
                             unite=True)


if args.ensemble_cv:

    labels_dct = {'rn_r': 1, 'rn_l': 2,
                  'sn_r': 3, 'sn_l': 4,
                  'den_r': 5, 'den_l': 6,
                  'stn_r': 7, 'stn_l': 8}

    sigmoid_labels = {
            'rn_r': 1, 'rn_l': 2,
            'sn_r': 3, 'sn_l': 4,
            'den_r': 5, 'den_l': 6,
            'stn_r': 7, 'stn_l': 8
        }

    softmax_labels = {'den_r': 5, 'den_l': 6, 'rn_sn_stn': [1, 2, 3, 4, 7, 8]}

    combined_labels = {
            'rn_r': 1, 'rn_l': 2,
            'sn_r': 3, 'sn_l': 4,
            'den_r': 5, 'den_l': 6,
            'stn_r': 7, 'stn_l': 8,
            'rn_sn_stn': [1, 2, 3, 4, 7, 8]
        }

    config['dataset'] = {'subjects': subjects}
    config['models_dir'] = [join(data_dir, 'models', 'region',
                                 model, 'test_labels') for model in models]

    labels_dir = join(data_dir, 'models', 'region', 'ensemble_sigmoid', 'test_labels')
    ut.ensemble_regions(
        config,
        labels_dct,
        labels_dir,
        # bck_thr=0.9,
        training_labels=sigmoid_labels
    )

    labels_dir = join(data_dir, 'models', 'region', 'ensemble_softmax', 'test_labels')
    ut.ensemble_regions(
        config,
        labels_dct,
        labels_dir,
        # bck_thr=0.9,
        training_labels=softmax_labels
    )

    labels_dir = join(data_dir, 'models', 'region', 'ensemble_combined', 'test_labels')
    ut.ensemble_regions(
        config,
        labels_dct,
        labels_dir,
        training_labels=combined_labels
    )


if args.segmentation_final:

    # Batch params
    config['training'] = subjects
    config['test'] = subjects

    # Model parameters
    config['input_shape'] = [1] + config['patch_shape']
    config['batch_norm'] = True

    for model in models:

        labels = {'rn_r': 1, 'rn_l': 2,
                  'sn_r': 3, 'sn_l': 4,
                  'den_r': 5, 'den_l': 6,
                  'stn_r': 7, 'stn_l': 8,
                  'rn_sn_stn': [1, 2, 3, 4, 7, 8]}

        # Train models
        current_model_dir = join(model_dir, 'models', 'region', model, 'final')
        ut.assert_dir(model_dir)

        for region in labels:

            if region in ['rn_r', 'rn_l',
                          'sn_r', 'sn_l',
                          'stn_r', 'stn_l']:
                config['batches_x_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                    'mixed_sigmoid', region)
                config['batches_y_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                    'mixed_sigmoid', region)
            else:
                config['batches_x_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                    'mixed_softmax', region)
                config['batches_y_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                    'mixed_softmax', region)

            model_path = join(model_dir, region)

            if model.endswith('fc_dense_net'):
                config['n_epochs'] = 75
                config['initial_learning_rate'] = 1e-3
                config['end_learning_rate'] = 1e-5
                config['decay_steps'] = 25
            else:
                config['n_epochs'] = 75
                config['initial_learning_rate'] = 1e-2
                config['end_learning_rate'] = 1e-4
                config['decay_steps'] = 25

            ut.train_model(config,
                           model,
                           model_path,
                           labels=labels[region],
                           max_n_train=1,
                           test_batch_random=True)


if args.label_patches:

    print('Downsampling data...')
    downsample_dir = join(data_dir, 'labels', 'downsample')
    ut.downsample_data(config, subjects, downsample_dir)

    print('Labeling patches...')
    model_dir = join(
        data_dir, 'models', 'patch', 'unetpp', 'final')

    mixed_types = ['sigmoid', 'softmax']

    for mixed_type in mixed_types:

        if mixed_type == 'sigmoid':
            regions = ['rn_r', 'rn_l',
                       'sn_r', 'sn_l',
                       'den_r', 'den_l',
                       'stn_r', 'stn_l']
        elif mixed_type == 'softmax':
            regions = ['rn_sn_stn', 'den_r', 'den_l']

        model_path = join(model_dir, 'mixed_' + mixed_type)

        # Label patches in test subjects

        full_images = [join(config['preproc_dir'], subject + '.nii.gz')
                       for subject in subjects]

        down_images = [join(downsample_dir, subject + '.nii.gz')
                       for subject in subjects]

        out_dir = join(
            data_dir, 'labels', 'patches', 'unetpp', mixed_type)

        ut.label_mixed_patches(config,
                               model_path,
                               regions,
                               subjects,
                               full_images,
                               down_images,
                               out_dir)


if args.label_regions:

    # Labeling test data
    config['dataset'] = {'subjects': subjects,
                         'data': [join('preproc',
                                  '%s_brain_norm.nii.gz') % s
                                  for s in subjects]}

    # Label dataset
    for model in models:

        model_dir = join(
            data_dir, 'models', 'region', model, 'final')

        # Label using model with sigmoid layer
        labels_dct = {'rn_r': 1, 'rn_l': 2,
                      'sn_r': 3, 'sn_l': 4,
                      'den_r': 5, 'den_l': 6,
                      'stn_r': 7, 'stn_l': 8}

        labels_dir = join(data_dir, 'labels', model, 'sigmoid')

        patches_dir = join(data_dir, 'labels', 'patches',
                           'unetpp', 'sigmoid')

        ut.label_regions(config,
                         model_dir,
                         labels_dct,
                         labels_dir,
                         patches_dir,
                         unite=True)

        # Label using model with softmax layer
        labels_dct = {'rn_sn_stn': [1, 2, 3, 4, 7, 8],
                      'den_r': 5, 'den_l': 6}

        labels_dir = join(data_dir, 'labels', model, 'softmax')

        patches_dir = join(data_dir, 'labels', 'patches',
                           'unetpp', 'softmax')

        ut.label_regions(config,
                         model_dir,
                         labels_dct,
                         labels_dir,
                         patches_dir,
                         unite=True)

    # Ensemble
    labels_dct = {'rn_r': 1, 'rn_l': 2,
                  'sn_r': 3, 'sn_l': 4,
                  'den_r': 5, 'den_l': 6,
                  'stn_r': 7, 'stn_l': 8}
    # Sigmoid labels
    sigmoid_labels = {
            'rn_r': 1, 'rn_l': 2,
            'sn_r': 3, 'sn_l': 4,
            'den_r': 5, 'den_l': 6,
            'stn_r': 7, 'stn_l': 8
        }

    # Softmax labels
    softmax_labels = {'den_r': 5, 'den_l': 6, 'rn_sn_stn': [1, 2, 3, 4, 7, 8]}

    combined_labels = {
            'rn_r': 1, 'rn_l': 2,
            'sn_r': 3, 'sn_l': 4,
            'den_r': 5, 'den_l': 6,
            'stn_r': 7, 'stn_l': 8,
            'rn_sn_stn': [1, 2, 3, 4, 7, 8]
        }

    config['dataset'] = {'subjects': subjects}

    config['models_dir'] = [join(data_dir, 'labels', model) for model in models]

    labels_dir = join(data_dir, 'labels', 'ensemble_sigmoid')
    ut.ensemble_regions(
        config,
        labels_dct,
        labels_dir,
        # bck_thr=0.9,
        training_labels=sigmoid_labels
    )

    labels_dir = join(data_dir, 'labels', 'ensemble_softmax')
    ut.ensemble_regions(
        config,
        labels_dct,
        labels_dir,
        # bck_thr=0.9,
        training_labels=softmax_labels
    )

    labels_dir = join(data_dir, 'labels', 'ensemble_combined')
    ut.ensemble_regions(
        config,
        labels_dct,
        labels_dir,
        # bck_thr=0.9,
        training_labels=combined_labels
    )


if args.test:

    # Debug
    labels_dct = {'rn_r': 1, 'rn_l': 2,
                  'sn_r': 3, 'sn_l': 4,
                  'den_r': 5, 'den_l': 6,
                  'stn_r': 7, 'stn_l': 8}

    labels = 'manual_labels/000_co.nii.gz'
    target = 'preproc/000_co_brain_norm.nii.gz'

    img = ants.image_read(target)
    data = img.numpy()

    labels_ = ants.image_read(labels).numpy()
    labels_mask = labels_ != 0.

    for region in labels_dct:

        mask = np.isin(labels_, labels_dct[region])
        mask_dil = sp.ndimage.morphology.binary_dilation(
            mask, iterations=5
        ).astype(float)
        # mask_dil[mask] = 0.
        mask_dil[labels_mask] = 0.
        fout = region + '.nii.gz'
        img_out = img.new_image_like(mask_dil)

        print(fout)
        ants.image_write(img_out, fout)
