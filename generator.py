import ants
import gc
import faulthandler

import numpy as np
import scipy as sp

from multiprocessing import Pool
from skimage.util.shape import view_as_windows
from itertools import product
from os.path import isfile, join
from importlib import reload
from keras.models import load_model

try:
    import utils as ut
    from metrics import dice_coefficient_loss, dice_coefficient
except:
    import src.utils as ut
    from src.metrics import dice_coefficient_loss, dice_coefficient

reload(ut)

faulthandler.enable()

custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,
                  'dice_coefficient': dice_coefficient}


def augment_data(config, n_augment=None, labels=None):

    """ Augment image using nonlinear warping and left/right flip """

    for ns, subject in enumerate(config['dataset']['subjects']):

        if n_augment is None:
            x_out = join(config['augment_dir'], subject, 'x.nii.gz')
            y_out = join(config['augment_dir'], subject, 'y.nii.gz')
        else:
            x_out = join(
                config['augment_dir'], subject, 'x_%i.nii.gz' % n_augment)
            y_out = join(
                config['augment_dir'], subject, 'y_%i.nii.gz' % n_augment)

        if not isfile(x_out) or not isfile(y_out):

            data = ants.image_read(config['dataset']['data'][ns])
            truth = ants.image_read(config['dataset']['truth'][ns])

            if data.spacing[0] != data.spacing[1] or \
                    data.spacing[0] != data.spacing[2]:
                raise ValueError('Non isotropic volume %s' %
                                 config['dataset']['data'][ns])

            if n_augment is None:
                data_ = data
                truth_ = truth
            else:
                data_list = [data.numpy(), truth.numpy()]
                order = [3, 0]
                data_deformed = ut.elastic_transform(data_list,
                                                     500, 10,
                                                     order=order)

                # Flip LR
                if np.random.choice([True, False]):
                    data_ = data.new_image_like(data_deformed[0][::-1, ...])
                    truth_ = truth.new_image_like(
                       ut.flip_labels_lr(
                        data_deformed[1].astype(float)[::-1, ...], labels))
                else:
                    data_ = data.new_image_like(data_deformed[0])
                    truth_ = truth.new_image_like(
                        data_deformed[1].astype(float))

            print(x_out)
            ants.image_write(data_, x_out)
            ants.image_write(truth_, y_out)


def augment_dataset(config, labels, n_augment=None):

    """ Wrapper for augment_data """

    ut.assert_dir(config['augment_dir'])
    for subject in config['dataset']['subjects']:
        ut.assert_dir(join(config['augment_dir'], subject))

    if n_augment is None:
        augment_data(config)
    else:
        params = list(product([config], range(n_augment), [labels]))
        with Pool(processes=config['n_jobs']) as pool:
            pool.starmap(augment_data, params)


def downsample_training_data(config, subjects, n_augment=None):

    """ Downsample data """

    if n_augment is not None:
        print('Processing augment %i' % n_augment)

    for ns, subject in enumerate(subjects):

        if n_augment is None:
            x_in = join(config['augment_dir'], subject, 'x.nii.gz')
            y_in = join(config['augment_dir'], subject, 'y.nii.gz')
            x_out = join(config['downsample_dir'], subject, 'x.nii.gz')
            y_out = join(config['downsample_dir'], subject, 'y.nii.gz')

        else:
            x_in = join(config['augment_dir'], subject,
                        'x_%i.nii.gz' % n_augment)
            y_in = join(config['augment_dir'], subject,
                        'y_%i.nii.gz' % n_augment)
            x_out = join(config['downsample_dir'], subject,
                         'x_%i.nii.gz' % n_augment)
            y_out = join(config['downsample_dir'], subject,
                         'y_%i.nii.gz' % n_augment)

        if not isfile(x_out) or not isfile(y_out):
            ut.downsample_nifti(config, x_in, x_out, y_in, y_out)


def downsample_training_dataset(config, subjects, n_augment=None):

    """ Wrapper for downsample_data """

    ut.assert_dir(config['downsample_dir'])
    for subject in subjects:
        ut.assert_dir(join(config['downsample_dir'], subject))

    if n_augment is None:
        downsample_training_data(config, subjects)
    else:
        params = list(product([config], [subjects], range(n_augment)))
        if config['n_jobs'] > 1:
            with Pool(processes=config['n_jobs']) as pool:
                pool.starmap(downsample_training_data, params)
        else:
            for params_ in params:
                downsample_training_data(*params_)


def generate_patch_batch(config, subjects, labels, n_augment=None):

    """ Create batches for the training of the patch extraction models """

    if n_augment is not None:
        print('Processing augment %i' % n_augment)

    for ns, subject in enumerate(subjects):

        if n_augment is None:
            x_down = join(config['downsample_dir'], subject, 'x.nii.gz')
            y_full = join(config['augment_dir'], subject, 'y.nii.gz')
        else:
            x_down = join(config['downsample_dir'], subject,
                          'x_%i.nii.gz' % n_augment)
            y_full = join(config['augment_dir'], subject,
                          'y_%i.nii.gz' % n_augment)

        y = ants.image_read(y_full)
        y_conform = None
        x = None

        for region in labels:

            if n_augment is None:
                patch_out = join(config['batches_dir'], region, subject,
                                 'y.nii.gz')
            else:
                patch_out = join(config['batches_dir'], region, subject,
                                 'y_%i.nii.gz' % n_augment)

            if not isfile(patch_out) or config['overwrite']:

                # Crop y to downsample shape
                if y_conform is None:
                    conform_shape = (np.floor(config['downsample_shape'] *
                                     np.array(config['down_res']) /
                                     config['full_res']))
                    y_conform = ut.adjust_shape(y, conform_shape)

                # Find center of mass
                mask = np.isin(y_conform.numpy(), labels[region]).astype(float)
                com = np.array(sp.ndimage.measurements.center_of_mass(mask))

                # Calculate corresponding coordinates in downsampeld space
                com_down = com * config['full_res'] / config['down_res']

                # Create binary patch
                winsize = (np.array(config['patch_shape']) *
                           config['full_res'] / config['down_res'])
                steps = np.ceil(com_down - winsize/2.).astype(int)
                if x is None:
                    x = ants.image_read(x_down)
                patch = np.zeros(x.shape)
                view = view_as_windows(patch, winsize)
                view[steps[0], steps[1], steps[2], ...] = 1.
                ants.image_write(x.new_image_like(patch), patch_out)


def preallocate_patch_batches(config, subjects, labels, n_augment=None):

    """ Wrapper for generate_patch_batch """

    ut.assert_dir(config['batches_dir'])
    for region in labels:
        ut.assert_dir(join(config['batches_dir'], region))
        for subject in subjects:
            ut.assert_dir(join(config['batches_dir'], region, subject))

    if n_augment is None:
        generate_patch_batch(config, subjects, labels)
    else:
        params = list(product(
            [config], [subjects], [labels], range(n_augment)))
        # if config['n_jobs'] > 1:
        with Pool(processes=config['n_jobs']) as pool:
            pool.starmap(generate_patch_batch, params)
        # else:
        #     for params_ in params:
        #         generate_patch_batch(*params_)


def crop_region_batches(config, region, model, n_augment=None):

    """ Create a patch centered on a region by cropping the image """

    labels = config['labels'][region]
    if not isinstance(labels, list):
        labels = [labels]

    for ns, subject in enumerate(config['dataset']['subjects']):

        if n_augment is None:
            x_in = join(config['augment_dir'], subject, 'x.nii.gz')
            x_down = join(config['downsample_dir'], subject, 'x.nii.gz')
            y_in = join(config['augment_dir'], subject, 'y.nii.gz')
            x_out = join(config['batches_dir'], region, subject, 'x.nii.gz')
            y_out = join(config['batches_dir'], region, subject, 'y.nii.gz')
        else:
            x_in = join(config['augment_dir'], subject,
                        'x_%i.nii.gz' % n_augment)
            x_down = join(config['downsample_dir'], subject,
                          'x_%i.nii.gz' % n_augment)
            y_in = join(config['augment_dir'], subject,
                        'y_%i.nii.gz' % n_augment)
            x_out = join(config['batches_dir'], region, subject,
                         'x_%i.nii.gz' % n_augment)
            y_out = join(config['batches_dir'], region, subject,
                         'y_%i.nii.gz' % n_augment)

        if not isfile(x_out) or not isfile(y_out):

            x = ants.image_read(x_in)
            y = ants.image_read(y_in).numpy()

            # Extract focus
            x_pred = ants.image_read(x_down)
            y_pred = model.predict_on_batch(
                        np.expand_dims(np.expand_dims(
                            x_pred.numpy(), axis=0), axis=0))
            patch = x_pred.new_image_like(y_pred[0, 0, ...])
            patch_up = ants.resample_image(patch, x.spacing,
                                           interp_type=4)
            patch_up = ut.adjust_shape(patch_up, x.shape)
            com = sp.ndimage.measurements.center_of_mass(patch_up.numpy())
            steps = np.floor(
                com - np.array(config['patch_shape'])/2.).astype(int)

            # Mask label
            if len(labels) == 1:
                y_bin = (y == labels[0]).astype(float)
            else:
                # Mask all regions
                mask = np.ones_like(y, dtype=bool)
                for label in labels:
                    mask[y == label] = False
                y_bin = y.copy()
                y_bin[mask] = 0.

            # Extract views on focus
            x_ = view_as_windows(x.numpy(), config['patch_shape'])[
                            steps[0], steps[1], steps[2]]
            y_ = view_as_windows(y_bin, config['patch_shape'])[
                            steps[0], steps[1], steps[2]]

            print(x_out)
            ut.save_like_ants(x_, x, x_out)
            ut.save_like_ants(y_, x, y_out)


def preallocate_region_batches(config, n_augment=None):

    """ Wrapper for crop_region_batches """

    print('Cropping patches...')

    ut.assert_dir(config['batches_dir'])

    for region in config['labels']:

        print('Loading model %s' % join(config['patch_models_dir'],
                                        '%s.h5' % region))

        ut.reset_keras()

        model = load_model(join(config['patch_models_dir'], '%s.h5' % region),
                           custom_objects=custom_objects,
                           compile=False)

        ut.assert_dir(join(config['batches_dir'], region))
        for subject in config['dataset']['subjects']:
            ut.assert_dir(join(config['batches_dir'], region, subject))

        if n_augment is None:
            crop_region_batches(config, region, model)
        else:
            for na in range(n_augment):
                crop_region_batches(config, region, model, n_augment=na)

        del model
        print(gc.collect())


def crop_mixed_region_batches(config, model, subjects, labels, n_augment=None):

    """ Create a patch centered on a region by cropping the image """

    for subject in subjects:

        if n_augment is None:
            x_full = join(config['augment_dir'], subject, 'x.nii.gz')
            y_full = join(config['augment_dir'], subject, 'y.nii.gz')
            x_down = join(config['downsample_dir'], subject, 'x.nii.gz')
        else:
            x_full = join(
                config['augment_dir'], subject, 'x_%i.nii.gz' % n_augment)
            y_full = join(
                config['augment_dir'], subject, 'y_%i.nii.gz' % n_augment)
            x_down = join(
                config['downsample_dir'], subject, 'x_%i.nii.gz' % n_augment)

        y_pred = None

        for nr, region in enumerate(labels):

            out_dir = join(config['batches_dir'], region, subject)
            ut.assert_dir(out_dir)

            if n_augment is None:
                x_out = join(out_dir, 'x.nii.gz')
                y_out = join(out_dir, 'y.nii.gz')
            else:
                x_out = join(out_dir, 'x_%i.nii.gz' % n_augment)

                y_out = join(out_dir, 'y_%i.nii.gz' % n_augment)

            if not isfile(x_out) or not isfile(y_out):

                x_full_ = ants.image_read(x_full)
                y_full_ = ants.image_read(y_full).numpy()
                x_down_ = ants.image_read(x_down)

                # Get prediction of patch location
                if y_pred is None:
                    y_pred = model.predict_on_batch(
                        np.expand_dims(np.expand_dims(
                            x_down_.numpy(), axis=0), axis=0))

                # Extract focus
                patch_down = x_down_.new_image_like(y_pred[0, nr, ...])
                patch_full = ants.resample_image(patch_down,
                                                 x_full_.spacing,
                                                 interp_type=4)
                patch_full = ut.adjust_shape(patch_full, x_full_.shape)
                com = sp.ndimage.measurements.center_of_mass(
                    patch_full.numpy())
                steps = np.floor(
                    com - np.array(config['patch_shape'])/2.).astype(int)

                # Extract views on focus
                x = view_as_windows(
                    x_full_.numpy(), config['patch_shape'])[
                                steps[0], steps[1], steps[2]]
                y = view_as_windows(
                    y_full_, config['patch_shape'])[
                                steps[0], steps[1], steps[2]]

                print(x_out)
                ut.save_like_ants(x, x_full_, x_out)
                ut.save_like_ants(y, x_full_, y_out)


def preallocate_mixed_region_batches(config,
                                     patch_model,
                                     subjects,
                                     labels,
                                     n_augment=None):

    """ Wrapper for crop_region_batches """

    print('Cropping patches...')

    ut.assert_dir(config['batches_dir'])

    print('Loading model ' + patch_model)

    ut.reset_keras()

    model = load_model(patch_model,
                       custom_objects=custom_objects,
                       compile=False)

    if n_augment is None:
        crop_mixed_region_batches(config,
                                  model,
                                  subjects,
                                  labels)
    else:
        for na in range(n_augment):
            crop_mixed_region_batches(config,
                                      model,
                                      subjects,
                                      labels,
                                      n_augment=na)

    del model
    print(gc.collect())


def fetch_batch(x_dir, y_dir, subjects, labels=None, n_epoch=None):

    """ Fetch batch data """

    # if not isinstance(labels, list):
    #     labels_ = ut.concat_labels(labels, background=True)

    x = []
    y = []

    if n_epoch is None:

        for subject in subjects:
            fx = join(x_dir, subject, 'x.nii.gz')
            fy = join(y_dir, subject, 'y.nii.gz')

            x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
            if labels is None:
                y.append(np.expand_dims(ants.image_read(fy).numpy(), axis=0))
            elif isinstance(labels, int):
                y.append(np.expand_dims(
                    np.isin(ants.image_read(fy).numpy(), labels), axis=0))
            else:
                y.append(ut.expand_labels(ants.image_read(fy).numpy(),
                                          labels=labels))

    else:

        for subject in subjects:

            fx = join(x_dir, subject, 'x_%i.nii.gz' % n_epoch)
            fy = join(y_dir, subject, 'y_%i.nii.gz' % n_epoch)

            x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
            if labels is None:
                y.append(np.expand_dims(ants.image_read(fy).numpy(), axis=0))
            elif isinstance(labels, int):
                y.append(np.expand_dims(
                    np.isin(ants.image_read(fy).numpy(), labels), axis=0))
            else:
                y.append(ut.expand_labels(ants.image_read(fy).numpy(),
                                          labels=labels))

    return np.stack(x, axis=0), np.stack(y, axis=0)


def fetch_mixed_batch(batch_type, x_dir, y_dir, subjects, labels,
                      region=None,
                      n_labels=None,
                      n_epoch=None):

    """ Fetch batch data """

    if isinstance(labels, list):
        regions = labels
    else:
        if region is None:
            regions = np.array(list(labels.keys()))
        else:
            regions = [region]

    x = []
    y = []

    if batch_type == 'test':

        if isinstance(labels, list):

            for subject in subjects:

                fx = join(x_dir, subject, 'x.nii.gz')
                x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
                y_ = []

                for region in regions:
                    fy = join(y_dir[region], subject, 'y.nii.gz')
                    y_.append(ants.image_read(fy).numpy())

                y.append(np.stack(y_, axis=0))

        else:

            for subject in subjects:
                for region in regions:

                    fx = join(x_dir[region], subject, 'x.nii.gz')
                    fy = join(y_dir[region], subject, 'y.nii.gz')

                    x.append(np.expand_dims(
                        ants.image_read(fx).numpy(), axis=0))

                    y.append(ut.expand_labels(ants.image_read(fy).numpy(),
                                              labels=labels[region],
                                              n_labels=n_labels))

    elif batch_type == 'train':

        if isinstance(labels, list):

            for subject in subjects:

                fx = join(x_dir, subject, 'x_%i.nii.gz' % n_epoch)
                x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
                y_ = []

                # DEBUG
                # print(fx)

                for region in regions:
                    fy = join(y_dir[region], subject, 'y_%i.nii.gz' % n_epoch)
                    y_.append(ants.image_read(fy).numpy())
                    # print(fy)

                y.append(np.stack(y_, axis=0))

        else:

            for subject in subjects:
                for region in regions:

                    fx = join(x_dir[region], subject, 'x_%i.nii.gz' % n_epoch)
                    fy = join(y_dir[region], subject, 'y_%i.nii.gz' % n_epoch)

                    # print(fx)
                    # print(fy)

                    x.append(np.expand_dims(
                        ants.image_read(fx).numpy(), axis=0))

                    y.append(ut.expand_labels(
                        ants.image_read(fy).numpy(),
                        labels=labels[region],
                        n_labels=n_labels))

    return np.stack(x, axis=0), np.stack(y, axis=0)
