import ants
import gc

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


def augment_data(config, n_augment=None):

    """ Augment image using nonlinear warping and left/right flip """

    for ns, subject in enumerate(config['dataset']['subjects']):

        if config['augment']:
            x_out = [join(config['augment_dir'], subject,
                          'x_%i.nii.gz' % (n_augment*2)),
                     join(config['augment_dir'], subject,
                          'x_%i.nii.gz' % (n_augment*2+1))]
            y_out = [join(config['augment_dir'],
                          subject, 'y_%i.nii.gz' % (n_augment*2)),
                     join(config['augment_dir'], subject,
                          'y_%i.nii.gz' % (n_augment*2+1))]
        else:
            x_out = [join(config['augment_dir'], subject, 'x.nii.gz')]
            y_out = [join(config['augment_dir'], subject, 'y.nii.gz')]

        if (np.any([not isfile(x_) for x_ in x_out]) or
                np.any([not isfile(y_) for y_ in y_out])):

            data = ants.image_read(config['dataset']['data'][ns])
            truth = ants.image_read(config['dataset']['truth'][ns])

            if data.spacing[0] != data.spacing[1] or \
                    data.spacing[0] != data.spacing[2]:
                raise ValueError('Non isotropic volume %s' %
                                 config['dataset']['data'][ns])

            if config['augment']:

                data_list = [data.numpy(), truth.numpy()]
                order = [3, 0]
                data_deformed = ut.elastic_transform(data_list, 500, 10,
                                                     order=order)

                # Cast to ants object
                data_ = data.new_image_like(data_deformed[0])
                truth_ = truth.new_image_like(data_deformed[1].astype(float))

            else:
                data_ = data
                truth_ = truth

            print(x_out[0])
            ants.image_write(data_, x_out[0])
            ants.image_write(truth_, y_out[0])

            # Flip LR
            if config['augment']:
                data_lr = data_.new_image_like(data_.numpy()[::-1, ...])
                truth_lr = truth_.new_image_like(
                               ut.flip_labels_lr(truth_.numpy()[::-1, ...],
                                                 config['labels']))
                print(x_out[1])
                ants.image_write(data_lr, x_out[1])
                ants.image_write(truth_lr, y_out[1])


def augment_dataset(config):

    """ Wrapper for augment_data """

    ut.assert_dir(config['augment_dir'])
    for subject in config['dataset']['subjects']:
        ut.assert_dir(join(config['augment_dir'], subject))

    if config['augment']:
        params = list(product([config], range(config['n_augment'])))
        with Pool(processes=config['n_jobs']) as pool:
            pool.starmap(augment_data, params)
    else:
        augment_data(config)


def downsample_data(config, n_augment=None):

    """ Downsample data """

    for ns, subject in enumerate(config['dataset']['subjects']):

        print(subject)

        if config['augment']:
            x_in = [join(config['augment_dir'], subject,
                         'x_%i.nii.gz' % (n_augment*2)),
                    join(config['augment_dir'], subject,
                         'x_%i.nii.gz' % (n_augment*2+1))]
            y_in = [join(config['augment_dir'], subject,
                         'y_%i.nii.gz' % (n_augment*2)),
                    join(config['augment_dir'], subject,
                         'y_%i.nii.gz' % (n_augment*2+1))]
            x_out = [join(config['downsample_dir'], subject,
                          'x_%i.nii.gz' % (n_augment*2)),
                     join(config['downsample_dir'], subject,
                          'x_%i.nii.gz' % (n_augment*2+1))]
            y_out = [join(config['downsample_dir'], subject,
                          'y_%i.nii.gz' % (n_augment*2)),
                     join(config['downsample_dir'], subject,
                          'y_%i.nii.gz' % (n_augment*2+1))]
        else:
            x_in = [join(config['augment_dir'], subject, 'x.nii.gz')]
            y_in = [join(config['augment_dir'], subject, 'y.nii.gz')]
            x_out = [join(config['downsample_dir'], subject, 'x.nii.gz')]
            y_out = [join(config['downsample_dir'], subject, 'y.nii.gz')]

        if (np.any([not isfile(x_) for x_ in x_out]) or
                np.any([not isfile(y_) for y_ in y_out])):

            for x_in_, y_in_, x_out_, y_out_ in zip(x_in, y_in, x_out, y_out):

                data = ants.image_read(x_in_)
                truth = ants.image_read(y_in_)

                # Downsample
                data_down = ants.resample_image(data, config['down_res'],
                                                interp_type=4)

                truth_down = ants.resample_image(truth, config['down_res'],
                                                 interp_type=1)

                data_down = ut.adjust_shape(data_down,
                                            config['downsample_shape'])
                truth_down = ut.adjust_shape(truth_down,
                                             config['downsample_shape'])

                ants.image_write(data_down, x_out_)
                ants.image_write(truth_down, y_out_)


def downsample_dataset(config):

    """ Wrapper for downsample_data """

    ut.assert_dir(config['downsample_dir'])
    for subject in config['dataset']['subjects']:
        ut.assert_dir(join(config['downsample_dir'], subject))

    if config['augment']:
        params = list(product([config], range(config['n_augment'])))
        with Pool(processes=config['n_jobs']) as pool:
            pool.starmap(downsample_data, params)
    else:
        downsample_data(config)


def generate_patch_batch(config, n_augment=None):

    """ Create batches for the training of the patch extraction models """

    # Binarize y according to the regions of interest
    for ns, subject in enumerate(config['dataset']['subjects']):

        if config['augment']:
            x_in = join(config['augment_dir'], subject,
                        'x_%i.nii.gz' % (n_augment*2))
            y_in = [join(config['downsample_dir'], subject,
                         'y_%i.nii.gz' % (n_augment*2)),
                    join(config['downsample_dir'], subject,
                         'y_%i.nii.gz' % (n_augment*2+1))]
        else:
            x_in = join(config['augment_dir'], subject, 'x.nii.gz')
            y_in = [join(config['downsample_dir'], subject, 'y.nii.gz')]

        data_spacing = ants.image_read(x_in).spacing

        for y_in_ in y_in:

            img = ants.image_read(y_in_)
            truth = img.numpy()

            for region in config['labels']:

                label_out = join(config['batches_dir'], region, subject, y_in_.split('/')[-1])

                if not isfile(label_out) or config['overwrite']:

                    labels = config['labels'][region]
                    if not isinstance(labels, list):
                        labels = [labels]

                    # Binarize truth mask
                    mask = np.zeros_like(truth)
                    for label in labels:
                        mask = np.logical_or(mask, truth == label)

                    com = np.array(sp.ndimage.measurements.center_of_mass(
                                    mask.astype(float)))
                    winsize = np.multiply(config['patch_shape'],
                                          data_spacing)/img.spacing
                    steps = np.ceil(com - winsize/2.).astype(int)

                    focus = np.zeros(img.shape)
                    view = view_as_windows(focus, winsize)
                    view[steps[0], steps[1], steps[2], ...] = 1.

                    ants.image_write(img.new_image_like(focus), label_out)


def preallocate_patch_batches(config):

    """ Wrapper for generate_patch_batch """

    ut.assert_dir(config['batches_dir'])
    for region in config['labels']:
        ut.assert_dir(join(config['batches_dir'], region))
        for subject in config['dataset']['subjects']:
            ut.assert_dir(join(config['batches_dir'], region, subject))

    if config['augment']:
        params = list(product([config], range(config['n_augment'])))
        with Pool(processes=config['n_jobs']) as pool:
            pool.starmap(generate_patch_batch, params)
    else:
        generate_patch_batch(config)


def crop_region_batches(config, region, model, n_augment=None):

    """ Create a patch centered on a region by cropping the image """

    label = config['labels'][region]

    for ns, subject in enumerate(config['dataset']['subjects']):

        if config['augment']:
            x_in = [join(config['augment_dir'], subject,
                         'x_%i.nii.gz' % (n_augment*2)),
                    join(config['augment_dir'], subject,
                         'x_%i.nii.gz' % (n_augment*2+1))]
            x_down = [join(config['downsample_dir'], subject,
                           'x_%i.nii.gz' % (n_augment*2)),
                      join(config['downsample_dir'], subject,
                           'x_%i.nii.gz' % (n_augment*2+1))]
            y_in = [join(config['augment_dir'], subject,
                         'y_%i.nii.gz' % (n_augment*2)),
                    join(config['augment_dir'], subject,
                         'y_%i.nii.gz' % (n_augment*2+1))]
            x_out = [join(config['batches_dir'], region, subject,
                          'x_%i.nii.gz' % (n_augment*2)),
                     join(config['batches_dir'], region, subject,
                          'x_%i.nii.gz' % (n_augment*2+1))]
            y_out = [join(config['batches_dir'], region, subject,
                          'y_%i.nii.gz' % (n_augment*2)),
                     join(config['batches_dir'], region, subject,
                          'y_%i.nii.gz' % (n_augment*2+1))]
        else:
            x_in = [join(config['augment_dir'], subject, 'x.nii.gz')]
            x_down = [join(config['downsample_dir'], subject, 'x.nii.gz')]
            y_in = [join(config['augment_dir'], subject, 'y.nii.gz')]
            x_out = [join(config['batches_dir'], region, subject, 'x.nii.gz')]
            y_out = [join(config['batches_dir'], region, subject, 'y.nii.gz')]

        if (np.any([not isfile(x_) for x_ in x_out]) or
                np.any([not isfile(y_) for y_ in y_out])):

            for x_in_, x_down_, y_in_, x_out_, y_out_ in \
                    zip(x_in, x_down, y_in, x_out, y_out):

                x = ants.image_read(x_in_)
                y = ants.image_read(y_in_).numpy()

                # Extract focus
                x_pred = ants.image_read(x_down_)
                y_pred = model.predict_on_batch(
                            np.expand_dims(np.expand_dims(
                                x_pred.numpy(), axis=0), axis=0))
                patch = x_pred.new_image_like(y_pred[0, 0, ...])
                patch_up = ants.resample_image(patch, x.spacing,
                                               interp_type=4)
                patch_up = ut.adjust_shape(patch_up, x.shape)
                com = sp.ndimage.measurements.center_of_mass(patch_up.numpy())
                steps = np.floor(com - np.array(config['patch_shape'])/2.).astype(int)

                # Mask label
                y_bin = (y == label).astype(float)

                # Extract views on focus
                x_ = view_as_windows(x.numpy(), config['patch_shape'])[
                                steps[0], steps[1], steps[2]]
                y_ = view_as_windows(y_bin, config['patch_shape'])[
                                steps[0], steps[1], steps[2]]

                print(x_out_)
                ut.save_like_ants(x_, x, x_out_)
                ut.save_like_ants(y_, x, y_out_)


def preallocate_region_batches(config):

    """ Wrapper for crop_region_batches """

    print('Cropping patches...')

    ut.assert_dir(config['batches_dir'])

    for region in config['labels']:

        print('Loading model %s' % join(config['patch_models_dir'],
                                        '%s.h5' % region))

        ut.reset_keras()
        custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,
                          'dice_coefficient': dice_coefficient}
        model = load_model(join(config['patch_models_dir'], '%s.h5' % region),
                           custom_objects=custom_objects,
                           compile=False)

        ut.assert_dir(join(config['batches_dir'], region))
        for subject in config['dataset']['subjects']:
            ut.assert_dir(join(config['batches_dir'], region, subject))

        if config['augment']:
            for n_augment in range(config['n_augment']):
                crop_region_batches(config, region, model, n_augment=n_augment)
        else:
            crop_region_batches(config, region, model)

        del model
        print(gc.collect())


def fetch_batch(x_dir, y_dir, subjects, region, indices=None):

    """ Fetch batch data """

    x = []
    y = []

    if indices is None:
        for subject in subjects:
            fx = join(x_dir, subject, 'x.nii.gz')
            fy = join(y_dir, subject, 'y.nii.gz')

            x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
            y.append(np.expand_dims(ants.image_read(fy).numpy(), axis=0))
    else:

        subjects_ = subjects[[i % len(subjects) for i in indices]]
        indices_ = np.floor(indices/len(subjects)).astype(int)

        for subject, i in zip(subjects_, indices_):

            fx = join(x_dir, subject, 'x_%i.nii.gz' % i)
            fy = join(y_dir, subject, 'y_%i.nii.gz' % i)

            x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
            y.append(np.expand_dims(ants.image_read(fy).numpy(), axis=0))

    return np.stack(x, axis=0), np.stack(y, axis=0)
