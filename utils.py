import os
import shlex
import ants
import shutil

import tensorflow as tf
import numpy as np
import scipy as sp
import nibabel as nib
import matplotlib.pyplot as plt

from os.path import isfile, join
from importlib import reload
from subprocess import Popen, PIPE
from skimage.util.shape import view_as_windows
from sklearn.model_selection import KFold
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from keras.models import load_model
from keras.backend import set_value, eval
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session, clear_session, \
                                             get_session

try:
    from metrics import dice_coefficient_loss, dice_coefficient, \
                        generalized_dice_coefficient_loss, \
                        generalized_dice_coefficient
    import generator as gn
    import models as mdl
except:
    from src.metrics import dice_coefficient_loss, dice_coefficient, \
                            generalized_dice_coefficient_loss, \
                            generalized_dice_coefficient
    import src.generator as gn
    import src.models as mdl

reload(gn)
reload(mdl)


def assert_dir(dir_path):
    full_path = os.path.abspath(dir_path)
    if not os.path.isdir(full_path):
        print('Creating %s' % full_path)
        os.makedirs(full_path)


def remake_dir(dir_path):

    """ Assert the existence of a directory """

    full_path = os.path.abspath(dir_path)
    if os.path.isdir(full_path):
        shutil.rmtree(full_path)
    os.makedirs(full_path)


def move(src, dest):

    """ Move a file """

    print('Moving %s to %s' % (src, dest))
    shutil.move(src, dest)


def copy(src, dest):
    print('Copying %s to %s' % (src, dest))
    shutil.copy(src, dest)


def run(cmd, live_verbose=False):

    """ Run a command """

    print('\n' + cmd)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    if output:
        print(output.decode('latin_1'))
    if error:
        print(error.decode('latin_1'))


def srs_norm(data, eps=None, mask=None):

    """ Computes the scaled robust sigmoid (SRS) normalization """

    if mask is not None and eps is not None:
        raise ValueError('Conflicting values for eps and mask parameters.')

    if eps is not None:
        mask = (np.abs(data) > eps)
    if mask is None and eps is None:
        mask = np.ones_like(data, dtype=bool)

    med = np.nanmedian(data[mask])
    sd = np.nanstd(data[mask], axis=0)
    data_sig = 1. / (1 + np.exp(-(data[mask] - med) / sd))
    min_x = np.nanmin(data_sig)
    max_x = np.nanmax(data_sig)

    data_ = np.zeros_like(data)
    data_[mask] = (data_sig - min_x) / (max_x - min_x)

    return data_


def expand_labels(data, labels=None, background=False):

    """ Transform integer labels to binary expension """

    if labels is None:
        labels = np.unique(data[data != 0])

    y = np.zeros([len(labels)] + list(data.shape), dtype=np.int8)
    for nl, label in enumerate(labels):
        y[nl, data == label] = 1

    if background:
        mask = np.expand_dims(np.array(np.sum(y, axis=0) == 0, dtype=np.int8),
                              axis=0)
        y = np.concatenate((mask, y), axis=0)

    return y


def condense_labels(data, labels_ids=None):

    """ Transform binary labels to integers """

    data_ = data.astype(bool)
    labels = np.zeros(np.array(data.shape)[[0, 1, 2]], dtype=float)
    if labels_ids is None:
        labels_ids = np.arange(data.shape[3]) + 1
    for nl, label in enumerate(labels_ids):
        labels[data_[:, :, :, nl]] = label
    return labels


def frames_to_list(data, pos='first'):

    """ Transform 4D data to list of 3D images """

    frames = list()
    if pos == 'first':
        for nf in range(data.shape[0]):
            frames.append(data[nf, ...])
    elif pos == 'last':
        for nf in range(data.shape[3]):
            frames.append(data[..., nf])
    return frames


def window_steps(data, patch_shape=None, buffer=1.2):

    """ Obtain center and minimal cropping mask of binary image """

    x = np.sum(np.sum(data, axis=2), axis=1)
    x_range = np.where(x)[0][[0,-1]]
    y = np.sum(np.sum(data, axis=2), axis=0)
    y_range = np.where(y)[0][[0,-1]]
    z = np.sum(np.sum(data, axis=1), axis=0)
    z_range = np.where(z)[0][[0,-1]]

    # Make sure cropping range is smaller than patch size
    if patch_shape is not None:
        if np.diff(x_range) > patch_shape[0]*buffer:
            raise ValueError('Cropping range larger than patch for axis x.')
        if np.diff(y_range) > patch_shape[1]*buffer:
            raise ValueError('Cropping range larger than patch for axis y.')
        if np.diff(z_range) > patch_shape[2]*buffer:
            raise ValueError('Cropping range larger than patch for axis z.')

    x_step = int(x_range[0] + np.floor(np.diff(x_range)[0]/2)
                            - patch_shape[0]/2)
    y_step = int(y_range[0] + np.floor(np.diff(y_range)[0]/2)
                            - patch_shape[1]/2)
    z_step = int(z_range[0] + np.floor(np.diff(z_range)[0]/2)
                            - patch_shape[2]/2)

    return [x_step, y_step, z_step]


def save_nifti(data, fname):

    """ Wrapper to save Nifti without affine information """

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, fname)


def close_keras():

    """ Close Keras """

    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()


def reset_keras():

    """ Reset Keras """

    close_keras()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))


def numpy_to_ants(data, ants_):

    """ Transform a numpy array to ants format using a reference """

    if data.ndim == 4 and ants_.dimension == 3:
        origin = list(ants_.origin) + [0.]
        direction = np.pad(ants_.direction, (0, 1),
                           mode='constant',
                           constant_values=0)
        direction[3, 3] = 1.
        spacing = list(ants_.spacing) + [1.]
    elif data.ndim == 3 and ants_.dimension == 4:
        origin = ants_.origin[:3]
        spacing = ants_.spacing[:3]
        direction = ants_.direction[:3, :3]
    else:
        origin = ants_.origin
        spacing = ants_.spacing
        direction = ants_.direction
    has_components = ants_.has_components
    is_rgb = ants_.is_rgb

    return ants.from_numpy(data, origin, spacing, direction,
                           has_components, is_rgb)


def save_like_ants(data, ants_, fname):

    """ Save numpy data to Nifti using an ants reference """

    ants.image_write(numpy_to_ants(data, ants_), fname)


def adjust_shape(ants_, target_shape):

    """ Crop or pad image to target shape """

    # Crop
    d = ants_.numpy()
    low = np.floor((np.array(ants_.shape)-target_shape)/2.).astype(int)
    high = np.ceil((np.array(ants_.shape)-target_shape)/2.).astype(int)
    l = low.copy()
    h = high.copy()
    l[[l_ < 0 for l_ in l]] = 0
    h[[h_ < 0 for h_ in h]] = 0
    h = np.array(ants_.shape) - h
    d = d[l[0]:h[0], l[1]:h[1], l[2]:h[2]]

    # Pad
    l = -low.copy()
    h = -high.copy()
    l[[l_ < 0 for l_ in l]] = 0
    h[[h_ < 0 for h_ in h]] = 0
    pad_width = [(l[0], h[0]), (l[1], h[1]), (l[2], h[2])]
    d = np.pad(d, pad_width, mode='constant', constant_values=0)

    # Set ants params
    spacing = ants_.spacing
    direction = ants_.direction
    has_components = ants_.has_components
    is_rgb = ants_.is_rgb

    aff = np.zeros([4, 4])
    aff[:3, :3] = ants_.direction*spacing
    aff[:3, 3] = ants_.origin
    aff[3, 3] = 1.
    inv_aff = np.linalg.inv(aff)
    inv_aff[:3, 3] -= low
    origin = list(np.linalg.inv(inv_aff)[:3, 3])

    # This creates temporary files... has to be avoided
    # img = ants_.to_nibabel()
    # inv_aff = np.linalg.inv(img.affine)
    # inv_aff[:3,3] -= low
    # aff = np.linalg.inv(inv_aff)
    # print(aff)
    # return ants.from_nibabel(nib.Nifti1Image(d, aff))

    return ants.from_numpy(d,
                           origin=origin,
                           spacing=spacing,
                           direction=direction,
                           has_components=has_components,
                           is_rgb=is_rgb)


def train_model(config, region, model, model_path, verbose=1):

    """ Train the different CNNs """

    # Assign parameters
    activation = 'sigmoid'
    loss = dice_coefficient_loss
    metrics = dice_coefficient

    if verbose > 1:
        print('Activation:')
        print(activation)
        print('Loss:')
        print(loss)
        print('Metric:')
        print(metrics)

    # Load or instantiate model
    fmodel = '%s.h5' % model_path
    fmetrics = '%s_metrics.npz' % model_path
    fmetrics_fig = '%s_metrics.png' % model_path
    if isfile(fmodel) and not config['overwrite']:

        # Load existing metrics
        train_metrics = np.load(fmetrics)['train_metrics'].tolist()
        test_metrics = np.load(fmetrics)['test_metrics'].tolist()
        initial_epoch = len(train_metrics)

        if initial_epoch == config['n_epochs']:
            return

        # Load previous model
        reset_keras()
        custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,
                          'dice_coefficient': dice_coefficient,
                          'generalized_dice_coefficient_loss': generalized_dice_coefficient_loss,
                          'generalized_dice_coefficient': generalized_dice_coefficient}
        model = load_model(fmodel,
                           custom_objects=custom_objects,
                           compile=True)
        print('Resuming training for model: %s' % fmetrics)
        print('Initial epoch: %i' % initial_epoch)
    else:

        # Define model
        reset_keras()
        if model == 'unet':
            model = mdl.unet_3d(config['input_shape'],
                                loss=loss,
                                metrics=metrics,
                                initial_learning_rate=config['initial_learning_rate'],
                                batch_norm=config['batch_norm'])

        elif model == 'vnet':
            model = mdl.vnet_3d(config['input_shape'],
                                loss=loss,
                                metrics=metrics,
                                initial_learning_rate=config['initial_learning_rate'],
                                batch_norm=config['batch_norm'])

        elif model == 'unetpp':
            model = mdl.unetpp_3d(config['input_shape'],
                                  activation=activation,
                                  loss=loss,
                                  metrics=metrics,
                                  batch_norm=config['batch_norm'],
                                  initial_learning_rate=config['initial_learning_rate'])

        plot_model(model, to_file='%s_model.png' % model_path)
        model.summary()

        print('Training model: %s' % fmetrics)

        train_metrics = list()
        test_metrics = list()
        initial_epoch = 0

    kf = KFold(n_splits=config['n_epochs'], shuffle=True)
    training_indices = [fold[1] for fold in
                        kf.split(range(config['n_epochs']*config['batch_size']))]

    test_x, test_y = gn.fetch_batch(config['batches_x_dir'],
                                    config['batches_y_dir'],
                                    config['test'],
                                    region)

    # Train model
    for ne in range(initial_epoch, config['n_epochs']):

        # Load test data
        if config['test_batch_random']:
            test_indices = np.random.choice(len(config['test']),
                                            config['batch_size'],
                                            replace=False)
            test_x, test_y = gn.fetch_batch(config['batches_x_dir'],
                                            config['batches_y_dir'],
                                            config['test'],
                                            region,
                                            indices=test_indices)

        train_x, train_y = gn.fetch_batch(config['batches_x_dir'],
                                          config['batches_y_dir'],
                                          config['training'],
                                          region,
                                          indices=training_indices[ne])

        # Update learning rate
        if config['end_learning_rate'] is not None:
            decayed_lr = decay_lr(config['initial_learning_rate'],
                                  config['end_learning_rate'],
                                  ne,
                                  config['n_epochs'],
                                  decay_steps=config['decay_steps'])
            set_value(model.optimizer.lr, decayed_lr)

        train_metrics_ = model.train_on_batch(train_x, train_y)
        test_metrics_ = model.test_on_batch(test_x, test_y)

        if verbose > 0:
            print('Epoch %i: Training: %f, Test: %f, Decay: %f' % (
                    ne, train_metrics_[1], test_metrics_[1],
                    eval(model.optimizer.lr)))

        train_metrics.append(train_metrics_)
        test_metrics.append(test_metrics_)

        # Save out best model and metrics after warm up
        if (ne % 100 == 0 and ne > 0) or ne == config['n_epochs']-1:
            model.save(fmodel)
            np.savez(fmetrics,
                     train_metrics=train_metrics,
                     test_metrics=test_metrics,
                     dice=test_metrics_[1])

        # Save metrics plot
        if (ne % 10 == 0 and ne > 0) or ne == config['n_epochs']-1:
            plt.plot(range(1, len(train_metrics)+1),
                     np.vstack(train_metrics)[:, 1])
            plt.plot(range(1, len(test_metrics)+1),
                     np.vstack(test_metrics)[:, 1])
            plt.legend(['Training', 'Test'])
            plt.xlabel('Epochs')
            plt.ylabel('Dice Coefficient')
            plt.title('Dice: %f' % test_metrics_[1])
            plt.axis([1, len(train_metrics)+1, 0, 1])
            plt.grid(linestyle='--')
            plt.savefig(fmetrics_fig, format='png')
            plt.close()

    close_keras()


def label_regions(config, unite=True):

    """ Perform the labeling of images """

    print('Extracting patches')

    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,
                      'dice_coefficient': dice_coefficient,
                      'generalized_dice_coefficient_loss': generalized_dice_coefficient_loss,
                      'generalized_dice_coefficient': generalized_dice_coefficient}

    assert_dir(join(config['labels_dir'], 'patch'))

    for region in config['labels']:

        assert_dir(join(config['labels_dir'], 'patch', region))
        model = None

        for ns, subject in enumerate(config['dataset']['subjects']):

            fout = join(config['labels_dir'], 'patch', region,
                        '%s.nii.gz' % subject)

            if not isfile(fout):

                if model is None:
                    reset_keras()
                    model = load_model(join(config['patch_models_dir'],
                                            '%s.h5' % region),
                                       custom_objects=custom_objects,
                                       compile=False)

                print('Processing %s' % subject)

                # Load data
                data = ants.image_read(config['dataset']['data'][ns])

                # Downsample and reshape
                data_down = ants.resample_image(data, config['down_res'],
                                                interp_type=4)

                data_down = adjust_shape(data_down, config['downsample_shape'])

                y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(
                                                data_down.numpy(), axis=0),
                                                axis=0))

                patch_down = data_down.new_image_like(y_pred[0, 0, ...])

                patch = ants.resample_image(patch_down, data.spacing,
                                            interp_type=4)
                patch = adjust_shape(patch, data.shape)

                ants.image_write(patch, fout)

    tf.keras.backend.clear_session()

    print('Labeling regions...')

    assert_dir(join(config['labels_dir'], 'region'))

    # Label structures
    for region in config['labels']:

        assert_dir(join(config['labels_dir'], 'region', region))
        model = None

        for ns, subject in enumerate(config['dataset']['subjects']):

            fout = join(config['labels_dir'], 'region', region,
                        '%s.nii.gz' % subject)

            if not isfile(fout) or config['overwrite']:

                if model is None:
                    reset_keras()
                    model = load_model(join(config['region_models_dir'],
                                            '%s.h5' % region),
                                       custom_objects=custom_objects,
                                       compile=False)

                print('Processing %s' % subject)

                # Load data
                data = ants.image_read(config['dataset']['data'][ns])
                fname = join(config['labels_dir'], 'patch', region,
                             '%s.nii.gz' % subject)
                patch = ants.image_read(fname)

                com = sp.ndimage.measurements.center_of_mass(patch.numpy())
                steps = np.floor(com - np.array(config['patch_shape'])/2.).astype(int)
                data_patch = view_as_windows(data.numpy(), config['patch_shape'])[
                                steps[0], steps[1], steps[2], ...]

                y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(
                                                data_patch, axis=0), axis=0))

                n_labels = y_pred.shape[1]
                pred = np.zeros(list(data.shape) + [n_labels])
                for nl in range(n_labels):
                    pred_view = view_as_windows(pred, config['patch_shape'] + [1])[
                                                steps[0], steps[1], steps[2], nl, ...]
                    pred_view[...] = np.expand_dims(y_pred[0, nl, ...], axis=-1)

                save_like_ants(pred, data, fout)

    tf.keras.backend.clear_session()

    print('Uniting segmentations')

    if unite:
        for subject in config['dataset']['subjects']:

            fout = join(config['labels_dir'], '%s.nii.gz' % subject)
            if not isfile(fout) or config['overwrite']:

                print('Processing %s' % subject)

                # Load and merge probability labels for all regions
                probs = []
                labels = []
                for region in config['labels']:

                    # Get labels
                    labels_ = config['labels'][region]
                    if not isinstance(labels_, list):
                        labels_ = [labels_]
                    labels += labels_

                    # Extract probabilities, excluding background
                    fname = join(config['labels_dir'], 'region', region,
                                 '%s.nii.gz' % subject)
                    img = nib.load(fname)

                    if len(labels_) > 1:
                        probs += frames_to_list(img.get_fdata(),
                                                pos='last')[1:]
                    else:
                        probs.append(img.get_fdata())

                # Concat probabilities
                cat_probs = np.zeros(list(probs[0].shape[:3]) +
                                     [np.max(labels) + 1])
                for prob, label in zip(probs, labels):
                    cat_probs[..., label] = np.squeeze(prob)

                # Derive final background, assuming no overlap
                cat_probs[..., 0] = 1. - np.sum(cat_probs, axis=-1)

                # Assign label to max probability
                cat_labels = np.argmax(cat_probs, axis=-1).astype(float)

                nib.save(nib.Nifti1Image(cat_labels, img.affine), fout)


def ensemble_regions(config):

    """ Perform the ensemble of multiple models across regions """

    assert_dir(config['ensemble_dir'])

    print('Creating ensemble probability maps...')

    for region in config['labels']:

        assert_dir(join(config['ensemble_dir'], region))

        for subject in config['dataset']['subjects']:

            fout = join(config['ensemble_dir'], region, '%s.nii.gz' % subject)

            if not isfile(fout) or config['overwrite']:
                print(fout)
                pred = []
                for model_dir in config['models_dir']:
                    fname = join(model_dir, 'region', region,
                                 '%s.nii.gz' % subject)
                    img = nib.load(fname)
                    data = img.get_fdata()

                    # Exclude maps with only 10 voxels
                    if np.sum(data) > 10:
                        pred.append(img.get_fdata())

                if len(pred) == 0:
                    pred = [np.zeros_like(data)]

                avg_pred = np.mean(np.stack(pred, axis=-1), axis=-1)
                nib.save(nib.Nifti1Image(avg_pred, img.affine), fout)

    print('Uniting labels...')

    for subject in config['dataset']['subjects']:

        print('Processing %s' % subject)

        fout = join(config['ensemble_dir'], '%s.nii.gz' % subject)
        if not isfile(fout):

            # Load and merge probability labels for all regions
            probs = []
            labels = []

            for region in config['labels']:
                fname = join(config['ensemble_dir'], region,
                             '%s.nii.gz' % subject)
                img = nib.load(fname)
                probs.append(img.get_fdata())
                if isinstance(config['labels'][region], list):
                    labels += config['labels'][region]
                else:
                    labels.append(config['labels'][region])

            # Concat probability maps
            cat_probs = np.zeros(list(probs[0].shape[:3]) +
                                 [np.max(labels) + 1])
            for prob, label in zip(probs, labels):
                cat_probs[..., label] = np.squeeze(prob)
            cat_probs[..., 0] = 1. - np.max(cat_probs, axis=-1)

            # Max voting
            cat_labels = np.argmax(cat_probs, axis=-1).astype(float)

            nib.save(nib.Nifti1Image(cat_labels, img.affine), fout)


def flip_labels_lr(data, labels):

    """ Left/right flip label images """

    data_lr = np.zeros_like(data, dtype=float)
    for label in labels:

        if not isinstance(labels[label], list):
            if label.endswith('_l') or label.endswith('_r'):

                if label.endswith('_l'):
                    label_ = label[:-2] + '_r'
                elif label.endswith('_r'):
                    label_ = label[:-2] + '_l'

                try:
                    lr_label = labels[label_]
                except:
                    raise ValueError('No matching opposite label for %s' % label)

                data_lr[data == labels[label]] = lr_label
            else:
                data_lr[data == labels[label]] = labels[label]

    return data_lr


def elastic_transform(image, alpha, sigma, random_state=None, order=3,
                      mode='constant'):

    """Elastic deformation of images as described in [Simard2003] (with modifications).
       [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    # Assume constant shape for all inputs
    if isinstance(image, list):
        shape = image[0].shape
    else:
        shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]),
                          np.arange(shape[0]),
                          np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    if isinstance(image, list):
        if not isinstance(order, list):
            order = [order]*len(image)
        if not isinstance(mode, list):
            mode = [mode]*len(image)
        deformed = []
        for image_, order_, mode_ in zip(image, order, mode):
            deformed.append(map_coordinates(image_, indices, order=order_,
                                            mode=mode_).reshape(shape))
    else:
        deformed = map_coordinates(image,
                                   indices,
                                   order=order,
                                   mode=mode).reshape(shape)

    return deformed


def decay_lr(initial_learning_rate,
             end_learning_rate,
             global_step,
             n_epochs,
             decay_steps=None):

    """ Step decay learning rate """

    if decay_steps is None:
        decay_rate = (end_learning_rate/initial_learning_rate)**(1/n_epochs)
        decayed_lr = initial_learning_rate*decay_rate**np.floor(global_step)
    else:
        total_steps = np.floor(float(n_epochs)/float(decay_steps)) - 1
        decay_rate = (end_learning_rate/initial_learning_rate)**(1/total_steps)
        decayed_lr = initial_learning_rate*decay_rate**np.floor(
                     float(global_step)/float(decay_steps))

    return decayed_lr


def dice_(y_true, y_pred, smooth=1.):

    """ Dice coefficient """

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def conform_dir_to_iso(in_dir, out_dir, subjects, iso_resolution,
                       conform_shape=None, conform_strides=True):

    """ Conform all Nifti files in directory to isotropic resolution """

    assert_dir(out_dir)

    print('Conforming data to isotropic resolution %f' % iso_resolution)

    for subject in subjects:

        x_out = join(out_dir, '%s.nii.gz' % subject)

        if not isfile(x_out):

            data = ants.image_read(join(in_dir, '%s.nii.gz' % subject))

            # Transform to isometric spacing
            resolution = [iso_resolution, iso_resolution, iso_resolution]
            data_iso = ants.resample_image(data, resolution, interp_type=4)

            # Make sure inputs in native resolution conform to specified shape
            if conform_shape is not None:
                data_iso = adjust_shape(data_iso, conform_shape)

            print(x_out)
            ants.image_write(data_iso, x_out)

            print('Conforming strides to (1,2,3)')
            cmd = 'mrconvert -strides 1,2,3 %s %s -force' % (x_out, x_out)
            run(cmd)

    print('All data conformed to isotropic resolution!\n')


def conform_files_to_iso(in_files, out_files, iso_resolution,
                         conform_shape=None, conform_strides=True):

    """ Conform Nifti images to an given isotropic resolution """

    print('Conforming data to isotropic resolution %f' % iso_resolution)

    for fin, fout in zip(in_files, out_files):

        if not isfile(fout):

            print(fin)
            data = ants.image_read(fin)

            # Transform to isometric spacing
            resolution = [iso_resolution, iso_resolution, iso_resolution]
            data_iso = ants.resample_image(data, resolution, interp_type=4)

            # Make sure inputs in native resolution conform to specified shape
            if conform_shape is not None:
                data_iso = adjust_shape(data_iso, conform_shape)

            print(fout)
            ants.image_write(data_iso, fout)

            print('Conforming strides to (1,2,3)')
            cmd = 'mrconvert -strides 1,2,3 %s %s -force' % (fout, fout)
            run(cmd)

    print('All data conformed to isotropic resolution!\n')


def create_swi_brainmask(config):

    """ Create brain mask using FSL's BET """

    for subject in config['dataset']['subjects']:

        brain_file = join(config['preproc_dir'], '%s_brain.nii.gz' % subject)

        if not isfile(brain_file):
            swi_file = join(config['preproc_dir'], '%s.nii.gz' % subject)
            cmd = 'bet %s %s -m -R' % (swi_file, brain_file)
            run(cmd)
            print(brain_file)

    print('All brain masks created!\n')


def N4_bias_correction(config):

    """ Wrapper N4 bias correction """

    for subject in config['dataset']['subjects']:

        N4_file = join(config['preproc_dir'],
                       '%s_brain_N4.nii.gz' % subject)
        if not isfile(N4_file):
            brain_file = join(config['preproc_dir'],
                              '%s_brain.nii.gz' % subject)
            mask_file = join(config['preproc_dir'],
                             '%s_brain_mask.nii.gz' % subject)
            cmd = 'N4BiasFieldCorrection -i %s -o %s -x %s' % (
                  brain_file, N4_file, mask_file)
            run(cmd)
            print(brain_file)

    print('N4 bias correction finished!\n')


def srs_normalize(config):

    """ Wrappper for SRS normalization """

    for subject in config['dataset']['subjects']:

        norm_file = join(config['preproc_dir'],
                         '%s_brain_norm.nii.gz' % subject)

        if not isfile(norm_file):

            N4_file = join(config['preproc_dir'],
                           '%s_brain_N4.nii.gz' % subject)
            mask_file = join(config['preproc_dir'],
                             '%s_brain_mask.nii.gz' % subject)
            brain = ants.image_read(N4_file)
            mask = ants.image_read(mask_file)

            # SRS normalization
            mask_ = mask.numpy().astype(bool)
            brain_norm = brain.new_image_like(srs_norm(brain.numpy().astype(
                                              float), mask=mask_))
            ants.image_write(brain_norm, norm_file)
            print(norm_file)

    print('All data normalized!\n')
