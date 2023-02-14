import os
import shlex
import ants
import shutil

import tensorflow as tf
import numpy as np
import scipy as sp
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

from os.path import isfile, join
from importlib import reload
from subprocess import Popen, PIPE
from skimage.util.shape import view_as_windows
from sklearn.model_selection import KFold
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from surface_distance.metrics import compute_surface_distances, \
                                     compute_robust_hausdorff

from keras.models import load_model
from keras.backend import set_value, eval
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session, clear_session, \
                                             get_session

try:
    from metrics import (
            dice_coefficient_loss,
            dice_coefficient,
            generalized_focal_tversky_loss,
        )
    import generator as gn
    import models as mdl
except:
    from src.metrics import (
            dice_coefficient_loss,
            dice_coefficient,
            generalized_focal_tversky_loss,
        )
    import src.generator as gn
    import src.models as mdl

reload(gn)
reload(mdl)

custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,
                  'dice_coefficient': dice_coefficient,
                  'tf': tf}


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


def srs_norm(data, eps=None, mask=None, spread=1., offset=0.):

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
    data_[mask] = (data_sig - min_x) / (max_x - min_x) * spread + offset

    return data_


def expand_labels(data, labels, n_labels=None, override_binary=None):

    """ Transform integer labels to binary expension """

    if isinstance(labels, int):
        labels = [labels]

    if override_binary is not None and len(np.unique(data)) > 2:
        print(labels)
        raise ValueError('Expected binary label but found the above labels.')

    if n_labels is None:
        y = np.zeros([len(labels)] + list(data.shape), dtype=np.int8)
        for nl, label in enumerate(labels):
            y[nl, data == label] = 1
    else:

        y = np.zeros([n_labels] + list(data.shape), dtype=np.int8)

        if override_binary is None:
            for label in labels:
                y[label, data == label] = 1
        else:
            y[override_binary, data == 1] = 1

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


def numpy_to_ants(data, ants_ref):

    """ Transform a numpy array to ants format using a reference """

    if data.ndim == 4 and ants_ref.dimension == 3:
        origin = list(ants_ref.origin) + [0.]
        direction = np.pad(ants_ref.direction, (0, 1),
                           mode='constant',
                           constant_values=0)
        direction[3, 3] = 1.
        spacing = list(ants_ref.spacing) + [1.]
    elif data.ndim == 3 and ants_ref.dimension == 4:
        origin = ants_ref.origin[:3]
        spacing = ants_ref.spacing[:3]
        direction = ants_ref.direction[:3, :3]
    else:
        origin = ants_ref.origin
        spacing = ants_ref.spacing
        direction = ants_ref.direction
    has_components = ants_ref.has_components
    is_rgb = ants_ref.is_rgb

    return ants.from_numpy(data, origin, spacing, direction,
                           has_components, is_rgb)


def save_like_ants(data, ants_ref, fname):

    """ Save numpy data to Nifti using an ants reference """

    ants.image_write(numpy_to_ants(data, ants_ref), fname)


def adjust_shape(ants_ref, target_shape):

    """ Crop or pad image to target shape """

    # Crop
    d = ants_ref.numpy()
    low = np.floor((np.array(ants_ref.shape)-target_shape)/2.).astype(int)
    high = np.ceil((np.array(ants_ref.shape)-target_shape)/2.).astype(int)
    l = low.copy()
    h = high.copy()
    l[[l_ < 0 for l_ in l]] = 0
    h[[h_ < 0 for h_ in h]] = 0
    h = np.array(ants_ref.shape) - h
    d = d[l[0]:h[0], l[1]:h[1], l[2]:h[2]]

    # Pad
    l = -low.copy()
    h = -high.copy()
    l[[l_ < 0 for l_ in l]] = 0
    h[[h_ < 0 for h_ in h]] = 0
    pad_width = [(l[0], h[0]), (l[1], h[1]), (l[2], h[2])]
    d = np.pad(d, pad_width, mode='constant', constant_values=0)

    # Set ants params
    spacing = ants_ref.spacing
    direction = ants_ref.direction
    has_components = ants_ref.has_components
    is_rgb = ants_ref.is_rgb

    aff = np.zeros([4, 4])
    aff[:3, :3] = ants_ref.direction*spacing
    aff[:3, 3] = ants_ref.origin
    aff[3, 3] = 1.
    inv_aff = np.linalg.inv(aff)
    inv_aff[:3, 3] -= low
    origin = list(np.linalg.inv(inv_aff)[:3, 3])

    # This creates temporary files... has to be avoided
    # img = ants_ref.to_nibabel()
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


def concat_labels(dct, background=False):

    labels = []
    for label in dct:
        if isinstance(dct[label], list):
            labels += dct[label]
        else:
            labels += [dct[label]]

    if background and 0 not in labels:  # Make sure background is included
        labels = [0] + labels

    labels.sort()

    return labels


def save_batches(x, y, out_x, out_y):

    for nb in range(x.shape[0]):

        x_ = np.squeeze(x[nb, ...])
        nib.save(nib.Nifti1Image(x_, np.eye(4)), out_x + '_%i.nii.gz' % nb)

        if y.shape[1] > 1:
            y_ = frames_to_list(np.squeeze(y[nb, ...]))
            y_ = np.stack(y_, axis=3)
        else:
            y_ = y[nb, 0, ...]
        y_ = y_.astype('float32')
        nib.save(nib.Nifti1Image(y_, np.eye(4)), out_y + '_%i.nii.gz' % nb)


def _generalized_dice_loss(y_true, y_pred, smooth=1.):

    dice_sum = 0.

    for i in range(y_pred.shape[1]):

        y_true_f = y_true[:, i, ...].flatten()
        y_pred_f = y_pred[:, i, ...].flatten()

        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / \
               (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

        dice_sum += dice

        print('Dice loss %i: %f' % (i, 1. - dice))

    return 1 - dice_sum / y_pred.shape[1]


def _focal_dice_loss(y_true, y_pred, beta=6., smooth=1.):

    loss = 0.

    for i in range(y_pred.shape[1]):

        y_true_f = y_true[:, i, ...].flatten()
        y_pred_f = y_pred[:, i, ...].flatten()

        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / \
               (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

        loss_ = 1 - np.power(dice, 1/beta)
        loss += loss_
        print('Focal dice loss %i: %f' % (i, loss_))

    return loss


def _tversky(y_true, y_pred, alpha=0.3):

    smooth = 1.

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    true_pos = np.sum(y_true_f * y_pred_f)
    false_pos = np.sum((1 - y_true_f)*y_pred_f)
    false_neg = np.sum(y_true_f*(1 - y_pred_f))

    return (true_pos + smooth) / \
           (true_pos + alpha*false_pos + (1-alpha)*false_neg + smooth)


def _generalized_tversky_loss(y_true, y_pred, alpha=0.3):

    tversky_sum = 0.

    for i in range(y_pred.shape[1]):
        tvk = _tversky(y_true[:, i, ...],
                       y_pred[:, i, ...],
                       alpha=alpha)
        print('Tversky %i: %f' % (i, tvk))
        tversky_sum += tvk

    return 1 - tversky_sum / y_pred.shape[1]


def _focal_tversky_loss(y_true, y_pred, alpha=0.3, gamma=3):

    tvk = _tversky(y_true, y_pred, alpha=alpha)
    return 1 - np.power(tvk, 1/gamma)


def _generalized_focal_tversky_loss(y_true, y_pred, alpha=0.3, gamma=6.):

    loss = 0.
    for i in range(y_pred.shape[1]):

        tvk_sum = _focal_tversky_loss(y_true[:, i, ...],
                                      y_pred[:, i, ...],
                                      alpha=alpha,
                                      gamma=gamma)
        print('Focal Tversky loss %i: %f' % (i, tvk_sum))
        loss += tvk_sum

    return loss / y_pred.shape[1]


def _attention_categorical_cross_entropy(y_true, y_pred):

    # Assumes the two last layers are background and foreground

    batch_size = 4
    patch_size = [64, 64, 64]

    loss = 0.
    for i in range(2):
        y_true_f = y_true[:, i, ...].flatten()
        y_pred_f = y_pred[:, i, ...].flatten()
        loss -= np.sum(y_true_f * np.log(y_pred_f))

    loss /= batch_size * np.prod(patch_size)

    return loss


def _dice_coefficient(y_true, y_pred, smooth=1.):

    """ Dice coefficient """

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
        (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def init_model(model_name=None,
               model_path=None,
               input_shape=None,
               n_labels=-1,
               activation=None,
               loss=None,
               metrics=None,
               initial_learning_rate=None,
               batch_norm=None):

    # Define model
    reset_keras()

    if model_name == 'unet':
        model = mdl.unet_3d(
            input_shape,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            initial_learning_rate=initial_learning_rate,
            batch_norm=batch_norm
        )

    if model_name == 'attn_unet':
        model = mdl.attn_unet_3d(
            input_shape,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            initial_learning_rate=initial_learning_rate,
            batch_norm=batch_norm
        )

    elif model_name == 'vnet':
        model = mdl.vnet_3d(
            input_shape,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            initial_learning_rate=initial_learning_rate,
            batch_norm=batch_norm
        )

    elif model_name == 'attn_vnet':
        model = mdl.attn_vnet_3d(
            input_shape,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            initial_learning_rate=initial_learning_rate,
            batch_norm=batch_norm
        )

    elif model_name == 'unetpp':
        model = mdl.unetpp_3d(input_shape,
                              n_labels=n_labels,
                              activation=activation,
                              loss=loss,
                              metrics=metrics,
                              batch_norm=batch_norm,
                              initial_learning_rate=initial_learning_rate)

    elif model_name == 'attn_unetpp':
        model = mdl.attn_unetpp_3d(
            input_shape,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            batch_norm=batch_norm,
            initial_learning_rate=initial_learning_rate
        )

    elif model_name == 'fc_dense_net':
        model = mdl.fc_dense_net(input_shape,
                                 depth=3,
                                 n_labels=n_labels,
                                 activation=activation,
                                 loss=loss,
                                 metrics=metrics,
                                 initial_learning_rate=initial_learning_rate)

    elif model_name == 'attn_fc_dense_net':
        model = mdl.attn_fc_dense_net(
            input_shape,
            depth=3,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            initial_learning_rate=initial_learning_rate
        )

    elif model_name == 'dilated_fc_dense_net':
        model = mdl.dilated_fc_dense_net(
            input_shape,
            depth=3,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            initial_learning_rate=initial_learning_rate
        )

    elif model_name == 'attn_dilated_fc_dense_net':
        model = mdl.attn_dilated_fc_dense_net(
            input_shape,
            depth=3,
            n_labels=n_labels,
            activation=activation,
            loss=loss,
            metrics=metrics,
            initial_learning_rate=initial_learning_rate
        )

    plot_model(model, to_file='%s_model.png' % model_path)
    # model.summary()

    return model


def train_model(
        config,
        model_name,
        model_path,
        labels=None,
        max_n_train=1,
        slow_thr=30,
        test_batch_random=False,
        verbose=1):

    """ Train the different CNNs """

    # Define paths
    fmodel = '%s.h5' % model_path
    fmetrics = '%s_metrics.npz' % model_path
    fmetrics_fig = '%s_metrics.png' % model_path

    if verbose:
        print(fmodel)

    # Define loss and activation based on number of labels
    if labels is None or isinstance(labels, int):
        activation = 'sigmoid'
        loss = dice_coefficient_loss
        metrics = dice_coefficient
        n_labels = 1
    else:
        activation = 'softmax'
        # loss = focal_dice_loss
        # metrics = focal_dice_loss

        loss = generalized_focal_tversky_loss
        metrics = generalized_focal_tversky_loss
        if 0 not in labels:  # Make sure to include background
            labels += [0]
        labels.sort()
        n_labels = len(labels)

    # Check convergence
    if isfile(fmodel) and not config['overwrite']:

        train_metrics = np.load(fmetrics)['train_metrics']
        print('Saved metric: %f' % train_metrics[-1])

        if len(train_metrics) == config['n_epochs']:
            if activation == 'sigmoid':

                if model_path.endswith('stn_l') or \
                   model_path.endswith('stn_r'):

                    print('Saved metric: %f' % train_metrics[-1])

                    if train_metrics[-1] >= 0.8:
                        return
                    else:
                        print('Saved metric: %f' % train_metrics[-1])

                else:

                    if train_metrics[-1] >= 0.85:
                        return
                    else:
                        print('Saved metric: %f' % train_metrics[-1])
            elif train_metrics[-1] < 0.5 and activation == 'softmax':
                return

        best_metric = train_metrics[-1]
        n_train = np.load(fmetrics)['n_train']

        if n_train == 1:

            print('Retraining model')
            print('N epochs %i' % len(train_metrics))
            print('Best metric %f' % best_metric)

            n_train = 0

    else:
        best_metric = -1
        n_train = 0

    # Define training steps
    n_steps = np.ceil(
        len(config['training']) / config['batch_size']).astype(int)
    kf = KFold(n_splits=n_steps)
    training_steps = [fold[1] for fold in kf.split(
                        range(len(config['training'])))]

    # Load test data
    if not test_batch_random:
        test_x, test_y = gn.fetch_batch(config['batches_x_dir'],
                                        config['batches_y_dir'],
                                        config['test'],
                                        labels=labels)

    # Train model
    train_metrics = None
    while ((train_metrics is None or
          (train_metrics[-1] < 0.85 and activation == 'sigmoid') or
          (train_metrics[-1] < 0.5 and activation == 'softmax')) and
          n_train < max_n_train):

        n_train += 1

        model = init_model(model_name=model_name,
                           model_path=model_path,
                           input_shape=config['input_shape'],
                           n_labels=n_labels,
                           activation=activation,
                           loss=loss,
                           metrics=metrics,
                           initial_learning_rate=
                           config['initial_learning_rate'],
                           batch_norm=config['batch_norm'])

        train_metrics = list()
        test_metrics = list()

        if max_n_train > 1:
            fmodel_ = '%s_n%i.h5' % (model_path, n_train)
            fmetrics_ = '%s_n%i_metrics.npz' % (model_path, n_train)
            fmetrics_fig_ = '%s_n%i_metrics.png' % (model_path, n_train)

        for ne in range(config['n_epochs']):

            if slow_thr is not None and ne == slow_thr and \
               train_metrics[-1] < 0.8 and activation == 'sigmoid':
                print('Slow convergence. Interrupting.')
                break

            if slow_thr is not None and ne == slow_thr and \
               train_metrics[-1] > 1. and activation == 'softmax':
                print('Slow convergence. Interrupting.')
                break

            # Update learning rate
            if config['end_learning_rate'] is not None:
                decayed_lr = decay_lr(config['initial_learning_rate'],
                                      config['end_learning_rate'],
                                      ne,
                                      config['n_epochs'],
                                      decay_steps=config['decay_steps'])
                set_value(model.optimizer.lr, decayed_lr)

            # Iterate over all steps
            epoch_metric = 0
            for ns in range(n_steps):

                train_x, train_y = gn.fetch_batch(
                    config['batches_x_dir'],
                    config['batches_y_dir'],
                    config['training'][training_steps[ns]],
                    labels=labels,
                    n_epoch=ne)

                if test_batch_random:
                    test_indices = np.random.choice(
                                        len(config['test']),
                                        config['batch_size'],
                                        replace=False
                                    )

                    test_x, test_y = gn.fetch_batch(
                                        config['batches_x_dir'],
                                        config['batches_y_dir'],
                                        config['test'][test_indices],
                                        labels=labels
                                    )

                epoch_metric += model.train_on_batch(train_x, train_y)[1]

            train_metrics.append(epoch_metric / n_steps)
            test_metrics.append(model.test_on_batch(test_x, test_y)[1])

            if verbose > 0:

                print('Epoch %i: Training: %f, Test: %f, Decay: %f' % (
                        ne + 1, train_metrics[-1], test_metrics[-1],
                        eval(model.optimizer.lr)))

            # Update & save metrics plot
            plt.plot(range(1, len(train_metrics) + 1), train_metrics)
            plt.plot(range(1, len(test_metrics)+1), test_metrics)
            plt.legend(['Training', 'Test'])
            plt.xlabel('Epochs')
            plt.ylabel('Dice Coefficient')
            plt.title('Dice: %f' % test_metrics[-1])
            plt.grid(linestyle='--')
            if max_n_train > 1:
                plt.savefig(fmetrics_fig_, format='png')
            else:
                plt.savefig(fmetrics_fig, format='png')
            plt.close()

        # Save out best model and metrics
        if max_n_train > 1:
            print(fmodel)
            model.save(fmodel_)
            np.savez(fmetrics_,
                     train_metrics=train_metrics,
                     test_metrics=test_metrics,
                     n_train=n_train)

            print('Previous best: %f' % best_metric)
            print('Current best: %f' % train_metrics[-1])

            # Save out best model and metrics
            if best_metric < train_metrics[-1]:
                copy(fmodel_, fmodel)
                copy(fmetrics_, fmetrics)
                copy(fmetrics_fig_, fmetrics_fig)
                best_metric = train_metrics[-1]

        elif slow_thr is not None and ne > slow_thr:
            print(fmodel)
            model.save(fmodel)
            np.savez(fmetrics,
                     train_metrics=train_metrics,
                     test_metrics=test_metrics,
                     n_train=n_train)

    close_keras()


def train_mixed_model(config,
                      model_name,
                      model_path,
                      labels,
                      ratios=None,
                      max_n_train=1,
                      slow_thr=30,
                      verbose=1):

    # Train the different CNNs

    # Define paths
    fmodel = model_path + '.h5'
    fmetrics = model_path + '_metrics.npz'
    fmetrics_fig = model_path + '_metrics.png'

    if verbose:
        print(fmodel)

    # Check convergence
    if isfile(fmodel) and not config['overwrite']:
        train_metrics = np.load(fmetrics)['train_metrics']

        if len(train_metrics) == config['n_epochs'] and\
           train_metrics[-1] >= 0.85:
            return

        best_metric = train_metrics[-1]
        n_train = np.load(fmetrics)['n_train']

        if n_train == 1:

            print('Retraining model')
            print('N epochs %i' % len(train_metrics))
            print('Best metric %f' % best_metric)

            n_train = 0

    else:
        best_metric = -1
        n_train = 0

    # Define training parameters
    if isinstance(labels, list):
        activation = 'sigmoid'
        loss = dice_coefficient_loss
        metrics = dice_coefficient
        n_labels = len(labels)
    else:
        raise ValueError('Not implemented')

    # Define training steps
    if isinstance(labels, list):
        n_steps = np.ceil(
            len(config['training']) / config['batch_size']).astype(int)
        kf = KFold(n_splits=n_steps)
        training_steps = [fold[1] for fold in kf.split(
                            range(len(config['training'])))]

    # Load test data
    test_x, test_y = gn.fetch_mixed_batch(
                        'test',
                        config['batches_x_dir'],
                        config['batches_y_dir'],
                        config['test'],
                        labels,
                        n_labels=n_labels)

    # Train model
    train_metrics = None
    while ((train_metrics is None or train_metrics[-1] < 0.85) and
           n_train < max_n_train):

        n_train += 1
        print('N_TRAIN %i' % n_train)
        print('config[n_epochs]=%i' % config['n_epochs'])

        model = init_model(model_name=model_name,
                           model_path=model_path,
                           input_shape=config['input_shape'],
                           n_labels=n_labels,
                           activation=activation,
                           loss=loss,
                           metrics=metrics,
                           initial_learning_rate=
                           config['initial_learning_rate'],
                           batch_norm=config['batch_norm'])

        train_metrics = list()
        test_metrics = list()

        if max_n_train > 1:
            fmodel_ = model_path + '_n%i.h5' % n_train
            fmetrics_ = model_path + '_n%i_metrics.npz' % n_train
            fmetrics_fig_ = model_path + '_n%i_metrics.png' % n_train

        for ne in range(config['n_epochs']):

            if slow_thr is not None and ne == slow_thr and \
               train_metrics[-1] < 0.8 and activation == 'sigmoid':
                print('Slow convergence. Interrupting.')
                break

            # Update learning rate
            if config['end_learning_rate'] is not None:
                decayed_lr = decay_lr(config['initial_learning_rate'],
                                      config['end_learning_rate'],
                                      ne,
                                      config['n_epochs'],
                                      decay_steps=config['decay_steps'])
                set_value(model.optimizer.lr, decayed_lr)

            train_metric = 0
            for ns in range(n_steps):

                train_x, train_y = gn.fetch_mixed_batch(
                      'train',
                      config['batches_x_dir'],
                      config['batches_y_dir'],
                      config['training'][training_steps[ns]],
                      labels,
                      n_epoch=ne)

                train_metric += model.train_on_batch(train_x, train_y)[1]

            train_metrics.append(train_metric / n_steps)
            test_metrics.append(model.test_on_batch(test_x, test_y)[1])

            if verbose > 0:
                print('Epoch %i: Training: %f, Test: %f, Decay: %f' % (
                        ne + 1, train_metrics[-1], test_metrics[-1],
                        eval(model.optimizer.lr)))

            # Update & save metrics plot
            plt.plot(range(1, len(train_metrics) + 1), train_metrics)
            plt.plot(range(1, len(test_metrics)+1), test_metrics)
            plt.legend(['Training', 'Test'])
            plt.xlabel('Epochs')
            plt.ylabel('Dice Coefficient')
            plt.title('Dice: %f' % test_metrics[-1])
            # plt.axis([1, len(train_metrics)+1, 0, 1])
            plt.grid(linestyle='--')
            if max_n_train > 1:
                plt.savefig(fmetrics_fig_, format='png')
            else:
                plt.savefig(fmetrics_fig, format='png')
            plt.close()

        # Save out best model and metrics
        if max_n_train > 1:
            print(fmodel)
            model.save(fmodel_)
            np.savez(fmetrics_,
                     train_metrics=train_metrics,
                     test_metrics=test_metrics,
                     n_train=n_train)

            print('Previous best: %f' % best_metric)
            print('Current best: %f' % train_metrics[-1])

            # Save out best model and metrics
            if best_metric < train_metrics[-1]:

                copy(fmodel_, fmodel)
                copy(fmetrics_, fmetrics)
                copy(fmetrics_fig_, fmetrics_fig)
                best_metric = train_metrics[-1]

        elif slow_thr is not None and ne > slow_thr:
            print(fmodel)
            model.save(fmodel)
            np.savez(fmetrics,
                     train_metrics=train_metrics,
                     test_metrics=test_metrics,
                     n_train=n_train)

    close_keras()


def label_patches(subjects,
                  model_path,
                  out_dir,
                  template_dir,
                  fdown=None,
                  down_res=None,
                  down_shape=None,
                  mixed=False):

    assert_dir(out_dir)
    model = None

    for ns, subject in enumerate(subjects):

        fout = join(out_dir, subject + '.nii.gz')

        if not isfile(fout):

            print(fout)

            if model is None:
                reset_keras()
                model = load_model(model_path,
                                   custom_objects=custom_objects,
                                   compile=False)

            # Load data
            data = ants.image_read(join(template_dir, subject + '.nii.gz'))
            if fdown is None:
                data_down = ants.resample_image(data,
                                                down_res,
                                                interp_type=4)
                data_down = adjust_shape(data_down, down_shape)
            else:
                data_down = ants.image_read(fdown[ns])

            y_pred = model.predict_on_batch(
                np.expand_dims(np.expand_dims(
                    data_down.numpy(), axis=0), axis=0))

            if mixed:
                patch = None
                for nf in range(y_pred.shape[1]):
                    patch_down = data_down.new_image_like(y_pred[0, nf, ...])
                    patch_ = ants.resample_image(patch_down,
                                                 data.spacing,
                                                 interp_type=4)
                    patch_ = adjust_shape(patch_, data.shape)
                    if patch is None:
                        patch = np.zeros(
                            list(patch_.shape) + [y_pred.shape[1]])
                    patch[..., nf] = patch_.numpy()
            else:
                patch_down = data_down.new_image_like(y_pred[0, 0, ...])
                patch = ants.resample_image(patch_down,
                                            data.spacing,
                                            interp_type=4)
                patch = adjust_shape(patch, data.shape).numpy()

            save_like_ants(patch, data, fout)

    if model is not None:
        tf.keras.backend.clear_session()


def label_mixed_patches(config,
                        model_path,
                        labels,
                        subjects,
                        full_images,
                        down_images,
                        out_dir):

    """ Create a patch centered on a region by cropping the image """

    model = None

    for subject, full_image, down_image in \
            zip(subjects, full_images, down_images):

        x_full = ants.image_read(full_image)
        x_down = ants.image_read(down_image)
        y_pred = None

        for nr, region in enumerate(labels):

            out_dir_ = join(out_dir, region)
            assert_dir(out_dir_)

            patch_out = join(out_dir_, subject + '.nii.gz')

            if not isfile(patch_out):

                # Get prediction of patch location
                if y_pred is None:

                    if model is None:
                        reset_keras()
                        model = load_model(model_path + '.h5',
                                           custom_objects=custom_objects,
                                           compile=False)

                    y_pred = model.predict_on_batch(
                        np.expand_dims(np.expand_dims(
                            x_down.numpy(), axis=0), axis=0))

                # Extract focus
                patch_down = x_down.new_image_like(y_pred[0, nr, ...])
                patch_full = ants.resample_image(patch_down,
                                                 x_full.spacing,
                                                 interp_type=4)
                patch_full = adjust_shape(patch_full, x_full.shape)

                print(patch_out)
                save_like_ants(patch_full.numpy(), x_full, patch_out)

    if model is not None:
        tf.keras.backend.clear_session()


def label_regions(config,
                  model_dir,
                  labels_dct,
                  labels_dir,
                  patches_dir,
                  unite=True):

    print('Labeling regions...')

    # Label structures
    for region in labels_dct:

        assert_dir(join(labels_dir, 'predictions', region))
        model = None

        for ns, subject in enumerate(config['dataset']['subjects']):

            fout = join(labels_dir, 'predictions', region,
                        '%s.nii.gz' % subject)

            if not isfile(fout) or config['overwrite']:

                if model is None:
                    reset_keras()
                    model = load_model(join(model_dir, region + '.h5'),
                                       custom_objects=custom_objects,
                                       compile=False)

                # Load data
                data = ants.image_read(config['dataset']['data'][ns])
                fname = join(patches_dir, region, subject + '.nii.gz')
                patch = ants.image_read(fname)

                com = sp.ndimage.measurements.center_of_mass(patch.numpy())
                steps = np.floor(
                    com - np.array(config['patch_shape'])/2.).astype(int)
                data_patch = view_as_windows(
                    data.numpy(), config['patch_shape'])[
                        steps[0], steps[1], steps[2], ...]

                y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(
                                                data_patch, axis=0), axis=0))

                n_labels = y_pred.shape[1]
                pred = np.zeros(list(data.shape) + [n_labels])
                for nl in range(n_labels):
                    pred_view = view_as_windows(
                        pred, config['patch_shape'] + [1])[
                        steps[0], steps[1], steps[2], nl, ...]
                    pred_view[...] = np.expand_dims(
                        y_pred[0, nl, ...], axis=-1)

                print(fout)
                save_like_ants(pred, data, fout)

    tf.keras.backend.clear_session()

    print('\nUniting segmentations')

    if unite:
        for subject in config['dataset']['subjects']:

            """ Add loop here for different label dictionaries """

            fout = join(labels_dir, '%s.nii.gz' % subject)
            if not isfile(fout) or config['overwrite']:

                # Load and merge probability labels for all regions
                probs = []
                labels = []
                for region in labels_dct:

                    # Get labels
                    labels_ = labels_dct[region]
                    if not isinstance(labels_, list):
                        labels_ = [labels_]
                    labels += labels_

                    # Extract probabilities, excluding background
                    fname = join(labels_dir, 'predictions', region,
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

                print(fout)
                nib.save(nib.Nifti1Image(cat_labels, img.affine), fout)

                # raise ValueError('Stop')


def label_mixed_regions(config,
                        model_path,
                        patch_model,
                        labels_dct,
                        labels_dir,
                        unite=True):

    print('Labeling regions...')

    assert_dir(join(labels_dir, 'region'))

    # Label structures
    for region in labels_dct:

        assert_dir(join(labels_dir, 'region', region))
        model = None

        for ns, subject in enumerate(config['dataset']['subjects']):

            fout = join(labels_dir, 'region', region,
                        '%s.nii.gz' % subject)

            if not isfile(fout) or config['overwrite']:

                print(fout)

                if model is None:
                    reset_keras()
                    model = load_model(model_path,
                                       custom_objects=custom_objects,
                                       compile=False)

                # Load data
                data = ants.image_read(config['dataset']['data'][ns])
                fname = join(
                    labels_dir, 'patch', region, '%s.nii.gz' % subject)
                patch = ants.image_read(fname)

                com = sp.ndimage.measurements.center_of_mass(patch.numpy())
                steps = np.floor(
                    com - np.array(config['patch_shape'])/2.).astype(int)
                data_patch = view_as_windows(
                    data.numpy(), config['patch_shape'])[
                                steps[0], steps[1], steps[2], ...]

                y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(
                                                data_patch, axis=0), axis=0))

                n_labels = y_pred.shape[1]
                pred = np.zeros(list(data.shape) + [n_labels])
                for nl in range(n_labels):
                    pred_view = view_as_windows(
                        pred, config['patch_shape'] + [1])[
                            steps[0], steps[1], steps[2], nl, ...]
                    pred_view[...] = np.expand_dims(
                        y_pred[0, nl, ...], axis=-1)

                save_like_ants(pred, data, fout)

    tf.keras.backend.clear_session()

    print('Uniting segmentations')

    if unite:
        for subject in config['dataset']['subjects']:

            """ Add loop here for different label dictionaries """

            fout = join(labels_dir, '%s.nii.gz' % subject)
            if not isfile(fout) or config['overwrite']:

                print(fout)

                # Load and merge probability labels for all regions
                labels = concat_labels(labels_dct)
                cat_probs = None

                for region in labels_dct:

                    # Get labels
                    labels_ = labels_dct[region]
                    if not isinstance(labels_, list):
                        labels_ = [labels_]

                    # Extract probabilities, excluding background
                    fname = join(labels_dir, 'region', region,
                                 '%s.nii.gz' % subject)
                    img = nib.load(fname)
                    probs = img.get_fdata()

                    if cat_probs is None:
                        cat_probs = np.zeros(list(probs.shape[:3]) +
                                             [np.max(labels) + 1])

                    for label in labels_:
                        cat_probs[..., label] = np.squeeze(
                                                    probs[..., label])

                # Derive final background, assuming no overlap
                cat_probs[..., 0] = 1. - np.sum(cat_probs, axis=-1)

                # Assign label to max probability
                cat_labels = np.argmax(cat_probs, axis=-1).astype(float)

                nib.save(nib.Nifti1Image(cat_labels, img.affine), fout)


def ensemble_regions(config,
                     labels_dct,
                     labels_dir,
                     training_labels=None,
                     bck_thr=None,
                     verbose=0):

    assert_dir(labels_dir)

    print('Creating ensemble probability maps...')

    if training_labels is None:
        training_labels = labels_dct

    for region in labels_dct:

        print('\nRegion ' + region)

        if verbose > 1:
            print(region)

        assert_dir(join(labels_dir, region))

        label = labels_dct[region]

        training_regions = np.array(list(training_labels.keys()))
        matching_regions = np.array(
                            [np.isin(label, training_labels[region_])
                             for region_ in training_regions])
        valid_regions = training_regions[matching_regions]

        for subject in config['dataset']['subjects']:

            fout = join(labels_dir, region, subject + '.nii.gz')
            if not isfile(fout) or config['overwrite']:

                pred = []

                for region_ in valid_regions:

                    labels = training_labels[region_]

                    for model_dir in config['models_dir']:

                        if isinstance(labels, list):
                            fname = join(
                                model_dir, 'softmax', 'predictions', region_,
                                subject + '.nii.gz')
                            img = nib.load(fname)
                            data = img.get_fdata()

                            # Make sure background is included
                            if 0 not in labels:
                                labels = [0] + labels

                            # Extract frame
                            pred_ = data[..., np.isin(labels, label)]

                            # Exclude maps with only 10 voxels
                            if np.max(pred_) > 1e-1:
                                # pred_ /= np.max(pred_)
                                pred.append(pred_)

                        else:
                            fname = join(
                                model_dir, 'sigmoid', 'predictions', region_,
                                subject + '.nii.gz')
                            img = nib.load(fname)
                            pred_ = img.get_fdata()

                            # Exclude maps with only 10 voxels
                            if np.max(pred_) > 1e-1:
                                # pred_ /= np.max(pred_)
                                pred.append(pred_)

                avg_pred = np.mean(np.concatenate(pred, axis=-1), axis=-1)

                print(fout)
                nib.save(nib.Nifti1Image(avg_pred, img.affine), fout)

    print('\nUniting labels...')

    if bck_thr is not None:
        bck_dir = join(labels_dir, 'background')
        assert_dir(bck_dir)

    for subject in config['dataset']['subjects']:

        fout = join(labels_dir, subject + '.nii.gz')
        if not isfile(fout):

            # Load and merge probability labels for all regions
            probs = []
            labels = []

            for region in labels_dct:
                fname = join(labels_dir, region, subject + '.nii.gz')
                img = nib.load(fname)
                probs.append(img.get_fdata())
                labels.append(labels_dct[region])

            max_label = np.max([np.max(l) for l in labels])
            cat_probs = np.zeros(list(probs[0].shape[:3]) +
                                 [max_label + 1])
            for prob, label in zip(probs, labels):
                if isinstance(label, list):
                    for nl, label_ in enumerate(label):
                        cat_probs[..., label_] = np.squeeze(prob[..., nl])
                else:
                    cat_probs[..., label] = np.squeeze(prob)

            # Assign label to max probability
            if bck_thr is None:
                cat_probs[..., 0] = 1. - np.sum(cat_probs, axis=-1)  # background
                cat_labels = np.argmax(cat_probs, axis=-1).astype(float)
            else:
                bck = 1. - np.sum(cat_probs, axis=-1)
                fbck = join(bck_dir, subject + '.nii.gz')
                print(fbck)
                nib.save(nib.Nifti1Image(bck, img.affine), fbck)

                cat_labels = np.zeros(list(probs[0].shape[:3]))
                foreground = bck < bck_thr
                cat_labels[foreground] = np.argmax(
                    cat_probs[foreground, :], axis=-1).astype(float)

            print(fout)
            nib.save(nib.Nifti1Image(cat_labels, img.affine), fout)


def flip_labels_lr(data, labels):

    """ Left/right flip label images """

    data_lr = np.zeros_like(data, dtype=float)
    for label in labels:

        if isinstance(labels[label], int):
            if label.endswith('_l') or label.endswith('_r'):

                if label.endswith('_l'):
                    label_ = label[:-2] + '_r'
                elif label.endswith('_r'):
                    label_ = label[:-2] + '_l'

                try:
                    lr_label = labels[label_]
                except:
                    raise ValueError('No matching opposite label for %s' %
                                     label)

                data_lr[data == labels[label]] = lr_label
            else:
                data_lr[data == labels[label]] = labels[label]
        else:
            print(labels)
            raise ValueError('Label id is not an integer.')

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
    indices = np.reshape(y + dy, (-1, 1)), \
                np.reshape(x + dx, (-1, 1)), \
                    np.reshape(z + dz, (-1, 1))

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
    return (2. * intersection + smooth) / \
           (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


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


def create_swi_brainmask(config, data_dir, f_val=0.5):

    """ Create brain mask using FSL's BET """

    for subject in config['dataset']['subjects']:

        brain_file = join(data_dir, subject + '_brain.nii.gz')

        if not isfile(brain_file):
            swi_file = join(data_dir, subject + '.nii.gz')
            cmd = 'bet %s %s -m -R -f %f' % (swi_file, brain_file, f_val)
            run(cmd)
            print(brain_file)

    print('All brain masks created!\n')


def N4_bias_correction(config, data_dir):

    """ Wrapper N4 bias correction """

    for subject in config['dataset']['subjects']:

        N4_file = join(data_dir, subject + '_brain_N4.nii.gz')
        if not isfile(N4_file):
            brain_file = join(data_dir, subject + '_brain.nii.gz')
            mask_file = join(data_dir, subject + '_brain_mask.nii.gz')
            cmd = 'N4BiasFieldCorrection -i %s -o %s -x %s' % (
                  brain_file, N4_file, mask_file)
            run(cmd)
            print(brain_file)

    print('N4 bias correction finished!\n')


def srs_normalize(config, data_dir, spread=1., offset=0.):

    """ Wrappper for SRS normalization """

    for subject in config['dataset']['subjects']:

        norm_file = join(data_dir, subject + '_brain_norm.nii.gz')

        if not isfile(norm_file):

            N4_file = join(data_dir, subject + '_brain_N4.nii.gz')
            mask_file = join(data_dir, subject + '_brain_mask.nii.gz')
            brain = ants.image_read(N4_file)
            mask = ants.image_read(mask_file)

            # SRS normalization
            mask_ = mask.numpy().astype(bool)
            brain_norm = brain.new_image_like(
                srs_norm(brain.numpy().astype(float),
                         mask=mask_,
                         spread=spread,
                         offset=offset))
            ants.image_write(brain_norm, norm_file)
            print(norm_file)

    print('All data normalized!\n')


def extract_metric(metrics,
                   model,
                   subjects,
                   labels_dct,
                   labels_dir,
                   manual_labels_dir=None,
                   trg_dir=None,
                   csv_pattern=None):

    """ Extract metrics """

    for nm, metric in enumerate(metrics):

        if csv_pattern is None:
            csv = join(labels_dir, metric + '.csv')
        else:
            csv = csv_pattern % metric

        if not isfile(csv):

            mt = {}
            for nr, region in enumerate(labels_dct):

                print('\t' + region)

                mt[region] = []

                for subject in subjects:

                    # Load prediction
                    if model == 'jlf':
                        fpred = join(labels_dir, subject, 'Labels.nii.gz')
                    else:
                        fpred = join(labels_dir, subject + '.nii.gz')
                    img = nib.load(fpred)
                    y_ = np.isin(img.get_fdata(), labels_dct[region])

                    if metric == 'volume':
                        vox_volume = np.prod(img.header.get_zooms())
                        mt_ = np.sum(y_)*vox_volume

                    elif metric == 'regional_avg':
                        fin = join(trg_dir, subject + '.nii.gz')
                        data = nib.load(fin).get_fdata()
                        mt_ = np.mean(data[y_])

                    elif metric == 'norm':
                        fin = join(trg_dir, subject + '_brain_norm.nii.gz')
                        data = nib.load(fin).get_fdata()
                        mt_ = np.mean(data[y_])

                    elif metric in ['dices', 'haus95']:

                        # Load manual labels
                        ftest = join(manual_labels_dir, subject + '.nii.gz')
                        y = np.isin(nib.load(ftest).get_fdata(),
                                    labels_dct[region])

                        if metric == 'dices':
                            mt_ = dice_(y, y_)
                        elif metric == 'haus95':
                            spacing = np.array(ants.image_read(ftest).spacing)
                            mt_ = compute_robust_hausdorff(
                                    compute_surface_distances(
                                        y, y_, spacing), 95)

                    # print(mt_)
                    mt[region].append(mt_)

            df = pd.DataFrame(mt).set_index(subjects).sort_index()

            print(csv)
            df.to_csv(csv, index_label='subjects')


def mean_fold_metric(model, csv, subjects, regions, n_folds):

    """ Compute fold-wise average of metrics """

    df_ = pd.read_csv(csv)

    # Generate splits for CV
    skf = KFold(n_splits=n_folds, shuffle=False)
    folds = [folds for folds in skf.split(subjects)]

    mt = np.empty((0, n_folds + 1))
    for region in regions:

        mt_ = []

        for n_fold, (training, test) in enumerate(folds):
            ind = df_['subjects'].isin(subjects[test])
            mt_.append(np.mean(df_[region][ind]))
        mt_.append(np.mean(mt_))  # Average of all folds
        mt = np.vstack((mt, mt_))

    df = pd.concat([pd.DataFrame(
                   {'model': model,
                    'region': regions}),
                    pd.DataFrame(mt)], axis=1)

    return df


def downsample_nifti(config, x_in, x_out, y_in=None, y_out=None):

    data = ants.image_read(x_in)
    data_down = ants.resample_image(data, config['down_res'],
                                    interp_type=4)
    data_down = adjust_shape(
                    data_down,
                    config['downsample_shape']
                )
    ants.image_write(data_down, x_out)

    if y_in is not None:
        truth = ants.image_read(y_in)
        truth_down = ants.resample_image(truth, config['down_res'],
                                         interp_type=1)
        truth_down = adjust_shape(
                        truth_down,
                        config['downsample_shape']
                    )
        ants.image_write(truth_down, y_out)


def downsample_data(config, subjects, out_dir=None):

    """ Downsample data """

    assert_dir(out_dir)

    for ns, subject in enumerate(subjects):

        x_in = join(config['preproc_dir'], subject + '_brain_norm.nii.gz')
        x_out = join(out_dir, subject + '.nii.gz')

        if not isfile(x_out):
            print('Processing ' + subject)
            downsample_nifti(config, x_in, x_out)


def compute_cnr(labels_dct, labels, target):

    img = ants.image_read(target)
    data = img.numpy()

    labels_ = ants.image_read(labels).numpy()
    labels_mask = labels_ != 0.

    cnr = {}
    for region in labels_dct:

        mask = np.isin(labels_, labels_dct[region])
        mask_dil = sp.ndimage.morphology.binary_dilation(
            mask, iterations=5
        )
        mask_dil[labels_mask] = False

        sig = data[mask]
        bck = data[mask_dil]
        sd_bck = np.std(bck)
        cnr[region] = np.abs(np.mean(sig) - np.mean(bck)) / sd_bck

    print(cnr)

    return cnr
