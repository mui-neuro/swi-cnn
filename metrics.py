from keras import backend as K
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1.):

    """ Dice coefficient """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):

    """ Dice coefficient loss """

    return -dice_coefficient(y_true, y_pred)


def generalized_dice_loss(y_true, y_pred):

    loss = 0.

    for i in range(y_pred.shape[1]):
        loss += 1 - dice_coefficient(
            y_true[:, i, ...],
            y_pred[:, i, ...]
        )

    return loss


def focal_dice_loss(y_true, y_pred, beta=6., smooth=1., debug=False):

    loss = 0.

    for i in range(y_pred.shape[1]):

        y_true_f = K.flatten(y_true[:, i, ...])
        y_pred_f = K.flatten(y_pred[:, i, ...])

        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / \
               (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        loss_ = 1 - K.pow(dice, 1/beta)

        if debug:
            loss_ = tf.Print(loss_, [loss_], "Focal dice loss")

        loss += loss_

    return loss


def tversky(y_true, y_pred, alpha=0.3):

    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_pos = K.sum((1 - y_true_f)*y_pred_f)
    false_neg = K.sum(y_true_f*(1 - y_pred_f))

    return (true_pos + smooth) / \
           (true_pos + alpha*false_pos + (1-alpha)*false_neg + smooth)


def tversky_loss(y_true, y_pred, alpha=0.3):

    return 1 - tversky(y_true, y_pred, alpha=alpha)


def focal_tversky_loss(y_true, y_pred, alpha=0.3, gamma=3.):

    tvk = tversky(y_true, y_pred, alpha=alpha)
    return 1 - K.pow(tvk, 1/gamma)


def generalized_tversky_loss(y_true, y_pred, alpha=0.3):

    loss = 0.

    for i in range(y_pred.shape[1]):
        loss += 1 - tversky(
            y_true[:, i, ...],
            y_pred[:, i, ...],
            alpha=alpha)

    return loss


def generalized_focal_tversky_loss(y_true,
                                   y_pred,
                                   alpha=0.3,
                                   gamma=3.,
                                   debug=False):

    loss = 0.

    for i in range(y_pred.shape[1]):
        loss += focal_tversky_loss(y_true[:, i, ...],
                                   y_pred[:, i, ...],
                                   alpha=alpha,
                                   gamma=gamma)

    return loss
