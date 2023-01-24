from keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1.):

    """ Dice coefficient """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):

    """ Dice coefficient loss """

    return -dice_coefficient(y_true, y_pred)


def generalized_dice_coefficient(y_true, y_pred):

    """ Generalized dice coefficient """

    num = 0.
    denom = 0.
    for index in range(1, y_pred.shape[1]):
        y_true_f = K.flatten(y_true[:, index, ...])
        y_pred_f = K.flatten(y_pred[:, index, ...])
        w = 1. / (K.sum(y_true_f)**2 + 1e-12)
        num += w * K.sum(y_true_f * y_pred_f)
        denom += w * (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-12)

    return 2.*num/denom


def generalized_dice_coefficient_loss(y_true, y_pred):

    """ Generalized dice coefficient loss """

    return -generalized_dice_coefficient(y_true, y_pred)
