from keras import Input, Model
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, Activation, \
                         BatchNormalization, Add, Concatenate, UpSampling3D, \
                         PReLU, Softmax

from keras.optimizers import Adam
from keras.backend import set_image_data_format

try:
    from metrics import dice_coefficient_loss, dice_coefficient
except:
    from src.metrics import dice_coefficient_loss, dice_coefficient

set_image_data_format("channels_first")


def unet_3d(input_shape,
            n_labels=1,
            depth=4,
            n_base_filters=32,
            activation='sigmoid',
            loss=dice_coefficient_loss,
            metrics=dice_coefficient,
            initial_learning_rate=0.00001,
            batch_norm=True,
            skip_type='concat',
            down_type='max'):

    """ 3D U-Net """

    inputs = Input(input_shape)
    current_layer = inputs

    skip_layers = []
    for layer_depth in range(depth):
        n_filters = n_base_filters*(2**layer_depth)
        layer1 = conv_block(current_layer,
                            n_filters,
                            batch_norm=batch_norm)
        layer2 = conv_block(layer1,
                            n_filters*2,
                            batch_norm=batch_norm)
        skip_layers.append(layer2)

        if layer_depth == depth-1:  # Move on to synthesis path
            current_layer = layer2
        else:  # Continue on analysis path
            if down_type == 'max':
                current_layer = MaxPooling3D()(layer2)
            elif down_type == 'conv':
                current_layer = conv_block(layer2,
                                           n_filters,
                                           strides=(2, 2, 2),
                                           batch_norm=batch_norm)

    for layer_depth in range(depth-2, -1, -1):
        n_filters = n_base_filters*(2**(layer_depth+1))

        if skip_type == 'add':
            layer1 = deconv_block(current_layer,
                                  n_filters,
                                  batch_norm=batch_norm)
            layer2 = Add()([skip_layers[layer_depth], layer1])
        elif skip_type == 'concat':
            layer1 = deconv_block(current_layer,
                                  n_filters*2,
                                  batch_norm=batch_norm)
            layer2 = Concatenate(axis=1)([skip_layers[layer_depth], layer1])

        current_layer = conv_block(layer2,
                                   n_filters,
                                   batch_norm=batch_norm)
        current_layer = conv_block(current_layer,
                                   n_filters,
                                   batch_norm=batch_norm)

    final_convolution = Conv3D(n_labels, kernel_size=(1, 1, 1))(current_layer)

    if activation == 'softmax':
        act = Softmax(axis=1)(final_convolution)
    else:
        act = Activation(activation)(final_convolution)

    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate),
                  loss=loss, metrics=metrics)

    return model


def vnet_3d(input_shape,
            n_labels=1,
            depth=5,
            n_base_filters=16,
            activation='sigmoid',
            loss=dice_coefficient_loss,
            metrics=dice_coefficient,
            initial_learning_rate=0.00001,
            batch_norm=True):

    """ V-Net """

    inputs = Input(input_shape)
    current_layer = inputs

    # Analysis path
    skip_layers = []
    for layer_depth in range(depth):
        n_filters = n_base_filters*(2**layer_depth)
        n_conv = layer_depth + 1
        if n_conv > 3:
            n_conv = 3

        if layer_depth < depth - 1:
            current_layer, skip_layer = vnet_block(current_layer,
                                                   n_conv,
                                                   n_filters,
                                                   direction='down',
                                                   return_skip=True,
                                                   batch_norm=batch_norm)
            skip_layers.append(skip_layer)
        else:
            current_layer = vnet_block(current_layer, n_conv, n_filters,
                                       direction='up',
                                       return_skip=False,
                                       batch_norm=batch_norm)

    # Synthesis path
    for layer_depth in range(depth-2, 0, -1):
        n_filters = n_base_filters*(2**(layer_depth))
        current_layer = vnet_block(current_layer, n_conv, n_filters,
                                   direction='up',
                                   skip_layer=skip_layers[layer_depth],
                                   return_skip=False,
                                   batch_norm=batch_norm)

    final_convolution = Conv3D(n_labels, kernel_size=(1, 1, 1))(current_layer)

    if activation == 'softmax':
        act = Softmax(axis=1)(final_convolution)
    else:
        act = Activation(activation)(final_convolution)

    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate),
                  loss=loss, metrics=metrics)

    return model


def vnet_block(input_layer, n_conv, n_filters,
               direction='down',
               skip_layer=None,
               return_skip=True,
               batch_norm=False,
               activation='prelu'):

    """ V-Net block """

    layer = input_layer

    if skip_layer is not None:
        layer = Add()([layer, skip_layer])

    layer = n_conv_block(layer, 2, n_filters,
                         kernel_size=(5, 5, 5),
                         batch_norm=batch_norm)
    skip_layer = Add()([input_layer, layer])

    if direction == 'down':
        out_layer = conv_block(skip_layer, int(n_filters*2),
                               kernel_size=(2, 2, 2),
                               strides=(2, 2, 2),
                               batch_norm=batch_norm,
                               activation=activation)
    elif direction == 'up':
        out_layer = deconv_block(skip_layer, int(n_filters/2),
                                 kernel_size=(2, 2, 2),
                                 strides=(2, 2, 2),
                                 batch_norm=batch_norm,
                                 activation=activation)

    if return_skip:
        return out_layer, skip_layer
    else:
        return out_layer


def unetpp_3d(input_shape,
              n_labels=1,
              depth=5,
              n_base_filters=16,
              activation='sigmoid',
              loss=dice_coefficient_loss,
              metrics=dice_coefficient,
              initial_learning_rate=0.00001,
              batch_norm=True):

    """ U-Net++ """

    layers = []

    # Backbone
    inputs = Input(input_shape)
    for l in range(depth):
        n_filters = n_base_filters*(2**l)
        if l != 0:
            layer = MaxPooling3D()(layers[l-1][0])
        else:
            layer = inputs
        layers.append([n_conv_block(layer, 2, n_filters)])

    # Decoder
    for j in range(1, depth):
        for i in range(depth-j-1, -1, -1):
            # print('(%i,%i)' % (i+1, j-1))
            n_filters = n_base_filters*(2**i)
            layer = UpSampling3D()(layers[i+1][j-1])
            # layer = deconv_block(layers[i+1][j-1], n_filters)
            skip = layers[i][0:j]
            layer = Concatenate(axis=1)(skip + [layer])
            layers[i].append(n_conv_block(layer, 2, n_filters))

    # final_layer = layers[0][-1]

    layer = Concatenate(axis=1)(layers[0][1:])
    # layer = conv_block(layer, n_base_filters)

    final_convolution = Conv3D(n_labels, kernel_size=(1, 1, 1))(layer)

    if activation == 'softmax':
        act = Softmax(axis=1)(final_convolution)
    else:
        act = Activation(activation)(final_convolution)

    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]
    # metrics = ['categorical_crossentropy']

    model.compile(optimizer=Adam(lr=initial_learning_rate),
                  loss=loss, metrics=metrics)

    return model


def conv_block(input_layer, n_filters,
               kernel_size=(3, 3, 3),
               strides=(1, 1, 1),
               batch_norm=True,
               padding='same',
               activation='relu'):

    """ Convolution block """

    layer = Conv3D(n_filters, kernel_size,
                   strides=strides,
                   padding=padding,
                   data_format='channels_first')(input_layer)
    if batch_norm:
        layer = BatchNormalization(axis=1)(layer)
    if activation == 'prelu':
        return PReLU()(layer)
    else:
        return Activation(activation)(layer)


def n_conv_block(input_layer, n_conv, n_filters,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 batch_norm=True,
                 padding='same',
                 activation='relu'):

    """ n-Convolution block """

    layer = input_layer
    for n in range(n_conv):
        layer = conv_block(layer, n_filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           batch_norm=batch_norm,
                           padding=padding,
                           activation=activation)
    return layer


def deconv_block(input_layer, n_filters, kernel_size=(3, 3, 3),
                 strides=(2, 2, 2),
                 batch_norm=True,
                 padding='same',
                 activation='relu'):

    """ Deconvolution block """

    layer = Conv3DTranspose(n_filters, kernel_size,
                            strides=strides,
                            padding=padding,
                            data_format='channels_first')(input_layer)
    if batch_norm:
        layer = BatchNormalization(axis=1)(layer)
    if activation == 'prelu':
        return PReLU()(layer)
    else:
        return Activation(activation)(layer)


def upsamp_block(input_layer,
                 size=(2, 2, 2),
                 batch_norm=True):

    """ Upsampling block """

    layer = UpSampling3D(size=size)(input_layer)

    if batch_norm:
        layer = BatchNormalization(axis=1)(layer)
    return Activation('relu')(layer)


def res_block(input_layer, n_filters,
              kernel_size=(3, 3, 3),
              batch_norm=True,
              padding='same'):

    """ Residual block """

    layer = conv_block(input_layer, n_filters,
                       kernel_size=(1, 1, 1),
                       batch_norm=batch_norm)
    layer = conv_block(layer, n_filters, batch_norm=batch_norm)
    layer = Conv3D(n_filters, kernel_size,
                   strides=(1, 1, 1),
                   padding=padding,
                   data_format='channels_first')(layer)
    if batch_norm:
        layer = BatchNormalization(axis=1)(layer)
    skip_layer = conv_block(input_layer, n_filters, kernel_size=(1, 1, 1))
    return Add()([skip_layer, layer])


def fc_dense_net(input_shape,
                 n_labels=1,
                 n_conv_filters=16,
                 n_dense_filters=8,
                 depth=3,
                 dense_offset=3,
                 activation='sigmoid',
                 loss=dice_coefficient_loss,
                 metrics=dice_coefficient,
                 initial_learning_rate=1e-4):

    inputs = Input(input_shape)
    skip_layers = []

    # Initial convolution
    layer = Conv3D(n_conv_filters, (3, 3, 3),
                   padding='same')(inputs)
    layer = BatchNormalization(axis=1)(layer)
    layer = Activation('relu')(layer)

    # Encoder
    for n in range(depth):
        n_iter = n + dense_offset
        dense_layer = dense_block(layer, n_dense_filters,
                                  n_iter=n_iter)
        skip_layer = Concatenate(axis=1)([dense_layer, layer])
        skip_layers.append(skip_layer)
        n_filters = (n_dense_filters*n_iter) + n_conv_filters
        layer = transition_down(skip_layer, n_filters=n_filters)

    layer = dense_block(layer, n_dense_filters,
                        n_iter=depth + dense_offset)

    # Decoder
    for n, skip_layer in enumerate(skip_layers[::-1]):
        n_iter = depth + dense_offset - n - 1
        n_filters = n_dense_filters*n_iter
        layer = transition_up(layer, n_filters=n_filters)
        layer = Concatenate(axis=1)([layer, skip_layer])
        layer = dense_block(layer, n_dense_filters,
                            n_iter=n_iter)

    layer = Conv3D(n_labels, (1, 1, 1),
                   padding='same')(layer)

    if activation == 'softmax':
        outputs = Softmax(axis=1)(layer)
    else:
        outputs = Activation(activation)(layer)

    model = Model(inputs=inputs, outputs=outputs)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate),
                  loss=loss, metrics=metrics)

    return model


def dense_block(layer, n_filters,
                n_iter=8,
                activation='relu',
                kernel_size=(3, 3, 3),
                dropout_rate=0.1):

    skip_layers = []

    for n in range(n_iter):
        skip_layer = layer
        layer = BatchNormalization(axis=1)(layer)
        layer = Activation(activation)(layer)
        layer = Conv3D(n_filters, kernel_size,
                       padding='same')(layer)
        # layer = Dropout(rate=dropout_rate)(layer)
        skip_layers.append(layer)
        if n < n_iter - 1:
            layer = Concatenate(axis=1)([skip_layer, layer])
        else:
            layer = Concatenate(axis=1)(skip_layers)

    return layer


def transition_down(layer, n_filters=1,
                    activation='relu',
                    dropout_rate=0.1):

    layer = BatchNormalization(axis=1)(layer)
    layer = Activation(activation)(layer)
    layer = Conv3D(n_filters, (1, 1, 1),
                   padding='same')(layer)
    # layer = Dropout(rate=dropout_rate)(layer)
    layer = MaxPooling3D()(layer)

    return layer


def transition_up(layer, n_filters=1,
                  activation='relu',
                  kernel_size=(3, 3, 3),):

    return Conv3DTranspose(
        n_filters, kernel_size,
        activation=activation,
        padding='same',
        strides=(2, 2, 2)
    )(layer)


def dilated_fc_dense_net(
    input_shape,
    n_labels=1,
    n_conv_filters=16,
    n_dense_filters=8,
    depth=3,
    dense_offset=3,
    activation='sigmoid',
    loss=dice_coefficient_loss,
    metrics=dice_coefficient,
    initial_learning_rate=1e-4):

    inputs = Input(input_shape)
    skip_layers = []

    # Initial convolution
    layer = Conv3D(n_conv_filters, (3, 3, 3),
                   padding='same')(inputs)
    layer = BatchNormalization(axis=1)(layer)
    layer = Activation('relu')(layer)

    # Encoder
    for n in range(depth):
        dilations = dil_seq(n)
        dense_layer = dilated_dense_block(
            layer,
            dilations,
            n_filters=n_dense_filters
        )
        skip_layer = Concatenate(axis=1)([dense_layer, layer])
        skip_layers.append(skip_layer)
        n_filters = (n_dense_filters*len(dilations)) + n_conv_filters
        layer = transition_down(skip_layer, n_filters=n_filters)

    layer = dense_block(layer, n_dense_filters,
                        n_iter=depth + dense_offset)

    # Decoder
    for n, skip_layer in enumerate(skip_layers[::-1]):
        n_iter = depth + dense_offset - n - 1
        n_filters = n_dense_filters*n_iter
        layer = transition_up(layer, n_filters=n_filters)
        layer = Concatenate(axis=1)([layer, skip_layer])
        layer = dense_block(layer, n_dense_filters,
                            n_iter=n_iter)

    layer = Conv3D(n_labels, (1, 1, 1),
                   padding='same')(layer)

    if activation == 'softmax':
        outputs = Softmax(axis=1)(layer)
    else:
        outputs = Activation(activation)(layer)

    model = Model(inputs=inputs, outputs=outputs)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate),
                  loss=loss, metrics=metrics)

    return model


def dil_seq(n_dil):
    dil = [1]
    for n in range(n_dil + 2):
        dil.append(int(2**n))
    return dil


def dilated_dense_block(layer, dilations,
                        n_filters=8,
                        kernel_size=(3, 3, 3),
                        dilation_rate=1,
                        dropout_rate=0.2,
                        padding='same'):

    debug = False

    skip_layers = []

    for n, dil in enumerate(dilations):

        bn_layer = BatchNormalization(axis=1)(layer)
        act_layer = Activation('elu')(bn_layer)
        skip_layer = Conv3D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding=padding)(act_layer)
        skip_layers.append(skip_layer)

        if n < len(dilations):
            layer = Concatenate(axis=1)([layer, skip_layer])

        if debug:
            print('dilated_dense_block N: %i' % dil)
            print('skip_layer.shape')
            print(skip_layer.shape)

    output_layer = Concatenate(axis=1)(skip_layers)

    return output_layer