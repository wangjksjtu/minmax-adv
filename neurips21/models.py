import tensorflow as tf
import numpy as np


def modelA(x, logits=False, training=False):
    # MLP
    with tf.variable_scope('flatten'):
        z = tf.layers.flatten(x)
    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dense(z, units=64, activation=tf.nn.relu)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelB(x, logits=False, training=False):
    # All-CNNs
    with tf.variable_scope('convs0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=64, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3), strides=(2, 2),
                                padding='same')
        z = tf.layers.dropout(z, 0.5, training=training)

    with tf.variable_scope('convs1'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3), strides=(2, 2),
                                padding='same')
        z = tf.layers.dropout(z, 0.5, training=training)

    with tf.variable_scope('convs2'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(1, 1),
                                padding='valid', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=10, kernel_size=(1, 1), strides=(2, 2),
                                padding='valid')

    logits_ = tf.reduce_mean(z, [1, 2])
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelC(x, logits=False, training=False):
    # LeNet
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=6, kernel_size=[5, 5],
                             padding='valid', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=16, kernel_size=[5, 5],
                             padding='valid', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        z = tf.layers.flatten(z)

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=120, activation=tf.nn.relu)
        z = tf.layers.dense(z, units=84, activation=tf.nn.relu)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelD(x, logits=False, training=False):
    # LeNetV2
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        z = tf.layers.flatten(z)

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelE(x, logits=False, training=False, arch='VGG16'):
    # VGG
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    params = cfg[arch]

    '''
    def _batch_norm(x, name=''):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                inputs=x,
                decay=.9,
                center=True,
                scale=True,
                activation_fn=None,
                updates_collections=None,
                is_training=training)
    '''

    z = x
    for i, param in enumerate(params):
        if param == 'M':
            with tf.variable_scope('maxpool' + str(i)):
                z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        else:
            with tf.variable_scope('conv' + str(i)):
                z = tf.layers.conv2d(z, filters=param, kernel_size=[3, 3],
                                     padding='same', activation=tf.nn.relu)
                # z = _batch_norm(z)
                z = tf.layers.batch_normalization(z, training=training)

    with tf.variable_scope('mlp'):
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelF(x, logits=False, training=False, arch='ResNet34'):
    # ResNet34
    cfg = {
        'ResNet18':  ['BasicBlock', [2, 2, 2, 2]],
        'ResNet34':  ['BasicBlock', [3, 4, 6, 3]],
        'ResNet50':  ['BasicBlock', [3, 4, 6 ,3]],
        'ResNet101': ['BottleNeck', [3, 4, 23 ,3]],
        'ResNet152': ['BottleNeck', [3, 8, 36 ,3]],
    }
    params = cfg[arch]

    def _basic_block(x, in_planes, planes, stride=1, name=''):
        expansion = 1
        with tf.variable_scope('block' + name):
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False,
                                 strides=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)

            shortcut = tf.identity(x)
            if stride != 1 or in_planes != planes * expansion:
                shortcut = tf.layers.conv2d(shortcut, filters=expansion*planes,
                                            kernel_size=[1, 1],
                                            padding='same', use_bias=False,
                                            strides=(stride, stride))
                shortcut = tf.layers.batch_normalization(shortcut, training=training)

            z += shortcut
            z = tf.nn.relu(z)

        return z

    def _bottleneck(x, in_planes, planes, stride=1, name=''):
        expansion = 4
        with tf.variable_scope('bottleneck' + name):
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False,
                                 stride=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=expansion*planes, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)

            shortcut = tf.identity(x)
            if stride != 1 or in_planes != planes * expansion:
                shortcut = tf.layers.conv2d(shortcut, filters=expansion*planes,
                                            kernel_size=[1, 1],
                                            padding='same', use_bias=False,
                                            strides=(stride, stride))
                shortcut = tf.layers.batch_normalization(shortcut, training=training)

            z += shortcut
            z = tf.nn.relu(z)

        return z

    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3],
                            padding='same', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

    if params[0] == 'BasicBlock':
        block = _basic_block
        expansion = 1
    else:
        block =  _bottleneck
        expansion = 4
    num_blocks = params[1]

    in_planes = 64
    def _make_layer(z, block, in_planes, planes, num_blocks, stride, name=''):
        strides = [stride] + [1]*(num_blocks-1)
        for i, stride in enumerate(strides):
            z = block(z, in_planes, planes, stride, name=name+str(i))
            in_planes = planes * expansion
        return z, in_planes

    z, in_planes = _make_layer(z, block, in_planes, 64, num_blocks[0], stride=1, name='1_')
    z, in_planes = _make_layer(z, block, in_planes, 128, num_blocks[1], stride=2, name='2_')
    z, in_planes = _make_layer(z, block, in_planes, 256, num_blocks[2], stride=2, name='3_')
    z, in_planes = _make_layer(z, block, in_planes, 512, num_blocks[3], stride=2, name='4_')

    with tf.variable_scope('mlp'):
        z = _global_avg_pool(z)
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelG(x, logits=False, training=False):
    # Wide-ResNet (Madry's Lab CIFAR Challenge)
    def _batch_norm(name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                inputs=x,
                decay=.9,
                center=True,
                scale=True,
                activation_fn=None,
                updates_collections=None,
                is_training=training)

    def _residual(x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = _batch_norm('init_bn', x)
                x = _relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = _batch_norm('init_bn', x)
                x = _relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = _conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = _batch_norm('bn2', x)
            x = _relu(x, 0.1)
            x = _conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                        [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _conv(name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())

        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    with tf.variable_scope('input'):
        x = _conv('init_conv', x, 3, 3, 16, _stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = _residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    filters = [16, 160, 320, 640]

    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
        x = res_func(x, filters[0], filters[1], _stride_arr(strides[0]),
                     activate_before_residual[0])
    for i in range(1, 5):
        with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1], _stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
        x = res_func(x, filters[1], filters[2], _stride_arr(strides[1]),
                     activate_before_residual[1])
    for i in range(1, 5):
        with tf.variable_scope('unit_2_%d' % i):
            x = res_func(x, filters[2], filters[2], _stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
        x = res_func(x, filters[2], filters[3], _stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, 5):
        with tf.variable_scope('unit_3_%d' % i):
            x = res_func(x, filters[3], filters[3], _stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = _batch_norm('final_bn', x)
      x = _relu(x, 0.1)
      x = _global_avg_pool(x)

    with tf.variable_scope('logit'):
        logits_ = _fully_connected(x, 10)

    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelH(x, logits=False, training=False, arch='MobileNet'):
    # MobileNet
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def _block(x, in_planes, planes, stride=1, name=''):
        with tf.variable_scope('block' + name):
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[3, 3],
                                padding='same', use_bias=False,
                                strides=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=planes, kernel_size=[1, 1],
                                padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
        return z

    def _global_avg_pool(x):
        return tf.layers.average_pooling2d(x, pool_size=[2,2], strides=2)


    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3,3],
                            padding='same', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

    block = _block
    num_blocks = cfg

    def _make_layer(z, in_planes, name=''):
        for index,(i) in enumerate(num_blocks):
            planes = i if isinstance(i, int) else i[0]
            stride = 1 if isinstance(i, int) else i[1]
            z = block(z, in_planes, planes, stride, name=name+str(index))
            in_planes = planes
        return z, in_planes

    z, _ = _make_layer(z, in_planes=32)

    with tf.variable_scope('mlp'):
        z = _global_avg_pool(z)
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelI(x, logits=False, training=False, arch='MobileNetV2'):
    # MobileNetV2
    # NOTE: change stride 2 -> 1 for CIFAR10
    cfg = [(1,  16, 1, 1), (6,  24, 2, 1), (6,  32, 3, 2),
           (6,  64, 4, 2), (6,  96, 3, 1), (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def block(x, in_planes, out_planes, expansion, stride, name=''):
        with tf.variable_scope('block' + name):
            planes = expansion * in_planes
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[1, 1],
                                padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                padding='same', use_bias=False,
                                strides=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(z, filters=out_planes, kernel_size=[1, 1],
                                padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)
            shortcut = tf.identity(x)
            if stride == 1 or in_planes != planes:
                shortcut = tf.layers.conv2d(shortcut, filters=out_planes,
                                            kernel_size=[1, 1],
                                            padding='valid', use_bias=False)
                shortcut = tf.layers.batch_normalization(shortcut, training=training)
            z = z + shortcut if stride == 1 else z
        return z

    def _make_layer(z, in_planes, name=''):
        for key, (expansion, out_planes, num_blocks, stride) in enumerate(cfg):
            strides = [stride] + [1]*(num_blocks-1)
            for i, stride in enumerate(strides):
                z = block(z, in_planes, out_planes, expansion, stride, name=name+'_'+str(key)+'_'+str(i))
                in_planes = out_planes
        return z, in_planes

    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                            padding='valid', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

        z, in_planes = _make_layer(z, in_planes=32)

        z = tf.layers.conv2d(z, filters=1280, kernel_size=[1, 1],
                            padding='valid', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)


    with tf.variable_scope('mlp'):
        z = _global_avg_pool(z)
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelJ(x, logits=False, training=False):
    # GoogleNet
    def inception(z, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        # 1x1 conv branch
        b1 = tf.layers.conv2d(z, filters=n1x1, kernel_size=[1, 1],
                                padding='valid')
        b1 = tf.layers.batch_normalization(b1, training=training)
        b1 = tf.nn.relu(b1)

        # 1x1 conv -> 3x3 conv branch
        b2 = tf.layers.conv2d(z, filters=n3x3red, kernel_size=[1, 1],
                                padding='valid')
        b2 = tf.layers.batch_normalization(b2, training=training)
        b2 = tf.nn.relu(b2)
        b2 = tf.layers.conv2d(b2, filters=n3x3, kernel_size=[3, 3],
                                padding='same')
        b2 = tf.layers.batch_normalization(b2, training=training)
        b2 = tf.nn.relu(b2)

        # 1x1 conv -> 5x5 conv branch
        b3 = tf.layers.conv2d(z, filters=n5x5red, kernel_size=[1, 1],
                                padding='valid')
        b3 = tf.layers.batch_normalization(b3, training=training)
        b3 = tf.nn.relu(b3)
        b3 = tf.layers.conv2d(b3, filters=n5x5, kernel_size=[3, 3],
                                padding='same')
        b3 = tf.layers.batch_normalization(b3, training=training)
        b3 = tf.nn.relu(b3)
        b3 = tf.layers.conv2d(b3, filters=n5x5, kernel_size=[3, 3],
                                padding='same')
        b3 = tf.layers.batch_normalization(b3, training=training)
        b3 = tf.nn.relu(b3)

        # 3x3 pool -> 1x1 conv branch
        b4 = tf.layers.max_pooling2d(z, pool_size=[1, 1], padding='same',
                                    strides=1)
        b4 = tf.layers.conv2d(b4, filters=pool_planes, kernel_size=[1, 1],
                                padding='valid')
        b4 = tf.layers.batch_normalization(b4, training=training)
        b4 = tf.nn.relu(b4)

        return tf.concat([b1, b2, b3, b4], 3)

    def pre_layer(z):
        z = tf.layers.conv2d(z, filters=192, kernel_size=[3, 3],
                                padding='same')
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

        return z

    with tf.variable_scope('init_conv'):
        z = pre_layer(x)
        z = inception(z, 192,  64,  96, 128, 16, 32, 32)
        z = inception(z, 256, 128, 128, 192, 32, 96, 64)

        z = tf.layers.max_pooling2d(z, pool_size=[3, 3], strides=2,
                                    padding='same')

    with tf.variable_scope('part_1'):
        z = inception(z, 480, 192,  96, 208, 16,  48,  64)
        z = inception(z, 512, 160, 112, 224, 24,  64,  64)
        z = inception(z, 512, 128, 128, 256, 24,  64,  64)
        z = inception(z, 512, 112, 144, 288, 32,  64,  64)
        z = inception(z, 528, 256, 160, 320, 32, 128, 128)
        z = tf.layers.max_pooling2d(z, pool_size=[3, 3], strides=2,
                                    padding='same')

    with tf.variable_scope('part_2'):
        z = inception(z, 832, 256, 160, 320, 32, 128, 128)
        z = inception(z, 832, 384, 192, 384, 48, 128, 128)
        z = tf.layers.max_pooling2d(z, pool_size=[8, 8], padding='valid',
                                    strides=1)

    with tf.variable_scope('mlp'):
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def modelK(x, logits=False, training=False, arch='DenseNet'):
    # DenseNet
    cfg = {
        'DenseNet121': [[6,12,24,16], 32],
        'DenseNet169': [[6,12,32,32], 32],
        'DenseNet201': [[6,12,48,32], 32],
        'DenseNet161': [[6,12,36,24], 48],
        'DenseNet':    [[6,12,24,16], 12],   # densenet_cifar
    }
    params = cfg[arch]
    nblocks, growth_rate = params
    reduction = 0.5

    def _bottleneck(x, in_planes, growth_rate, name=''):
        with tf.variable_scope('bottleneck' + name):
            z = tf.layers.batch_normalization(x, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(x, filters=4*growth_rate, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)

            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(z, filters=growth_rate, kernel_size=[3, 3],
                                 padding='same', use_bias=False)
            z = tf.concat([z, x], axis=3)

        return z

    def _transition(x, in_planes, out_planes, name=''):
        with tf.variable_scope('transition' + name):
            z = tf.layers.batch_normalization(x, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(x, filters=out_planes, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)

            z = tf.layers.average_pooling2d(z, pool_size=[2, 2], strides=2)

        return z

    def _make_dense_layer(z, block, in_planes, nblock, name=''):
        for i in range(nblock):
            z = block(z, in_planes, growth_rate, name=name+str(i))
            in_planes += growth_rate

        return z

    num_planes = 2 * growth_rate
    block = _bottleneck
    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=num_planes, kernel_size=[3, 3],
                             padding='same', use_bias=False)

    with tf.variable_scope('dense_block0'):
        z = _make_dense_layer(z, block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        z = _transition(z, num_planes, out_planes)
        num_planes = out_planes

    with tf.variable_scope('dense_block1'):
        z = _make_dense_layer(z, block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        z = _transition(z, num_planes, out_planes)
        num_planes = out_planes

    with tf.variable_scope('dense_block2'):
        z = _make_dense_layer(z, block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        z = _transition(z, num_planes, out_planes)
        num_planes = out_planes

    with tf.variable_scope('dense_block3'):
        z = _make_dense_layer(z, block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate


    with tf.variable_scope('mlp'):
        z = tf.layers.batch_normalization(x, training=training)
        z = tf.nn.relu(z)
        z = tf.layers.average_pooling2d(z, pool_size=[4, 4], strides=4)
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y
