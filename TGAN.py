from tfGAN_indvBN import batcher,lrelu,safe_log,init_normal
import tensorflow as tf
import numpy as np
from os import sep, getcwd
from time import time
from os.path import join, exists

BATCH_NORM_DECAY = 0.999
BATCH_RENORM = False

def get_one_hot(targets, depth):
    return np.eye(depth)[np.array(targets).reshape(-1)]

class GANBase(object):
    name = 'GANBase'

    def __init__(self, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None, use_batch_norm_G=True,
                 use_batch_norm_D=False, name=None, log_and_save=True, seed=np.random.randint(int(1e8)), debug=False):
        # parameters
        self.n_noise = 100
        self.n_pixel = 32
        self.n_channel = 3
        self.n_class = 10
        if mask is None or mask.dtype is not bool:
            self.mask = np.ones(self.n_pixel * self.n_pixel, dtype=bool)
            self.mask[mask] = False
        else:
            self.mask = tf.constant(mask.reshape(), dtype=tf.float32, name='mask')
        self.mask = tf.constant(self.mask.reshape(1, self.n_pixel, self.n_pixel, 1), dtype=tf.float32, name='mask')
        self.batch_norm_G = use_batch_norm_G
        self.batch_norm_D = use_batch_norm_D
        self.seed = seed
        self.n_extra_generator_layers = n_extra_generator_layers
        self.n_extra_discriminator_layers = n_extra_discriminator_layers
        self.log_and_save = log_and_save
        self.debug = debug
        self.filename = self.name
        if name is not None:
            self.name += '_' + name
        if self.debug:
            self.name += '_debug'
        self.path = getcwd() + sep + 'output' + sep + self.filename + sep

        # network variables
        self.batch_ind = tf.placeholder(tf.int32, 0, 'batch_ind')
        self.batch_size = tf.placeholder(tf.int32, 0, 'batch_size')
        self.training = tf.placeholder(tf.bool, 1, 'training')
        # old labels to delete
        #self.input_x = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel), 'image')
        #self.input_y = tf.placeholder(tf.float32, (None, self.n_class), 'class')

        # new labels
        self.input_z_g = tf.placeholder(tf.float32, (None, self.n_noise), 'pure_noise')
        self.input_y_g=tf.placeholder(tf.float32,(None,self.n_class),'self_chosen_class')
        self.input_labeled_x = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel),
                                              'labeled_image')
        self.input_labeled_y = tf.placeholder(tf.float32, (None, self.n_class), 'labeled_class')
        self.input_x_c = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel),
                                        'unlabeled_image')

        # self.input_x = tf.Variable(self.input_x_ph, trainable=False, collections=[])
        # self.input_z_g = tf.Variable(self.input_z_g_ph, trainable=False, collections=[])

        # logging'
        self.saver = None
        self.writer_train = None
        self.writer_test = None

        # etc
        self.session = None

    def _build_generator(self, tensor=None, training=False, batch_norm=None):
        assert self.n_pixel % 16 == 0, "isize has to be a multiple of 16"
        nfilt = 2000
        csize = 4
        if tensor is None:
            tensor = self.input_z_g
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x, name=None):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, nfilt)):
                tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tf.reshape(tensor, [-1, 1, 1, self.n_noise]),
                                                                  nfilt, 4, 2, 'valid', use_bias=not batch_norm,
                                                                  kernel_initializer=init_normal(),
                                                                  name='conv'), name='bn'))

            # upscaling layers
            while csize < self.n_pixel / 2:
                with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt / 2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt / 2, 4, 2, 'same',
                                                                      use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))
                csize *= 2
                nfilt /= 2

            # extra layers
            for it in range(self.n_extra_generator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = tf.nn.relu(
                        bn(tf.layers.conv2d_transpose(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                      kernel_initializer=init_normal(),
                                                      name='conv'), name='bn'))
                    # TODO in original DCGAN struct, result is flattenned(3->1), not setting filtercount to 1 (self.n_channel)
            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, self.n_channel)):
                tensor = tf.layers.conv2d_transpose(tensor, self.n_channel, 4, 2, 'same', activation=tf.tanh,
                                                    kernel_initializer=init_normal(),
                                                    name='conv')

            # mask layer
            return tensor * self.mask

    def _build_discriminator_base(self, tensor=None, training=False, batch_norm=None):
        nfilt = 500
        if tensor is None:
            tensor = self.input_x
        if batch_norm is None:
            batch_norm = self.batch_norm_D
        if batch_norm:
            def bn(tensor, name=None):
                return tf.contrib.layers.batch_norm(tensor, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        # initial layer
        with tf.variable_scope('initial.{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 4, 2, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv'), name='bn'))
        nfilt /= 2
        csize = self.n_pixel / 2

        # extra layers
        for it in range(self.n_extra_discriminator_layers):
            with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))

        # downscaling layers
        while csize > 4:
            with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt * 2)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt * 2, 4, 2, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))
            nfilt *= 2
            csize /= 2

        return tensor

    def _build_loss(self, label_strength=1.):
        raise NotImplementedError

    def _start_logging_and_saving(self, sess, log=True, save=True):
        if self.log_and_save and (log or save):
            # saver to save model
            if save:
                self.saver = tf.train.Saver()
            # summary writer
            if log:
                self.writer_train = tf.summary.FileWriter(join(self.path, self.name, 'train'), sess.graph)
                self.writer_test = tf.summary.FileWriter(join(self.path, self.name, 'test'), sess.graph)

            print('Saving to ' + self.path)

    def _log(self, summary, counter=None, test=False):
        if self.log_and_save:
            if test:
                self.writer_test.add_summary(summary, counter)
            else:
                self.writer_train.add_summary(summary, counter)

    def _save(self, session, counter=None):
        if self.log_and_save:
            self.saver.save(session, join(self.path, self.name, self.name + '.ckpt'), counter)

    def _restore(self, session):
        if self.log_and_save:
            self.saver.restore(session, tf.train.latest_checkpoint(join(self.path, self.name)))

    def load(self, path=None):
        self._build_loss()
        self.session = tf.Session()
        self._start_logging_and_saving(None, log=False)
        if path is None:
            path = tf.train.latest_checkpoint(join(self.path, self.name))
        self.saver.restore(self.session, path)


class TRIGAN(GANBase):
    name = 'TRIGAN'
    """NEW"""

    # built from build_discriminator_base
    # from triple gan paper, used on svhn
    def _build_classifier_base(self, tensor=None, training=False, batch_norm=None):
        nfilt = 128
        if tensor is None:
            tensor = self.input_x_c
        if batch_norm is None:
            batch_norm = self.batch_norm_D
        if batch_norm:
            def bn(tensor, name=None):
                return tf.contrib.layers.batch_norm(tensor, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        # tf.layers.conv2d(inputs,filters,kernel_size,strides,padding...)
        def do(tensor, rate=0.5, name=None):
            return tf.contrib.layers.dropout(tensor, keep_prob=rate, is_training=training)

        # dropout before layers
        with tf.variable_scope('initial_dropout{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = tf.layers.dropout(tensor, rate=0.2, training=training, seed=self.seed, name='do')

        # initial layer
        for it in range(2):
            with tf.variable_scope('first_part-{0}.{1}-{2}'.format(it, self.n_channel, nfilt)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))
        with tf.variable_scope('first_part-last{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = do(lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 2, 'same', use_bias=not batch_norm,
                                                  kernel_initializer=init_normal(),
                                                  name='conv'), name='bn')), rate=0.5, name='do')

        nfilt = 256
        for it in range(2):
            with tf.variable_scope('second_part-{0}.{1}-{2}'.format(it, self.n_channel, nfilt)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))
        with tf.variable_scope('second_part-last{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = do(lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 2, 'same', use_bias=not batch_norm,
                                                  kernel_initializer=init_normal(),
                                                  name='conv'), name='bn')), rate=0.5, name='do')
        nfilt = 512
        with tf.variable_scope('third_part{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv'), name='bn'))
        nfilt = 256
        with tf.variable_scope('third_part{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 1, 1, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv'), name='bn'))
        nfilt = 128
        with tf.variable_scope('third_part{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 1, 1, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv'), name='bn'))

        with tf.variable_scope('last_layer{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = tf.reduce_mean(tensor, [1, 2], name='rm')
            tensor = lrelu(tf.layers.dense(tensor, self.n_class))

        # csize = self.n_pixel / 2

        # extra layers
        # for it in range(self.n_extra_discriminator_layers):
        #    with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
        #        tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
        #                                           kernel_initializer=init_normal(),
        #                                           name='conv'), name='bn'))

        # downscaling layers
        # while csize > 4:
        #    with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt * 2)):
        #        tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt * 2, 4, 2, 'same', use_bias=not batch_norm,
        #                                           kernel_initializer=init_normal(),
        #                                           name='conv'), name='bn'))
        #    nfilt *= 2
        #    csize /= 2

        return tensor

    def _build_classifier(self, tensor=None, training=False):
        if tensor is not None:
            input = tensor
            org_labels=None
        else:
            input = self.input_labeled_x
            org_labels=self.input_labeled_y
        with tf.variable_scope('classifier') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # classifier base
            tensor = self._build_classifier_base(tensor, training)

            # final layer
            d_out = self.n_class
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tensor, [-1, d_out])

        return input, out_logits

    def _build_generator(self, tensor=None, label=None, training=False, batch_norm=None):
        assert self.n_pixel % 16 == 0, "isize has to be a multiple of 16"
        nfilt = 2048
        csize = 4
        if label is None:
            if self.input_y_g is None:
                #TO BE EDITED
                #sample generated images
                batch_size=(self.input_z_g).shape[0]
                label=get_one_hot(np.repeat(np.tile(self.n_class),(batch_size/self.m_class)+1),depth=self.n_class)
                label=label[0:batch_size,:]
            else:
                #get label from input
                label=self.input_y_g
        if tensor is None:
            # add label to noise
            tensor = tf.concat([self.input_z_g, label], 1)
        else:
            # assuming tensor is a specific noise
            tensor = tf.concat([tensor, label], 1)
            #tensor = tf.concat([tensor, tf.one_hot(label, self.n_class)], 1)
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x, name=None):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            # return the same if bn is not aactivated
            bn = tf.identity

        with tf.variable_scope('generator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise + self.n_class, nfilt)):
                tensor = tf.nn.relu(
                    bn(tf.layers.conv2d_transpose(tf.reshape(tensor, [-1, 1, 1, self.n_noise + self.n_class]),
                                                  nfilt, 4, 2, 'valid', use_bias=not batch_norm,
                                                  kernel_initializer=init_normal(),
                                                  name='conv'), name='bn'))

            # upscaling layers
            while csize < self.n_pixel / 2:
                with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt // 2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt // 2, 4, 2, 'same',
                                                                      use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))
                csize *= 2
                nfilt //= 2

            # extra layers
            for it in range(self.n_extra_generator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = tf.nn.relu(
                        bn(tf.layers.conv2d_transpose(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                      kernel_initializer=init_normal(),
                                                      name='conv'), name='bn'))
                    # TODO in original DCGAN struct, result is flattenned(3->1), not setting filtercount to 1 (self.n_channel)
            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, self.n_channel)):
                tensor = tf.layers.conv2d_transpose(tensor, self.n_channel, 4, 2, 'same', activation=tf.tanh,
                                                    kernel_initializer=init_normal(),
                                                    name='conv')

            # mask layer
            return tensor * self.mask, label

    '''----END----'''

    # implementing discriminator with labels(not on every layer)
    def _build_discriminator(self, tensor=None, label=None, training=False, batch_norm=None):
        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()
            nfilt = 512
            # this means that labeled real data is  inputted
            if tensor is None and label is None:
                # add labels to input
                x_shape = self.input_labeled_x.get_shape()
                # WHY??
                #tensor = tf.concat([self.input_labeled_x,
                #                    tf.reshape(self.input_labeled_y, [-1, 1, 1, self.n_class]) * tf.ones(
                #                        [x_shape[0], x_shape[1], x_shape[2], self.n_class])], axis=3)
                tensor = tf.concat([self.input_labeled_x,tf.tile(tf.reshape(self.input_labeled_y, [-1, 1, 1, self.n_class]),[1,x_shape[1],x_shape[2],1])] , axis=3)
            elif tensor is None or label is None:
                print('Tensor and label must be both None or both exists')
                raise
            else:
                x_shape = tensor.get_shape()
                print(label.get_shape())
                label_copy=tf.tile(tf.reshape(label, [-1, 1, 1, self.n_class]),[1, x_shape[1], x_shape[2], 1])
                tensor = tf.concat([tensor,label_copy], axis=3)

            if batch_norm is None:
                batch_norm = self.batch_norm_D
            if batch_norm:
                def bn(tensor, name=None):
                    return tf.contrib.layers.batch_norm(tensor, is_training=training,
                                                        renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
            else:
                bn = tf.identity

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_channel+self.n_class, nfilt)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 4, 2, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))
            nfilt //= 2
            csize = self.n_pixel // 2

            # extra layers
            for it in range(self.n_extra_discriminator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                       kernel_initializer=init_normal(),
                                                       name='conv'), name='bn'))

            # downscaling layers
            while csize > 4:
                with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt * 2)):
                    tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt * 2, 4, 2, 'same', use_bias=not batch_norm,
                                                       kernel_initializer=init_normal(),
                                                       name='conv'), name='bn'))
                nfilt *= 2
                csize /= 2

                # final layer
                d_out = 2
                with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                    out_logits = tf.reshape(
                        tf.reduce_mean(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                        name='conv'), axis=3), [-1, d_out])
                    # TODO consider to output the labels instead of softmax&out_logits
        return tf.nn.softmax(out_logits), out_logits

    '''
    def _build_discriminator(self, tensor=None, training=False):
#Is it needed to add labels to start of discrminator input? A:YES
#Input tensor MUST be matrix+vector
        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # final layer
            d_out = 2
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

        return tf.nn.softmax(out_logits), out_logits
'''
    def _build_metrics(self):
        training=False
        consist_x, consist_logits = self._build_classifier(self.input_labeled_x, training=training)
        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.input_labeled_y, 0), predictions=tf.argmax(consist_logits, 0))
        #tf.summary.scalar('acc', acc)<-FIX THIS
        return acc

    def _build_loss(self, label_strength=1., training=False):
        # input  labels, get fake data (x_g,y_g)
        # TODO provide labels to output
        fake_x_g, fake_y_g = self._build_generator(
            training=training)  # tf.random_normal((self.batch_size, self.n_noise)))

        # input unlabeled pictures, get fake labels (x_c,y_c)
        # fake_x_c are real pictures
        fake_x_c, fake_y_c = self._build_classifier(training=training)
        consist_x, consist_logits = self._build_classifier(self.input_labeled_x,training=training)

        fake_label_2, fake_logits_2 = self._build_discriminator(fake_x_c, fake_y_c, training=training)
        fake_label, fake_logits = self._build_discriminator(fake_x_g, fake_y_g, training=training)
        real_label, real_logits = self._build_discriminator(training=training)
        label_goal = tf.concat((tf.ones((tf.shape(fake_logits)[0], 1)), tf.zeros((tf.shape(fake_logits)[0], 1))), 1)
        label_goal_2 = tf.concat((tf.ones((tf.shape(fake_logits_2)[0], 1)), tf.zeros((tf.shape(fake_logits_2)[0], 1))),
                                 1)
        label_goal_consist = tf.concat(
            (tf.ones((tf.shape(consist_logits)[0], 1)), tf.zeros((tf.shape(consist_logits)[0], 1))),
            1)
        label_smooth = tf.concat((label_strength * tf.ones((tf.shape(real_logits)[0], 1)),
                                  (1 - label_strength) * tf.ones((tf.shape(real_logits)[0], 1))), 1)

        # generator
        # self.lossG =
        # -safe_log(1 - fake_label[:, -1]) or -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1-label_goal))
        # -safe_log(fake_label[:, 0]) (better) or tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal) (best)
        lossG_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal))
        lossG = lossG_d
        # discriminator
        lossD_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits, labels=label_smooth))
        lossD_g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1 - label_goal))
        lossD_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits_2, labels=1 - label_goal_2))
        lossD = lossD_d + lossD_g + lossD_c
        # TODO buildlossC
        lossC_c = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=consist_logits, labels=self.input_labeled_y))
        lossC = lossC_c
        # summaries
        if training:
            tf.summary.image('fake', fake_x_c)
            tf.summary.image('real', self.input_labeled_x)
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label[:, -1])
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d', lossD_d)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('lossC', lossC)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD, lossC

    def train(self, labeled_x, labeled_y, unlabeled_x, test_x, test_y, n_ptepoch=0, n_epochs=25, n_batch=128,
              learning_rate=2e-4, label_strength=1.):

        # handle data
        # get count of data
        n_labeled = labeled_x.shape[0]
        n_unlabeled = unlabeled_x.shape[0]
        n_test = test_x.shape[0]
        # train = tf.constant(trainx, name='train')
        # test = tf.constant(testx, name='test')
        # dataset = tf.contrib.data.Dataset.from_tensor_slices(self.input_x)
        # iterator = dataset.make_initializable_iterator()
        # train = tf.contrib.data.Dataset.from_tensor_slices(trainx)
        # train = tf.contrib.data.Dataset.from_tensor_slices(testx)
        # iterator_train = train.make_initializable_iterator()

        # setup learning
        # train_batch = tf.train.shuffle_batch([train], n_batch, 50000, 10000, 2,
        #                                      enqueue_many=True, allow_smaller_final_batch=True, name='batch')
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD, lossC = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD, evalC = self._build_loss(label_strength=label_strength)
        accC = self._build_metrics()
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        tvarsC = [var for var in tf.trainable_variables() if 'classifier' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)
            adamC = tf.contrib.layers.optimize_loss(loss=lossC,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optC',
                                                    variables=tvarsC)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})

            start = time()
            nc, lc = 0, 0
            # pretrain classifier
            for ptepoch in range(n_ptepoch):
                for batch_index, n_batch_actual in batcher(n_labeled, n_batch):
                    nc += n_batch_actual
                    # train classifier first
                    temp = sess.run(adamC, {self.input_labeled_x: labeled_x[batch_index:batch_index + n_batch_actual],
                                            self.input_labeled_y: get_one_hot(labeled_y[batch_index:batch_index + n_batch_actual],self.n_class),
                                            self.input_z_g: np.random.randn(n_batch_actual, self.n_noise).astype(
                                                np.float32),
                                            self.input_y_g: get_one_hot(np.repeat(np.arange(self.n_class), max(1, int(n_batch / self.n_class))),self.n_class)})
                    lc += temp * n_batch_actual
                    if nc % (10 * n_batch) == 0:
                        print('pretrain {:d}/{:d}:  pretrain loss: {:f}  time: {:d} seconds' \
                              .format(n_ptepoch + 1, n_ptepoch, lc / nc, int(time() - start)))
            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                n, lg, ld,lc = 0, 0, 0,0

                for unlabeled_batch_index, n_batch_actual in batcher(n_unlabeled, n_batch):
                    if n % n_labeled == 0:
                        labeled_generator = batcher(n_labeled, n_batch)
                        labeled_bi, labeled_nba = next(labeled_generator)
                    else:
                        labeled_bi, labeled_nba = next(labeled_generator)
                    n += n_batch_actual
                    # discriminator
                    temp = sess.run(adamD,
                                    {self.input_labeled_x: labeled_x[labeled_bi:labeled_bi + labeled_nba],
                                     self.input_labeled_y: get_one_hot(labeled_y[labeled_bi:labeled_bi + labeled_nba],self.n_class),
                                     self.input_x_c: unlabeled_x[
                                                     unlabeled_batch_index:unlabeled_batch_index + n_batch_actual],
                                     self.input_z_g: np.random.randn(labeled_nba, self.n_noise).astype(np.float32),
                                     self.input_y_g: get_one_hot(
                                         np.repeat(np.arange(self.n_class), max(1, round(n_batch / self.n_class))),
                                         depth=self.n_class)})
                    ld += temp * n_batch_actual
                    # self._log(summary, step)
                    # print 'epoch {:d}/{:d} (part {:d}D):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds'\
                    #     .format(epoch, n_epochs, n, (lg + ld)/n, lg/n, ld/n, int(time() - start))
                    # generator
                    temp, summary, step = sess.run([adamD, merged_summary, global_step],
                                                   {self.input_labeled_x: labeled_x[
                                                                          labeled_bi:labeled_bi + labeled_nba],
                                                    self.input_labeled_y: get_one_hot(labeled_y[
                                                                                      labeled_bi:labeled_bi + labeled_nba],self.n_class),
                                                    self.input_x_c: unlabeled_x[
                                                                    unlabeled_batch_index:unlabeled_batch_index + n_batch_actual],
                                                    self.input_z_g: np.random.randn(n_batch_actual,
                                                                                    self.n_noise).astype(np.float32),
                                                    self.input_y_g: get_one_hot(
                                                        np.repeat(np.arange(self.n_class),
                                                                  max(1, round(n_batch / self.n_class))),
                                                        depth=self.n_class)})
                    lg += temp * n_batch_actual
                    # self._log(summary, step)
                    # print 'epoch {:d}/{:d} (part {:d}G):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds'\
                    #     .format(epoch, n_epochs, n, (lg + ld)/n, lg/n, ld/n, int(time() - start))
                    if n % (10 * n_batch) == 0:
                        self._log(summary, step)
                        print('epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                              .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, int(time() - start)))
                # save after each epoch
                self._save(sess, step)

                # evaluate
                n, lge, lde, lce = 0, 0, 0, 0
                nu, ace=0 ,0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    sd_label = get_one_hot(np.repeat(np.arange(self.n_class),max(1, round(n_batch_actual / self.n_class)+1)),depth=self.n_class)
                    sd_label_count=int(max(1, round(n_batch_actual / self.n_class)+1)*self.n_class)
                    nu+=sd_label_count
                    out = sess.run([evalG, evalD, evalC,accC],
                                   {self.training: [False],
                                    self.input_y_g: sd_label,
                                    self.input_labeled_x: test_x[batch_index:batch_index + n_batch_actual],
                                    self.input_labeled_y: get_one_hot(test_y[batch_index:batch_index + n_batch_actual],self.n_class),
                                    self.input_x_c: test_x[batch_index:batch_index + n_batch_actual],
                                    self.input_z_g: np.random.randn(sd_label_count, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
                    lce += out[2] * n_batch_actual
                    ace+=out[3]*sd_label_count
                print('epoch {:d}/{:d}:  classify acc: C: {:f} evaluation loss: {:f} (G: {:f}  D: {:f} C: {:f})  time: {:d} seconds' \
                      .format(epoch + 1, n_epochs,ace/nu, (lge + lde + lce) / n, lge / n, lde / n, lce / n,
                              int(time() - start)))

if __name__=='__main__':
    from data_process import load_cifar10

    labeled_data, labeled_label, unlabeled_data, test_data, test_label=load_cifar10()
    debug=False
    triple_gan = TRIGAN(debug=debug)
    triple_gan.train(labeled_data, labeled_label, unlabeled_data, test_data, test_label,
                 n_batch=100, label_strength=0.9)
    # a = 1