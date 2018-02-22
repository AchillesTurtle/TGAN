import numpy as np
import tensorflow as tf
from collections import OrderedDict
from time import time
from os import sep, getcwd
from os.path import join, exists


BATCH_NORM_DECAY = 0.999
BATCH_RENORM = False


def batcher(n_data, batch_size):
    for ind in range(0, n_data - batch_size + 1, batch_size):
        yield ind, np.minimum(batch_size, n_data - ind)


def shuffle_data(data, label=None, nitems=None):
    if data.__class__ is list or label:
        if nitems is None:
            if data.__class__ is list:
                nitems = len(data[0])
                ind = np.random.permutation(nitems)
                data = [x[ind] for x in data]
            else:
                nitems = len(data)
                ind = np.random.permutation(nitems)
                data = data[ind]
        if label is not None:
            label = label[ind]
            return data, label
    else:
        np.random.shuffle(data)
    return data


def lrelu(tensor, name=None):
    return tf.maximum(tensor, 0.2 * tensor, name=name)


def init_normal():
    return tf.random_normal_initializer(0, 0.02, None, tf.float32)

def safe_log(logit):
    return tf.log(tf.where(tf.equal(logit, 0.), tf.ones_like(logit), logit))

class NanInfException (Exception):
    pass


class GANBase (object):

    name = 'GANBase'

    def __init__(self, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None, use_batch_norm_G=True,
                 use_batch_norm_D=False, name=None, log_and_save=True, seed=np.random.randint(int(1e8)), debug=False):
        # parameters
        self.n_noise = 100
        self.n_pixel = 32
        self.n_channel = 1
        if mask.dtype is not bool:
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
        self.training = tf.placeholder(tf.bool, 0, 'training')
        self.input_x = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel), 'image')
        self.input_n = tf.placeholder(tf.float32, (None, self.n_noise), 'noise')
        # self.input_x = tf.Variable(self.input_x_ph, trainable=False, collections=[])
        # self.input_n = tf.Variable(self.input_n_ph, trainable=False, collections=[])

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
            tensor = self.input_n
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
            while csize < self.n_pixel/2:
                with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt/2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt/2, 4, 2, 'same',
                                                                      use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))
                csize *= 2
                nfilt /= 2

            # extra layers
            for it in range(self.n_extra_generator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))

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
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt*2, 4, 2, 'same', use_bias=not batch_norm,
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


class DCGAN (GANBase):

    name = 'DCGAN'

    def _build_discriminator(self, tensor=None, training=False):

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

    def _build_loss(self, label_strength=1., training=False):
        fake = self._build_generator(training=training) #tf.random_normal((self.batch_size, self.n_noise)))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label, real_logits = self._build_discriminator(training=training)
        label_goal = tf.concat((tf.ones((tf.shape(fake_logits)[0], 1)), tf.zeros((tf.shape(fake_logits)[0], 1))), 1)
        label_smooth = tf.concat((label_strength * tf.ones((tf.shape(fake_logits)[0], 1)),
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        # self.lossG =
        # -safe_log(1 - fake_label[:, -1]) or -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1-label_goal))
        # -safe_log(fake_label[:, 0]) (better) or tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal) (best)
        lossG = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal))

        # discriminator
        lossD_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits, labels=label_smooth))
        lossD_g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1-label_goal))
        lossD = lossD_d + lossD_g

        # summaries
        if training:
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label[:, -1])
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d', lossD_d)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD

    def train(self, trainx, testx, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train = trainx.shape[0]
        n_test = testx.shape[0]
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
        lossG, lossD = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
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

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                n, lg, ld = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train, n_batch):
                    n += n_batch_actual
                    # discriminator
                    temp = sess.run(adamD,
                                    {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                     self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    ld += temp * n_batch_actual
                    # self._log(summary, step)
                    # print 'epoch {:d}/{:d} (part {:d}D):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds'\
                    #     .format(epoch, n_epochs, n, (lg + ld)/n, lg/n, ld/n, int(time() - start))
                    # generator
                    temp, summary, step = sess.run([adamD, merged_summary, global_step],
                                                   {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
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
                n, lge, lde = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    out = sess.run([evalG, evalD],
                                   {self.training: [False],
                                    self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
              #  print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
              #      .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, int(time() - start))


class SSGAN (GANBase):

    name = 'SSGAN'

    def __init__(self, n_y, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None,
                 use_batch_norm_G=True, use_batch_norm_D=False,
                 name=None, seed=np.random.randint(int(1e8)), log_and_save=True, debug=False):

        # label variables
        self.n_y = n_y
        self.input_y = tf.placeholder(tf.float32, (None, self.n_y), 'label')
        self.input_x_l = tf.placeholder(tf.float32, (None, 32, 32, 1), 'labeled_image')

        # init
        # if not hasattr(self, 'name'):
        #     self.name = 'SSGAN'
        super(SSGAN, self).__init__(n_extra_generator_layers=n_extra_generator_layers,
                                    n_extra_discriminator_layers=n_extra_discriminator_layers,
                                    mask=mask, use_batch_norm_G=use_batch_norm_G, use_batch_norm_D=use_batch_norm_D,
                                    name=name, log_and_save=log_and_save, seed=seed, debug=debug)

    def _build_discriminator(self, tensor=None, training=False):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # final layer
            d_out = 1 + self.n_y
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

        return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):

        fake = self._build_generator(training=training) #tf.random_normal((self.batch_size, self.n_noise)))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator(self.input_x_l, training=training)
        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1)))

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            tf.summary.image('confusion_matrix', tf.reshape(tf.confusion_matrix(
                tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16), [1, self.n_y, self.n_y, 1]))
            # discriminator performance
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD

    def train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u.shape[0]
        n_train_l = trainx_l.shape[0]
        n_test = testx.shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
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

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            if self.debug:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                step = 0
                n, lg, ld = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                    n += n_batch_actual
                    # discriminator
                    randind = np.random.choice(n_train_l, n_batch_actual)
                    _, summary, temp = sess.run([adamD, merged_summary, lossD],
                                                {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                                 self.input_x_l: trainx_l[randind],
                                                 self.input_y: trainy[randind],
                                                 self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    ld += temp * n_batch_actual
                    # generator
                    _, summary, step, temp = sess.run([adamG, merged_summary, global_step, lossG],
                                                      {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                                       self.input_x_l: trainx_l[randind],
                                                       self.input_y: trainy[randind],
                                                       self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lg += temp * n_batch_actual
                    if n % (1 * n_batch) == 0:
                        self._log(summary, step)
                        print('epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, int(time() - start)))
                    if (n % 100 * n_batch) == 0:
                        # evaluate
                        m, lge, lde = 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_test, n_batch):
                            m += n_batch_actual
                            out = sess.run([evalG, evalD],
                                           {self.input_x: testx[batch_index:batch_index + n_batch_actual],
                                            self.input_x_l: testx[batch_index:batch_index + n_batch_actual],
                                            self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                            self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                            lge += out[0] * n_batch_actual
                            lde += out[1] * n_batch_actual
                        print('epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, (lge + lde) / m, lge / m, lde / m, int(time() - start)))
                        self._save(sess, step)
                # save after each epoch
                self._save(sess, step)

                # evaluate
                n, lge, lde = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    out = sess.run([evalG, evalD],
                                   {self.input_x: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_x_l: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
                print('epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                    .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, int(time() - start)))


class VEEGAN (SSGAN):

    name = 'VEEGAN'

    def _build_encoder(self, tensor=None, training=False):

        with tf.variable_scope('encoder') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], self.n_noise)):
                tensor = tf.reshape(tf.layers.conv2d(tensor, self.n_noise, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                     name='conv'), [-1, self.n_noise])

        return tensor

    def _build_discriminator(self, tensor=None, encoding_tensor=None, training=False):

        if encoding_tensor is None:
            encoding_tensor = self._build_encoder(training=training)

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # concatenate encoding
            tensor = tf.contrib.layers.flatten(tensor)
            tensor = tf.concat((tensor, encoding_tensor), axis=1)
            nfilt = tensor.shape[-1]

            # # dense layer
            # with tf.variable_scope('dense.{0}-{1}'.format(nfilt, nfilt)):
            #     tensor = (tf.layers.dense(tensor, nfilt, activation=lrelu, kernel_initializer=init_normal(),
            #                                 use_bias=not self.batch_norm, name='dense'))

            # final layers
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, 1 + self.n_y)):
                out_logits = tf.layers.dense(tensor, 1 + self.n_y, kernel_initializer=init_normal(), name='dense')

        return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):
        # networks
        fake = self._build_generator(training=training)
        fake_encoding = self._build_encoder(fake, training=training)
        fake_label, fake_logits = self._build_discriminator(fake, self.input_n, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator(self.input_x_l, training=training)

        # labels
        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # encoder loss
        lossF = tf.reduce_mean(tf.squared_difference(self.input_n, fake_encoding))

        # generator loss
        lossG = tf.reduce_mean(-safe_log(fake_label[:, 0])) + lossF

        # discriminator loss
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(real_label_u[:, 0]))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            tf.summary.image('confusion_matrix', tf.reshape(tf.confusion_matrix(
                tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16), [1, self.n_y, self.n_y, 1]))
            # discriminator performance
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('lossF', lossF)
            tf.summary.scalar('loss', lossG + lossD + lossF)

        return lossG, lossD, lossF

    def train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u.shape[0]
        n_train_l = trainx_l.shape[0]
        n_test = testx.shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD, lossF = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD, evalF = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        tvarsF = [var for var in tf.trainable_variables() if 'encoder' in var.name]
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
            adamF = tf.contrib.layers.optimize_loss(loss=lossF,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optF',
                                                    variables=tvarsF)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            if self.debug:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                step = 0
                n, lg, ld, lf = 0, 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                    n += n_batch_actual
                    # discriminator
                    randind = np.random.choice(n_train_l, n_batch_actual)
                    _, temp = sess.run([adamD, lossD],
                                       {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                        self.input_x_l: trainx_l[randind],
                                        self.input_y: trainy[randind],
                                        self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    ld += temp * n_batch_actual
                    # encoder
                    _, temp = sess.run([adamF, lossF],
                                       {self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lf += temp * n_batch_actual
                    # generator
                    _, summary, step, temp = sess.run([adamG, merged_summary, global_step, lossG],
                                                      {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                                       self.input_x_l: trainx_l[randind],
                                                       self.input_y: trainy[randind],
                                                       self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lg += temp * n_batch_actual
                    if n % (1 * n_batch) == 0:
                        self._log(summary, step)
                       # print 'epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                        #    .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, lf / n, int(time() - start))
                # save after each epoch
                self._save(sess, step)

                # evaluate
                n, lge, lde, lfe = 0, 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    out = sess.run([evalG, evalD, evalF, merged_summary],
                                   {self.input_x: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_x_l: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
                    lfe += out[2] * n_batch_actual
                self._log(out[3], step, test=True)
                print('epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                    .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, lfe / n, int(time() - start)))


# WIP
class WGAN (GANBase):

    name = 'WGAN'

    def _build_discriminator(self, tensor=None):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, batch_norm=False)

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], 1)):
                tensor = tf.reshape(tf.layers.conv2d(tensor, 1, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, 1])

        return tensor

    def _build_loss(self, label_strength=1., training=False, penalty_strength=10):
        # networks
        fake = self._build_generator(training=training)
        epsilon = tf.random_uniform((tf.shape(fake)[0], 1, 1, 1), 0.0, 1.0)
        semifake = epsilon * self.input_x + (1 - epsilon) * fake
        fake_logits = self._build_discriminator(fake)
        semifake_logits = self._build_discriminator(semifake)
        real_logits = self._build_discriminator()

        # generator loss
        lossG = -tf.reduce_mean(fake_logits)

        # discriminator loss
        lossD_d = -tf.reduce_mean(real_logits)
        lossD_g = -lossG
        norm_gradD = tf.norm(tf.reshape(tf.gradients(semifake_logits, semifake)[0], [tf.shape(semifake)[0], -1]), axis=1)
        lossD_p = penalty_strength * tf.reduce_mean(tf.square(norm_gradD - 1))
        lossD = lossD_d + lossD_g + lossD_p

        # summaries
        if training:
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            tf.summary.histogram('distance', real_logits - fake_logits)
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d', lossD_d)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD_p', lossD_p)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD

    def train(self, trainx, testx, n_epochs=25, n_batch=128, learning_rate=1e-4, label_strength=1.,
              penalty_stength=10, n_updated_d=5):

        # handle data
        n_train = trainx.shape[0]
        n_test = testx.shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(label_strength=label_strength, penalty_strength=penalty_stength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength, penalty_strength=penalty_stength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0, beta2=0.9),
                                                    # clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate/2,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0, beta2=0.9),
                                                    # clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                n, lg, ld = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train, n_batch):
                    n += n_batch_actual
                    # discriminator
                    _, temp = sess.run([adamD, lossD],
                                       {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                        self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    if n % (n_updated_d * n_batch) == 0:
                        ld += temp * n_batch_actual
                        # generator
                        _, summary, step, temp = sess.run([adamG, merged_summary, global_step, lossG],
                                                          {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                                           self.input_n: np.random.randn(n_batch_actual,
                                                                                         self.n_noise).astype(np.float32)})
                        lg += temp * n_batch_actual
                        self._log(summary, step)
                       # print 'epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                      #      .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, int(time() - start))
                # save after each epoch
                self._save(sess, step)

                # evaluate
                n, lge, lde = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    out = sess.run([evalG, evalD, merged_summary],
                                   {self.input_x: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
                    self._log(out[2], step, test=True)
               # print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
              #      .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, int(time() - start))


# WIP
class WACGAN (SSGAN):

    name = 'WSSGAN'

    def _build_discriminator(self, tensor=None):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, batch_norm=False)

            # final W layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], 1)):
                w = tf.reshape(tf.layers.conv2d(tensor, 1, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                name='conv'), [-1, 1])

            # final AC layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], self.n_y)):
                ac_logits = tf.reshape(tf.layers.conv2d(tensor, self.n_y, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                        name='conv'), [-1, self.n_y])

        return w, tf.nn.softmax(ac_logits), ac_logits

    def _build_loss(self, label_strength=1., training=False, penalty_strength=10):
        # networks
        fake = self._build_generator(training=training)
        epsilon = tf.random_uniform((tf.shape(fake)[0], 1, 1, 1), 0.0, 1.0)
        semifake_u = epsilon * self.input_x + (1 - epsilon) * fake
        semifake_l = epsilon * self.input_x_l + (1 - epsilon) * fake
        fake_wlogits, fake_acclass, fake_aclogits = self._build_discriminator(fake)
        semifake_wlogits_u, semifake_acclass_u, semifake_aclogits_u = self._build_discriminator(semifake_u)
        semifake_wlogits_l, semifake_acclass_l, semifake_aclogits_l = self._build_discriminator(semifake_l)
        real_wlogits_u, real_acclass_u, real_aclogits_u = self._build_discriminator()
        real_wlogits_l, real_acclass_l, real_aclogits_l = self._build_discriminator(self.input_x_l)
        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(real_aclogits_l)[0], 1))), 1)

        # generator loss
        lossG = -tf.reduce_mean(fake_wlogits)

        # discriminator loss
        lossD_d = -(tf.reduce_mean(real_wlogits_u) + tf.reduce_mean(real_wlogits_l)) / 2
        lossD_g = -lossG
        norm_gradD_u = tf.norm(tf.gradients(semifake_wlogits_u, semifake_u)[0], axis=1)
        norm_gradD_l = tf.norm(tf.gradients(semifake_wlogits_l, semifake_l)[0], axis=1)
        lossD_p = tf.reduce_mean(penalty_strength * tf.square(tf.concat((norm_gradD_u, norm_gradD_l), 0) - 1))
        lossD_ac = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_aclogits_l, labels=label_smooth))
        lossD = lossD_d + lossD_g + lossD_p + lossD_ac

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            # classifier performance
            pred = tf.argmax(semifake_acclass_l, 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            tf.summary.image('confusion_matrix', tf.reshape(tf.confusion_matrix(
                tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16), [1, self.n_y, self.n_y, 1]))
            # discriminator performance
            tf.summary.histogram('distance_u', real_wlogits_u - fake_wlogits)
            tf.summary.histogram('distance_l', real_wlogits_l - fake_wlogits)
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d', lossD_d)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD_g', lossD_p)
            tf.summary.scalar('lossD_ac', lossD_ac)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD


class MSSGAN (GANBase):

    name = 'MSSGAN'

    def __init__(self, n_y, additional_features=None, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None,
                 use_batch_norm_G=True, use_batch_norm_D=False, name=None,
                 seed=np.random.randint(int(1e8)), log_and_save=True, debug=False):

        # label variables
        self.n_y = n_y
        self.input_y = tf.placeholder(tf.float32, (None, self.n_y), 'label')
        self.input_x_l = tf.placeholder(tf.float32, (None, 32, 32, 1), 'labeled_image')

        # additional feature sets
        self.additional_feature_names = []
        self.additional_feature_dimensions = []
        self.input_x_additional = []
        self.input_x_l_additional = []
        if additional_features is not None:
            assert isinstance(additional_features, dict), 'additional_features must be of type dict or None'
            for key, val in additional_features.iteritems():
                assert isinstance(key, str), 'additional_features keys must be of type str'
                assert isinstance(val, int), 'additional_features keys must be of type int'
                self.additional_feature_names.append(key)
                self.additional_feature_dimensions.append(val)
                self.input_x_additional.append(tf.placeholder(tf.float32, (None, val), key))
                self.input_x_l_additional.append(tf.placeholder(tf.float32, (None, val), 'labeled_' + key))

        # init
        super(MSSGAN, self).__init__(n_extra_generator_layers=n_extra_generator_layers,
                                     n_extra_discriminator_layers=n_extra_discriminator_layers,
                                     mask=mask, use_batch_norm_G=use_batch_norm_G, use_batch_norm_D=use_batch_norm_D,
                                     name=name, log_and_save=log_and_save, seed=seed, debug=debug)

    def _build_additional_dense_generator(self, n_out, name, tensor=None, n_hidden_layers=4, n_hidden_nodes=128,
                                          training=False, batch_norm=None):
        if tensor is None:
            tensor = self.input_n
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator_' + name) as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, n_hidden_nodes)):
                tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                       use_bias=not batch_norm, name='dense')))
            # extra layers
            for it in range(n_hidden_layers-1):
                with tf.variable_scope('extra-{0}.{1}'.format(it, n_hidden_nodes)):
                    tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                           use_bias=not batch_norm, name='dense')))

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(n_hidden_nodes, n_out)):
                tensor = tf.layers.dense(tensor, n_out, kernel_initializer=init_normal(), name='dense')

            return tensor

    def _build_additional_dense_discriminator_base(self, tensor, name, n_hidden_layers=3, n_hidden_nodes=128,
                                                   n_out_nodes=128, training=False, batch_norm=None):
        if batch_norm is None:
            batch_norm = self.batch_norm_D
        if batch_norm:
            def bn(x):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        # initial layer
        with tf.variable_scope('initial.{0}-{1}'.format(tensor.shape[1], n_hidden_nodes)):
            tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                   use_bias=not batch_norm, name='dense')))
        # extra layers
        for it in range(n_hidden_layers-1):
            with tf.variable_scope('extra-{0}.{1}'.format(it, n_hidden_nodes)):
                tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                       use_bias=not batch_norm, name='dense')))

        with tf.variable_scope('base_final-{0}.{1}'.format(n_hidden_nodes, n_out_nodes)):
            tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_out_nodes, kernel_initializer=init_normal(),
                                                   use_bias=not batch_norm, name='dense')))

        return tensor

    def _build_multi_discriminator_base(self, tensor=None, training=False, batch_norm=None):
        base_out_size = 100

        if tensor is None:
            tensor = [self.input_x] + self.input_x_additional
        assert not isinstance(tensor, (list, tuple)) or len(self.additional_feature_names) == len(tensor) - 1, \
            'wrong number of input tensors'
        if batch_norm is None:
            batch_norm = self.batch_norm_D

        # discriminator base for image
        with tf.variable_scope('image'):
            if isinstance(tensor, (list, tuple)):
                tensor_pre = [self._build_discriminator_base(tensor[0], training, batch_norm=batch_norm)]
            else:
                tensor_pre = [self._build_discriminator_base(tensor, training, batch_norm=batch_norm)]

        # additional discriminator bases
        for it, name in enumerate(self.additional_feature_names):
            with tf.variable_scope(name):
                tensor_pre.append(self._build_additional_dense_discriminator_base(tensor[it + 1],
                                                                                  self.additional_feature_names[it],
                                                                                  n_out_nodes=base_out_size,
                                                                                  training=training,
                                                                                  batch_norm=batch_norm))

        # concatenate
        for it, name in enumerate(self.additional_feature_names):
            tensor_pre[it + 1] = tf.expand_dims(tf.expand_dims(tensor_pre[it + 1], 1), 1)
            tensor_pre[it + 1] = tf.tile(tensor_pre[it + 1], [1, int(tensor_pre[0].shape[1]), int(tensor_pre[0].shape[2]), 1])

        return tf.concat(tensor_pre, axis=3)

    def _build_discriminator(self, tensor=None, training=False, batch_norm=None):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_multi_discriminator_base(tensor, training, batch_norm)

            # last conv layer
            d_out = 1 + self.n_y
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

            return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):

        fake = [self._build_generator(training=training)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_dense_generator(self.additional_feature_dimensions[it],
                                                               self.additional_feature_names[it], training=training))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1)))

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake_image', fake[0], max_outputs=1)
            tf.summary.image('real_image', self.input_x, max_outputs=1)
            # generated/real psds
            try:
                tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
                tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
            except IndexError:
                pass
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                              [1, self.n_y, self.n_y, 1])
            tf.summary.image('confusion_matrix', cmat)
            tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.1))
            # discriminator performance
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD

    def train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u[0].shape[0]
        n_train_l = trainx_l[0].shape[0]
        n_test = testx[0].shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
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

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            if self.debug:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            while True:
                try:
                    for epoch in range(n_epochs):
                        # train on epoch
                        start = time()
                        step = 0
                        n, lg, ld = 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                            # prep
                            n += n_batch_actual
                            randind = np.random.choice(n_train_l, n_batch_actual)
                            feed_dict = {self.input_x: trainx_u[0][batch_index:batch_index + n_batch_actual],
                                         self.input_x_l: trainx_l[0][randind],
                                         self.input_y: trainy[randind],
                                         self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                            for it in range(len(self.additional_feature_names)):
                                feed_dict[self.input_x_additional[it]] = trainx_u[it + 1][batch_index:batch_index + n_batch_actual]
                                feed_dict[self.input_x_l_additional[it]] = trainx_l[it + 1][randind]
                            # discriminator
                            temp = sess.run(adamD, feed_dict)
                            ld += temp * n_batch_actual
                            # generator
                            feed_dict[self.input_n] = np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)
                            temp, summary, step = sess.run([adamG, merged_summary, global_step], feed_dict)
                            lg += temp * n_batch_actual
                            if np.isnan([lg, ld]).any() or np.isinf([lg, ld]).any():
                                raise NanInfException
                            if n % (2 * n_batch) == 0:
                                self._log(summary, step)
                               # print 'epoch {:d}/{:d} (part {:d}/{:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                                 #   .format(epoch + 1, n_epochs, n, n_train_u, (lg + ld) / n, lg / n, ld / n, int(time() - start))
                            if n % (100 * n_batch) == 0:
                                m, lge, lde = 0, 0, 0
                                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                                    m += n_batch_actual
                                    feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                                 self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                                    for it in range(len(self.additional_feature_names)):
                                        feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                        feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                    out = sess.run([evalG, evalD, merged_summary], feed_dict)
                                    lge += out[0] * n_batch_actual
                                    lde += out[1] * n_batch_actual
                                self._log(out[2], step, test=True)
                                #print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                                 #   .format(epoch + 1, n_epochs, (lge + lde) / m, lge / m, lde / m, int(time() - start))
                                # save
                                self._save(sess, step)

                        # save after each epoch
                        self._save(sess, step)

                        # evaluate
                        n, lge, lde = 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_test, n_batch):
                            n += n_batch_actual
                            feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                         self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                                         self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                         self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                            for it in range(len(self.additional_feature_names)):
                                feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                            out = sess.run([evalG, evalD, merged_summary], feed_dict)
                            lge += out[0] * n_batch_actual
                            lde += out[1] * n_batch_actual
                        self._log(out[2], step, test=True)
                        #print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                       #     .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, int(time() - start))
                    break

                except NanInfException:
                    if epoch >= 10:
                        a=1
                    #print 'Got NaNs or infs. Resetting parameters and starting again.'
                    try:
                        self._restore(sess)
                    except:
                        step = sess.run(global_step)
                        sess.run(tf.global_variables_initializer())
                        tf.assign(global_step, step)
                    trainx_u = shuffle_data(trainx_u)
                    trainx_l, trainy = shuffle_data(trainx_l, trainy)
                    testx, testy = shuffle_data(testx, testy)


class SymMSSGAN (MSSGAN):

    name = 'SymMSSGAN'

    def _build_loss(self, label_strength=1., training=False):

        fake = [self._build_generator(training=training)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_dense_generator(self.additional_feature_dimensions[it],
                                                               self.additional_feature_names[it], training=training))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        fake[0] *= -1
        _, fake_flipped_logits = self._build_discriminator(fake, training=training)
        _, real_flipped_logits_u = self._build_discriminator([-self.input_x] + self.input_x_additional, training=training)
        _, real_flipped_logits_l = self._build_discriminator([-self.input_x_l] + self.input_x_l_additional,
                                                             training=training)

        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1)))

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD_sym = tf.sqrt(tf.reduce_mean(tf.squared_difference(fake_logits, fake_flipped_logits))
                            + tf.reduce_mean(tf.squared_difference(real_logits_u, real_flipped_logits_u))
                            + tf.reduce_mean(tf.squared_difference(real_logits_l, real_flipped_logits_l)))
        lossD = lossD_d_l + lossD_d_u + lossD_g + lossD_sym

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake_image', fake[0], max_outputs=1)
            tf.summary.image('real_image', self.input_x, max_outputs=1)
            # generated/real images
            tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
            tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                              [1, self.n_y, self.n_y, 1])
            tf.summary.image('confusion_matrix', cmat)
            tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.1))
            # discriminator performance
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD_sym', lossD_sym)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD


class MVEEGAN (MSSGAN):

    name = 'MVEEGAN'

    # def _build_additional_dense_encoder_base(self, tensor, name, n_hidden_layers=4, n_hidden_nodes=128,
    #                                                training=False, batch_norm=None):
    #     if batch_norm is None:
    #         batch_norm = self.batch_norm_D
    #     if batch_norm:
    #         def bn(x):
    #             return tf.contrib.layers.batch_norm(x, is_training=training,
    #                                                 renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
    #     else:
    #         bn = tf.identity
    #
    #     with tf.variable_scope('discriminator_' + name) as scope:
    #         # set reuse if necessary
    #         if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
    #             scope.reuse_variables()
    #
    #         # initial layer
    #         with tf.variable_scope('initial.{0}-{1}'.format(tf.shape(tensor)[0], n_hidden_nodes)):
    #             tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
    #                                                    name='dense')))
    #         # extra layers
    #         for it in range(n_hidden_layers-1):
    #             with tf.variable_scope('extra-{0}.{1}'.format(it, n_hidden_nodes)):
    #                 tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
    #                                                        name='dense')))
    #
    #         return tensor

    def _build_encoder(self, tensor=None, training=False, batch_norm=None):

        with tf.variable_scope('encoder') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # encoder base
            tensor = self._build_multi_discriminator_base(tensor, training, batch_norm)

            # last conv layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], self.n_noise)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, self.n_noise, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, self.n_noise])

            # with tf.variable_scope('image'):
            #     if isinstance(tensor, (list, tuple)):
            #         tensor_pre = self._build_discriminator_base(tensor[0], training)
            #     else:
            #         tensor_pre = self._build_discriminator_base(tensor, training)
            #     tensor_pre = [tf.contrib.layers.flatten(tensor_pre)]
            #
            # # additional encoder bases
            # for it, name in enumerate(self.additional_feature_names):
            #     with tf.variable_scope(name):
            #         tensor_pre.append(self._build_additional_dense_generator(self.additional_feature_dimensions[it],
            #                                                                  self.additional_feature_names[it],
            #                                                                  tensor[it + 1]))
            #
            # # flatten
            # tensor = tf.concat(tensor_pre, axis=1)
            #
            # # final layer
            # with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], self.n_noise)):
            #     out_logits = tf.layers.dense(tensor, self.n_noise, kernel_initializer=init_normal(), name='dense')

        return out_logits

    def _build_discriminator(self, tensor=None, encoding_tensor=None, training=False, batch_norm=None):

        if encoding_tensor is None:
            encoding_tensor = self._build_encoder(training=training)

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_multi_discriminator_base(tensor, training, batch_norm)

            # concatenate encoding tensor
            encoding_tensor = tf.expand_dims(tf.expand_dims(encoding_tensor, 1), 1)
            encoding_tensor = tf.tile(encoding_tensor, [1, int(tensor.shape[1]), int(tensor.shape[2]), 1])
            tensor = tf.concat((tensor, encoding_tensor), axis=3)

            # last conv layer
            d_out = 1 + self.n_y
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

            return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):

        fake = [self._build_generator(training=training)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_dense_generator(self.additional_feature_dimensions[it],
                                                               self.additional_feature_names[it], training=training))
        fake_encoding = self._build_encoder(fake, training=training)
        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # encoder loss
        lossF = tf.reduce_mean(tf.squared_difference(self.input_n, fake_encoding))

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1))) + lossF

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        # lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_d_u = tf.reduce_mean(-label_strength * safe_log(1 - real_label_u[:, -1])
                                   - (1 - label_strength) * safe_log(real_label_u[:, -1]))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake_image', fake[0], max_outputs=1)
            tf.summary.image('real_image', self.input_x, max_outputs=1)
            # generated/real psds
            tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
            tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                              [1, self.n_y, self.n_y, 1])
            tf.summary.image('confusion_matrix', cmat)
            tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.1))
            # discriminator performance
            # tf.summary.histogram('D_fake', fake_label[:, -1])
            # tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('lossF', lossF)
            tf.summary.scalar('loss', lossG + lossD + lossF)

        return lossG, lossD, lossF

    def train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u[0].shape[0]
        n_train_l = trainx_l[0].shape[0]
        n_test = testx[0].shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD, lossF = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD, evalF = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        tvarsF = [var for var in tf.trainable_variables() if 'encoder' in var.name]
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
            adamF = tf.contrib.layers.optimize_loss(loss=lossF,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optF',
                                                    variables=tvarsF)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:

            # initialize variables
            init = tf.global_variables_initializer()
            sess.run(init)

            if self.debug:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            while True:
                try:
                    for epoch in range(n_epochs):
                        # train on epoch
                        start = time()
                        step = 0
                        n, lg, ld, lf = 0, 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                            # prep
                            n += n_batch_actual
                            randind = np.random.choice(n_train_l, n_batch_actual)
                            feed_dict = {self.input_x: trainx_u[0][batch_index:batch_index + n_batch_actual],
                                         self.input_x_l: trainx_l[0][randind],
                                         self.input_y: trainy[randind],
                                         self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                            for it in range(len(self.additional_feature_names)):
                                feed_dict[self.input_x_additional[it]] = trainx_u[it + 1][batch_index:batch_index + n_batch_actual]
                                feed_dict[self.input_x_l_additional[it]] = trainx_l[it + 1][randind]
                            # update
                            tempg, tempd, tempf, summary, step = \
                                sess.run([adamG, adamD, adamF, merged_summary, global_step], feed_dict)
                            lg += tempg * n_batch_actual
                            ld += tempd * n_batch_actual
                            lf += tempf * n_batch_actual
                            # check
                            if np.isnan([lg, ld, lf]).any() or np.isinf([lg, ld, lf]).any():
                                raise NanInfException
                            if n % (2 * n_batch) == 0:
                                self._log(summary, step)
                               # print 'epoch {:d}/{:d} (part {:d}/{:d}):  training loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                               #     .format(epoch + 1, n_epochs, n, n_train_u, (lg + ld) / n, lg / n, ld / n, lf / n, int(time() - start))
                            if n % (100 * n_batch) == 0:
                                # evaluate
                                m, lge, lde, lfe = 0, 0, 0, 0
                                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                                    m += n_batch_actual
                                    feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                                 self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                                    for it in range(len(self.additional_feature_names)):
                                        feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                        feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                    out = sess.run([evalG, evalD, evalF, merged_summary], feed_dict)
                                    lge += out[0] * n_batch_actual
                                    lde += out[1] * n_batch_actual
                                    lfe += out[2] * n_batch_actual
                                self._log(out[3], step, test=True)
                               # print 'epoch {:d}/{:d} (part {:d}/{:d}):  evaluation loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                                #    .format(epoch + 1, n_epochs, m, n_train_u, (lge + lde) / m, lge / m, lde / m, lfe / m, int(time() - start))
                                # save
                                self._save(sess, step)

                        # save after each epoch
                        self._save(sess, step)

                        # evaluate
                        n, lge, lde, lfe = 0, 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_test, n_batch):
                            n += n_batch_actual
                            feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                         self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                                         self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                         self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                            for it in range(len(self.additional_feature_names)):
                                feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                            out = sess.run([evalG, evalD, evalF, merged_summary], feed_dict)
                            lge += out[0] * n_batch_actual
                            lde += out[1] * n_batch_actual
                            lfe += out[2] * n_batch_actual
                        self._log(out[3], step, test=True)
                      #  print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                       #     .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, lfe / n, int(time() - start))
                    break

                except NanInfException:
                    if epoch >= 10:
                        a=1
                   # print 'Got NaNs or infs. Resetting parameters and starting again.'
                    try:
                        self._restore(sess)
                    except:
                        step = sess.run(global_step)
                        sess.run(init)
                        tf.assign(global_step, step)
                    trainx_u = shuffle_data(trainx_u)
                    trainx_l, trainy = shuffle_data(trainx_l, trainy)
                    testx, testy = shuffle_data(testx, testy)

    def eval(self, input_):
        raise NotImplementedError


    def inference_2_matfile(self, graph=None):
        from scipy.io import savemat

        # get graph
        if graph is None:
            graph = tf.get_default_graph()

        # extract parameters
        # {x.name: x.eval() for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
        #  (x.name.startswith('encoder/') or x.name.startswith('discriminator/'))}
        params = dict()
        for op in graph.get_operations():
            name = op.name
            if (name.startswith('encoder/') or name.startswith('discriminator/')) \
                    and (name.endswith('kernel') or name.endswith('bias')):
                try:
                    name = name.replace('/', '__').replace('-', '_').replace('.', '_')
                    params[name] = graph.get_tensor_by_name(op.name + ':0').eval()
                except:
                    pass

        # save
        savemat(join(self.path, self.name, 'MVEEGAN_inference'), params, long_field_names=True)


if __name__ == '__main__':
    from data_loaders import mnist
    from data_loaders.icldata import ICLabelDataset

    debug = False
    seed = 0

    icl = ICLabelDataset(features=('all'), transform='none', do_pca=False, unique=True, combine_output=False,
                         expert_only='luca', testing=False, seed=seed)
    icl_data = icl.load_semi_supervised()
    topo_data = list()
    for it in range(4):
        temp = 0.99 * icl_data[it][0]['topo'] / np.abs(icl_data[it][0]['topo']).max(1, keepdims=True)
        topo_data.append(icl.pad_topo(temp).astype(np.float32).reshape(-1, 32, 32, 1))

    input_data = [[topo_data[x],
                   0.99 * icl_data[x][0]['psd'],
                   0.99 * icl_data[x][0]['autocorr'],
                   # 0.99 * icl_data[x][0]['psd_var'],
                   # 0.99 * icl_data[x][0]['psd_kurt']
                   ] for x in range(4)]

    for it in range(1, 4):
        input_data[it][0] = np.concatenate((topo_data[it], -topo_data[it], np.flip(topo_data[it], 2), -np.flip(topo_data[it], 2)))
        for it2 in range(1, len(input_data[0])):
            input_data[it][it2] = np.tile(input_data[it][it2], (4, 1))
    train_labels, test_labels = np.tile(icl_data[1][1][0], (4, 1)), np.tile(icl_data[3][1][0], (4, 1))

    # symmssgan = SymMSSGAN(icl_data[1][1][0].shape[1], additional_features=OrderedDict(psd_med=input_data[1][1].shape[1],
    #                                                                             # psd_var=input_data[1][2].shape[1],
    #                                                                             # psd_kurt=input_data[1][3].shape[1]
    #                                                                             ),
    #                 mask=np.setdiff1d(np.arange(1024), icl.topo_ind),
    #                 name='icltopo', debug=debug)
    # symmssgan.train(input_data[0], input_data[1], train_labels, input_data[3], test_labels,
    #              learning_rate=1e-5, label_strength=0.9)
    # a = 1

    # mssgan = MSSGAN(icl_data[1][1][0].shape[1], additional_features=OrderedDict(psd_med=input_data[1][1].shape[1],
    #                                                                             # psd_var=input_data[1][2].shape[1],
    #                                                                             # psd_kurt=input_data[1][3].shape[1]
    #                                                                             ),
    #                 mask=np.setdiff1d(np.arange(1024), icl.topo_ind),
    #                 name='icltopopsd', debug=debug)
    # mssgan.train(input_data[0], input_data[1], train_labels, input_data[3], test_labels,
    #              learning_rate=1e-5, label_strength=0.9)

    mveegan = MVEEGAN(icl_data[1][1][0].shape[1], additional_features=OrderedDict(psd_med=input_data[1][1].shape[1],
                                                                                  autocorr=input_data[1][2].shape[1],
                                                                                  ),
                      mask=np.setdiff1d(np.arange(1024), icl.topo_ind),
                      name='icl_bnG_smoothu_topo_psd_autocorr', debug=debug)
    mveegan.train(input_data[0], input_data[1], train_labels, input_data[3], test_labels,
                  learning_rate=1e-5, label_strength=0.9)
    a = 1

    # # load SSL data
    # seed = np.random.randint(1, 2147462579)
    # mnist_data = mnist.load_semi_supervised(filter_std=0.0, train_valid_combine=True)
    # mnist_data = [[x[0].astype(np.float32), x[1].astype(np.float32)] for x in mnist_data]
    #
    # mask = np.zeros((32, 32), dtype=bool)
    # mask[2:30, 2:30] = True
    # mask_ind = np.where(mask.flatten())[0]
    # for it, (x, y) in enumerate(mnist_data):
    #     new = np.zeros((x.shape[0], 32 * 32), dtype=np.float32)
    #     new[:, mask_ind] = 0.99 * (x * 2 - 1)
    #     mnist_data[it][0] = new.reshape(-1, 32, 32, 1)

    # veegan = VEEGAN(mnist_data[1][1].shape[1], mask=np.setdiff1d(np.arange(1024), mask_ind), name='mnist', debug=debug)
    # # veegan.load()
    # veegan.train(mnist_data[0][0], mnist_data[1][0], mnist_data[1][1], mnist_data[2][0], mnist_data[2][1],
    #              n_batch=100, label_strength=0.9)
    # a = 1
    #
    # ssgan = SSGAN(mnist_data[1][1].shape[1], mask=np.setdiff1d(np.arange(1024), mask_ind), debug=debug)
    # ssgan.train(mnist_data[0][0], mnist_data[1][0], mnist_data[1][1], mnist_data[2][0], mnist_data[2][1],
    #             n_batch=100, label_strength=0.9)
    # a = 1

    # ssgan = SSGAN(icl_data[1][1][0].shape[1], mask=np.setdiff1d(np.arange(1024), icl.topo_ind),
    #               name='icltopo', debug=debug)
    # ssgan.train(topo_data[0], topo_data[1], train_labels, topo_data[2], test_labels,
    #             label_strength=0.9) # learning_rate=1e-5,
    # a = 1

    # wgan = WGAN(mask=np.setdiff1d(np.arange(1024), mask_ind), debug=debug)
    # wgan.train(mnist_data[0][0], mnist_data[1][0], n_batch=100, label_strength=0.9)
    # a = 1

