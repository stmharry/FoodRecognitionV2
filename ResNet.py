from __future__ import print_function

import enum
import numpy as np
import os
import scipy.io
import subprocess
import sys
import tensorflow as tf
import time

ROOT_PATH = os.path.dirname(__file__)
DEEPBOX_PATH = os.path.join(ROOT_PATH, 'DeepBox')
if DEEPBOX_PATH not in sys.path:
    sys.path.append(DEEPBOX_PATH)

from deepbox import util
from deepbox.model import Model

IS_DEBUG = False
META = None


def DEBUG(value, name=None, func=None):
    if not IS_DEBUG:
        return value

    if name is None:
        name = value.name
    show = value
    if func is not None:
        show = func(show)
        name = '%s(%s)' % (func.__name__, name)
    return tf.Print(value, [show], '%s: ' % name)


def set_meta(meta):
    global META
    META = meta


def prob_list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x


class Meta(object):
    WORKING_DIR = '/tmp/' + time.strftime('%Y%-m-%d-%H%M%S')
    CLASSNAMES_FILENAME = 'class_names.txt'
    CLASS_NAMES = list()

    @staticmethod
    def train(image_dir, working_dir=WORKING_DIR, classnames_filename=CLASSNAMES_FILENAME):
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir)

        class_names = list()
        for class_name in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, class_name)
            if not class_name.startswith('.') and os.path.isdir(class_dir):
                class_names.append(class_name)
        np.savetxt(os.path.join(working_dir, classnames_filename), class_names, delimiter=',', fmt='%s')

        return Meta(working_dir=working_dir, class_names=class_names)

    @staticmethod
    def test(working_dir=WORKING_DIR, classnames_filename=CLASSNAMES_FILENAME):
        classnames_path = os.path.join(working_dir, classnames_filename)
        if os.path.isfile(classnames_path):
            class_names = np.loadtxt(classnames_path, dtype=np.str, delimiter=',')

        return Meta(working_dir=working_dir, class_names=class_names)

    def __init__(self, working_dir=WORKING_DIR, class_names=CLASS_NAMES):
        self.working_dir = working_dir
        self.class_names = class_names


class ImageUtil(object):
    @staticmethod
    def get_shape(value):
        return tuple(value.get_shape().as_list())

    @staticmethod
    def get_size(value):
        return ImageUtil.get_shape(value)[1:3]

    @staticmethod
    def get_channel(value):
        return ImageUtil.get_shape(value)[3]

    @staticmethod
    def random(lower, upper, dtype=tf.float32):
        return tf.random_uniform((), lower, upper, dtype=dtype)

    @staticmethod
    def random_resize(value, size_range, max_log_aspect_ratio):
        aspect_ratio = tf.exp(ImageUtil.random(-max_log_aspect_ratio, +max_log_aspect_ratio))
        new_shorter_size = ImageUtil.random(size_range[0], size_range[1])

        new_height_and_width = tf.cond(
            tf.less(aspect_ratio, 1.0),
            lambda: (new_shorter_size / aspect_ratio, new_shorter_size),
            lambda: (new_shorter_size, new_shorter_size * aspect_ratio),
        )

        value = tf.expand_dims(value, 0)
        value = tf.image.resize_bilinear(value, tf.to_int32(tf.pack(new_height_and_width)))
        value = tf.squeeze(value, [0])
        return value

    @staticmethod
    def random_crop(value, size):
        shape = tf.shape(value)
        height = shape[0]
        width = shape[1]

        offset_height = ImageUtil.random(0, height - size + 1, dtype=tf.int32)
        offset_width = ImageUtil.random(0, width - size + 1, dtype=tf.int32)

        value = tf.slice(
            value,
            tf.pack((offset_height, offset_width, 0)),
            tf.pack((size, size, -1)))
        value.set_shape((size, size, 3))
        return value

    @staticmethod
    def random_flip(value):
        value = tf.image.random_flip_left_right(value)
        return value

    @staticmethod
    def random_adjust_rgb(value, max_delta=63, contrast_range=(0.5, 1.5)):
        value = tf.image.random_brightness(value, max_delta=max_delta)
        value = tf.image.random_contrast(value, lower=contrast_range[0], upper=contrast_range[1])
        return value


class Blob(object):
    class Content(enum.Enum):
        IMAGE_LABEL = 0
        VALUE = 1

    def __init__(self, **kwargs):
        assert ('images' in kwargs) + ('values' in kwargs) == 1, 'Too many arguments!'

        if 'images' in kwargs:
            images = prob_list(kwargs['images'])
            if 'labels' in kwargs:
                labels = prob_list(kwargs['labels'])
            else:
                labels = [tf.constant(-1, dtype=tf.int64) for _ in xrange(len(images))]

            self.content = Blob.Content.IMAGE_LABEL
            self.images = images
            self.labels = labels
        elif 'values' in kwargs:
            values = prob_list(kwargs['values'])
            self.content = Blob.Content.VALUE
            self.values = values

    def as_tuple_list(self):
        return zip(self.images, self.labels)

    def func(self, f):
        return f(self)

    def kwargs(self):
        if self.content == Blob.Content.IMAGE_LABEL:
            values = self.images + self.labels
        elif self.content == Blob.Content.VALUE:
            values = self.values

        return dict(
            feed_dict=dict(),
            fetch={value.name: value for value in values})


class BaseProducer(object):
    def get_queue_enqueue(self, values, dtype=tf.float32, shape=None, auto=False):
        queue = tf.FIFOQueue(self.capacity, dtypes=[dtype], shapes=None if shape is None else [shape])
        enqueue = queue.enqueue_many([values])
        if auto:
            queue_runner = tf.train.QueueRunner(queue, [enqueue])
            tf.train.add_queue_runner(queue_runner)
        return (queue, enqueue)


class SimpleProducer(BaseProducer):
    def blob(self, name='image', shape=None, dtype=tf.float32):
        self.placeholder = tf.placeholder(
            name=name,
            shape=shape,
            dtype=dtype)
        return Blob(images=self.placeholder)

    def get_kwargs(self, image):
        return dict(
            feed_dict={self.placeholder: image},
            fetch=dict())


class QueueProducer(BaseProducer):
    CAPACITY = 1024

    def __init__(self, capacity=CAPACITY):
        self.capacity = capacity

    def blob(self, name='image', shape=None, dtype=tf.float32):
        self.placeholder = tf.placeholder(
            name=name,
            shape=shape,
            dtype=dtype)
        (self.queue, self.enqueue) = self.get_queue_enqueue(values=[self.placeholder], dtype=dtype, shape=shape, auto=False)
        image = self.queue.dequeue()
        return Blob(images=image)

    def kwargs(self, image):
        return dict(
            feed_dict={self.placeholder: image},
            fetch=dict(queue_producer_enqueue=self.enqueue))


class FileProducer(BaseProducer):
    CAPACITY = 32
    NUM_TRAIN_INPUTS = 8
    NUM_TEST_INPUTS = 1
    SUBSAMPLE_SIZE = 64

    def __init__(self,
                 capacity=CAPACITY,
                 num_train_inputs=NUM_TRAIN_INPUTS,
                 num_test_inputs=NUM_TEST_INPUTS,
                 subsample_size=SUBSAMPLE_SIZE):

        self.capacity = capacity
        self.num_train_inputs = num_train_inputs
        self.num_test_inputs = num_test_inputs
        self.subsample_size = subsample_size

    def _blob(self,
              image_dir,
              num_inputs=1,
              subsample_divisible=True,
              check=False,
              shuffle=False):

        filename_list = list()
        classname_list = list()

        for class_name in META.class_names:
            class_dir = os.path.join(image_dir, class_name)
            for (file_dir, _, file_names) in os.walk(class_dir):
                for file_name in file_names:
                    if not file_name.endswith('.jpg'):
                        continue
                    if (hash(file_name) % self.subsample_size == 0) != subsample_divisible:
                        continue
                    filename_list.append(os.path.join(file_dir, file_name))
                    classname_list.append(class_name)

        label_list = map(META.class_names.index, classname_list)

        if check:
            num_file_list = list()
            for (num_file, filename) in enumerate(filename_list):
                print('\033[2K\rChecking image %d / %d' % (num_file + 1, len(filename_list)), end='')
                sp = subprocess.Popen(['identify', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (stdout, stderr) = sp.communicate()
                if stderr:
                    os.remove(filename)
                    print('\nRemove %s' % filename)
                else:
                    num_file_list.append(num_file)
                sys.stdout.flush()
            print('')

            filename_list = map(filename_list.__getitem__, num_file_list)
            label_list = map(label_list.__getitem__, num_file_list)

        images = list()
        labels = list()
        for num_input in xrange(num_inputs):
            if shuffle:
                perm = np.random.permutation(len(filename_list))
                filename_list = map(filename_list.__getitem__, perm)
                label_list = map(label_list.__getitem__, perm)

            filename_queue = self.get_queue_enqueue(filename_list, dtype=tf.string, shape=(), auto=True)[0]
            (key, value) = tf.WholeFileReader().read(filename_queue)
            image = tf.to_float(tf.image.decode_jpeg(value))

            label_queue = self.get_queue_enqueue(label_list, dtype=tf.int64, shape=(), auto=True)[0]
            label = label_queue.dequeue()

            images.append(image)
            labels.append(label)

        return Blob(images=images, labels=labels)

    def trainBlob(self, image_dir, check=True):
        return self._blob(
            image_dir,
            num_inputs=self.num_train_inputs,
            subsample_divisible=False,
            check=check,
            shuffle=True)

    def testBlob(self, image_dir, check=False):
        return self._blob(
            image_dir,
            num_inputs=self.num_test_inputs,
            subsample_divisible=True,
            check=check,
            shuffle=False)

    def kwargs(self):
        return dict()


class Preprocess(object):
    NUM_TEST_CROPS = 4
    TRAIN_SIZE_RANGE = (224, 320)
    TEST_SIZE_RANGE = (256, 256)
    MAX_LOG_ASPECT_RATIO = 0.75

    NET_SIZE = 224
    NET_CHANNEL = 3
    MEAN_PATH = os.path.join(ROOT_PATH, 'archive/ResNet-mean.mat')

    def __init__(self,
                 num_test_crops=NUM_TEST_CROPS,
                 train_size_range=TRAIN_SIZE_RANGE,
                 test_size_range=TEST_SIZE_RANGE,
                 max_log_aspect_ratio=MAX_LOG_ASPECT_RATIO,
                 net_size=NET_SIZE,
                 net_channel=NET_CHANNEL,
                 mean_path=MEAN_PATH):

        self.num_test_crops = num_test_crops
        self.train_size_range = train_size_range
        self.test_size_range = test_size_range
        self.max_log_aspect_ratio = max_log_aspect_ratio

        self.net_size = net_size
        self.net_channel = net_channel
        self.shape = (net_size, net_size, net_channel)

        self.mean_path = mean_path
        self.mean = scipy.io.loadmat(mean_path)['mean']

    def _train(self, image):
        image = ImageUtil.random_resize(image, size_range=self.train_size_range, max_log_aspect_ratio=self.max_log_aspect_ratio)
        image = ImageUtil.random_crop(image, size=self.net_size)
        image = ImageUtil.random_flip(image)
        image = ImageUtil.random_adjust_rgb(image)
        image = image - self.mean
        image.set_shape(self.shape)

        return image

    def train(self, blob):
        return Blob(images=map(self._train, blob.images), labels=blob.labels)

    def _test_map(self, image):
        image = ImageUtil.random_resize(image, size_range=self.test_size_range, max_log_aspect_ratio=0.0)
        image = ImageUtil.random_crop(image, size=self.net_size)
        image = ImageUtil.random_flip(image)
        image = image - self.mean

        return image

    def _test(self, image):
        image = tf.tile(tf.expand_dims(image, dim=0), multiples=(self.num_test_crops, 1, 1, 1))
        image = tf.map_fn(self._test_map, image)
        image.set_shape((self.num_test_crops,) + self.shape)

        return image

    def test(self, blob):
        return Blob(images=map(self._test, blob.images), labels=blob.labels)


class Batch(object):
    BATCH_SIZE = 64
    NUM_TEST_CROPS = 4
    TRAIN_CAPACITY = 4096 + 1024
    TEST_CAPACITY = 64
    MIN_AFTER_DEQUEUE = 4096

    def __init__(self,
                 batch_size=BATCH_SIZE,
                 num_test_crops=NUM_TEST_CROPS,
                 train_capacity=TRAIN_CAPACITY,
                 test_capacity=TEST_CAPACITY,
                 min_after_dequeue=MIN_AFTER_DEQUEUE):

        self.batch_size = batch_size
        self.num_test_crops = num_test_crops
        self.train_capacity = train_capacity
        self.test_capacity = test_capacity
        self.min_after_dequeue = min_after_dequeue

    def make_size(self, batch_size):
        batch_size = tf.constant(batch_size, dtype=tf.int32)
        zero = tf.constant(0, dtype=tf.int32)

        total_size = tf.Variable(-1, trainable=False, dtype=tf.int32)
        (batch_size_, dec_batch_size) = tf.cond(
            tf.equal(total_size, -1),
            lambda: (batch_size, zero),
            lambda: (tf.minimum(total_size, batch_size),) * 2)
        batch_size_ = DEBUG(batch_size_, 'Batch.batch_size_')

        next_total_size = total_size - dec_batch_size
        next_total_size = DEBUG(next_total_size, 'Batch.next_total_size')

        total_size_ = tf.placeholder_with_default(next_total_size, shape=())
        assign = total_size.assign(total_size_)
        assign = DEBUG(assign, 'Batch.assign')

        return (batch_size_, total_size_, assign)

    def train(self, blob):
        (self.train_batch_size, self.train_total_size, self.train_assign) = self.make_size(self.batch_size)

        (image, label) = tf.tuple(
            tf.train.shuffle_batch_join(
                blob.as_tuple_list(),
                batch_size=self.train_batch_size,
                capacity=self.train_capacity,
                min_after_dequeue=self.min_after_dequeue),
            control_inputs=[self.train_assign])
        return Blob(images=image, labels=label)

    def test(self, blob):
        (self.test_batch_size, self.test_total_size, self.test_assign) = self.make_size(self.batch_size / self.num_test_crops)

        (image, label) = tf.tuple(
            tf.train.batch_join(
                blob.as_tuple_list(),
                batch_size=self.test_batch_size,
                capacity=self.test_capacity),
            control_inputs=[self.test_assign])

        shape = ImageUtil.get_shape(image)
        image = tf.reshape(image, (-1,) + shape[2:])

        return Blob(images=image, labels=label)

    def kwargs(self, total_size, phase):
        if phase == Net.Phase.TRAIN:
            return dict(
                feed_dict={self.train_total_size: total_size},
                fetch=dict(batch_train_assign=self.train_assign))
        elif phase == Net.Phase.TEST:
            return dict(
                feed_dict={self.test_total_size: total_size},
                fetch=dict(batch_test_assign=self.test_assign))


class Net(object):
    class Phase(enum.Enum):
        NONE = 0
        TRAIN = 1
        TEST = 2

    NET_VARIABLES = 'net_variables'
    NET_COLLECTIONS = [tf.GraphKeys.VARIABLES, NET_VARIABLES]
    MODEL_FILENAME = 'model'

    LEARNING_RATE = 1e-1
    LEARNING_RATE_MODES = dict(normal=1.0, slow=0.0)
    LEARNING_RATE_DECAY_STEPS = 0
    LEARNING_RATE_DECAY_RATE = 1.0
    WEIGHT_DECAY = 0.0

    GPU_FRAC = 1.0

    @staticmethod
    def placeholder(name=None, shape=(), dtype=tf.float32, default=None):
        if default is None:
            return tf.placeholder(
                name=name,
                shape=shape,
                dtype=dtype)
        else:
            return tf.placeholder_with_default(
                input=default,
                name=name,
                shape=shape)

    @staticmethod
    def get_const_variable(value, name, shape=(), dtype=tf.float32, trainable=False, collections=None):
        return tf.get_variable(
            name,
            shape=shape,
            dtype=dtype,
            initializer=tf.constant_initializer(value),
            trainable=trainable,
            collections=collections)

    @staticmethod
    def get_assignable_variable(value, name, shape=(), dtype=tf.float32, collections=None):
        placeholder = Net.placeholder(name, shape=shape, dtype=dtype)
        var = Net.get_const_variable(value, name, shape=shape, dtype=dtype, trainable=False, collections=collections)
        assign = var.assign(placeholder)
        return (placeholder, var, assign)

    @staticmethod
    def expand(size):
        return (1,) + size + (1,)

    @staticmethod
    def avg_pool(value, name, size, stride=None, padding='SAME'):
        with tf.variable_scope(name):
            if stride is None:
                stride = size
            value = tf.nn.avg_pool(value, ksize=Net.expand(size), strides=Net.expand(stride), padding=padding, name='avg_pool')
        return value

    @staticmethod
    def max_pool(value, name, size, stride=None, padding='SAME'):
        with tf.variable_scope(name):
            if stride is None:
                stride = size
            value = tf.nn.max_pool(value, ksize=Net.expand(size), strides=Net.expand(stride), padding=padding, name='max_pool')
        return value

    def __init__(self,
                 learning_rate=LEARNING_RATE,
                 learning_modes=LEARNING_RATE_MODES,
                 learning_rate_decay_steps=LEARNING_RATE_DECAY_STEPS,
                 learning_rate_decay_rate=LEARNING_RATE_DECAY_RATE,
                 weight_decay=WEIGHT_DECAY,
                 gpu_frac=GPU_FRAC,
                 is_train=False,
                 is_show=False):
        assert len(META.class_names), 'Only create net when META.class_names is not empty!'

        self.learning_rate = Net.get_const_variable(learning_rate, 'learning_rate')
        self.learning_modes = learning_modes
        self.weight_decay = weight_decay
        self.gpu_frac = gpu_frac
        self.is_train = is_train
        self.is_show = is_show

        (self.phase, self.phase_, self.phase_assign) = Net.get_assignable_variable(Net.Phase.NONE.value, 'phase', dtype=tf.int32)
        self.class_names = Net.get_const_variable(META.class_names, 'class_names', shape=(len(META.class_names),), dtype=tf.string, collections=Net.NET_COLLECTIONS)
        self.global_step = Net.get_const_variable(0, 'global_step')
        self.model_path = os.path.join(META.working_dir, Net.MODEL_FILENAME)

        if (learning_rate_decay_steps > 0) and (learning_rate_decay_rate < 1.0):
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=learning_rate_decay_steps,
                decay_rate=learning_rate_decay_rate,
                staircase=True)

    def case(self, phase_fn_pairs, shapes=None):
        pred_fn_pairs = [(tf.equal(self.phase_, phase_.value), fn) for (phase_, fn) in phase_fn_pairs]
        values = tf.case(pred_fn_pairs, default=pred_fn_pairs[0][1])
        if shapes is not None:
            for (value, shape) in zip(values, shapes):
                value.set_shape(shape)
        return values

    def make_stat(self):
        assert hasattr(self, 'prob'), 'net has no attribute "prob"!'

        self.target = tf.one_hot(self.label, len(META.class_names))
        self.target_frac = tf.reduce_mean(self.target, 0)
        self.loss = - tf.reduce_mean(self.target * tf.log(self.prob + util.EPSILON)) * len(META.class_names)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            self.loss += tf.add_n(regularization_losses)

        self.pred = tf.argmax(self.prob, 1)
        self.correct = tf.to_float(tf.equal(self.label, self.pred))
        self.correct_frac = tf.reduce_mean(tf.expand_dims(self.correct, 1) * self.target, 0)
        self.acc = tf.reduce_mean(self.correct)

    def make_train_op(self):
        train_ops = []
        for (learning_mode, learning_rate_relative) in Net.LEARNING_RATE_MODES.iteritems():
            variables = tf.get_collection(learning_mode)
            if variables:
                train_ops.append(tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate * learning_rate_relative,
                    epsilon=1.0).minimize(self.loss, global_step=self.global_step))
        self.train_op = tf.group(*train_ops)

    def make_show(self):
        def identity(value):
            return value

        postfix_funcs = {
            Net.Phase.TRAIN: {
                'raw': identity,
                'avg': lambda value: util.exponential_moving_average(value, num_updates=self.global_step)},
            Net.Phase.TEST: {
                'raw': identity,
                'avg': lambda value: util.exponential_moving_average(value, num_updates=self.global_step)}}

        self.show_dict = {
            phase: {
                '%s_%s_%s' % (phase.name, attr, postfix): func(getattr(self, attr))
                for (postfix, func) in postfix_funcs[phase].iteritems()
                for attr in ['loss', 'acc']}
            for phase in [Net.Phase.TRAIN, Net.Phase.TEST]}

        self.show_dict[Net.Phase.TRAIN].update({
            attr: getattr(self, attr) for attr in ['learning_rate']})

        self.summary = {
            phase: tf.merge_summary([tf.scalar_summary(name, attr) for (name, attr) in self.show_dict[phase].iteritems()])
            for phase in [Net.Phase.TRAIN, Net.Phase.TEST]}

    def finalize(self):
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_frac)))
        self.saver = tf.train.Saver(tf.get_collection(Net.NET_VARIABLES))
        self.summary_writer = tf.train.SummaryWriter(META.working_dir)

        self.sess.run(tf.initialize_all_variables())
        if os.path.isfile(self.model_path):
            print('Model restored from %s' % self.model_path)
            self.saver.restore(tf.get_default_session(), self.model_path)
        self.model = Model(self.global_step)

    def start(self, default_phase=Phase.NONE):
        self.sess.run(self.phase_assign, feed_dict={self.phase: default_phase.value})
        tf.train.start_queue_runners()
        print('Filling queues...')


class ResNet(Net):
    RESNET_PARAMS_PATH = os.path.join(ROOT_PATH, 'archive/ResNet-50-params.mat')
    NUM_TEST_CROPS = 4

    def __init__(self,
                 learning_rate=Net.LEARNING_RATE,
                 learning_modes=Net.LEARNING_RATE_MODES,
                 learning_rate_decay_steps=Net.LEARNING_RATE_DECAY_STEPS,
                 learning_rate_decay_rate=Net.LEARNING_RATE_DECAY_RATE,
                 weight_decay=Net.WEIGHT_DECAY,
                 gpu_frac=Net.GPU_FRAC,
                 resnet_params_path=RESNET_PARAMS_PATH,
                 num_test_crops=NUM_TEST_CROPS,
                 is_train=False,
                 is_show=False):

        super(ResNet, self).__init__(
            learning_rate=learning_rate,
            learning_modes=learning_modes,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_rate=learning_rate_decay_rate,
            weight_decay=weight_decay,
            gpu_frac=gpu_frac,
            is_train=is_train,
            is_show=is_show)

        self.resnet_params_path = resnet_params_path
        self.num_test_crops = num_test_crops
        if not os.path.isfile(self.model_path):
            self.resnet_params = scipy.io.loadmat(resnet_params_path)

    def get_initializer(self, name, index, is_vector, default):
        if os.path.isfile(self.model_path):
            return None
        elif name in self.resnet_params:
            print('%s initialized from ResNet' % name)
            if is_vector:
                value = self.resnet_params[name][index][0][:, 0]
            else:
                value = self.resnet_params[name][index][0]
            return tf.constant_initializer(value)
        else:
            return default

    def conv(self, value, conv_name, out_channel, size=(1, 1), stride=(1, 1), padding='SAME', biased=False, norm_name=None, activation_fn=None, learning_mode='normal'):
        in_channel = ImageUtil.get_channel(value)

        if self.learning_modes[learning_mode] > 0:
            collections = Net.NET_COLLECTIONS + [learning_mode]
            trainable = True
        else:
            collections = Net.NET_COLLECTIONS
            trainable = False

        weights_initializer = self.get_initializer(
            conv_name,
            index=0,
            is_vector=False,
            default=tf.truncated_normal_initializer(stddev=(2. / (in_channel * stride[0] * stride[1])) ** 0.5))

        if self.weight_decay > 0:
            weight_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
        else:
            weight_regularizer = None

        with tf.variable_scope(conv_name):
            weight = tf.get_variable(
                'weight',
                shape=size + (in_channel, out_channel),
                initializer=weights_initializer,
                regularizer=weight_regularizer,
                trainable=trainable,
                collections=collections)

        value = tf.nn.conv2d(value, weight, strides=Net.expand(stride), padding=padding)

        if biased:
            bias_initializer = self.get_initializer(
                conv_name,
                index=1,
                is_vector=True,
                default=tf.constant_initializer(0.1))

            with tf.variable_scope(conv_name):
                bias = tf.get_variable(
                    'bias',
                    shape=(out_channel,),
                    initializer=bias_initializer,
                    trainable=trainable,
                    collections=collections)

            value = tf.nn.bias_add(value, bias)

        if norm_name is not None:
            bn_name = 'bn%s' % norm_name
            scale_name = 'scale%s' % norm_name

            mean_initializer = self.get_initializer(
                bn_name,
                index=0,
                is_vector=True,
                default=tf.constant_initializer(0.0))

            variance_initializer = self.get_initializer(
                bn_name,
                index=1,
                is_vector=True,
                default=tf.constant_initializer(1.0))

            with tf.variable_scope(bn_name):
                mean = tf.get_variable(
                    'mean',
                    shape=(out_channel,),
                    initializer=mean_initializer,
                    trainable=False,
                    collections=Net.NET_COLLECTIONS)
                variance = tf.get_variable(
                    'variance',
                    shape=(out_channel,),
                    initializer=variance_initializer,
                    trainable=False,
                    collections=Net.NET_COLLECTIONS)

            scale_initializer = self.get_initializer(
                scale_name,
                index=0,
                is_vector=True,
                default=tf.constant_initializer(1.0))

            offset_initializer = self.get_initializer(
                scale_name,
                index=1,
                is_vector=True,
                default=tf.constant_initializer(0.0))

            with tf.variable_scope(scale_name):
                scale = tf.get_variable(
                    'scale',
                    shape=(out_channel,),
                    initializer=scale_initializer,
                    trainable=trainable,
                    collections=collections)
                offset = tf.get_variable(
                    'offset',
                    shape=(out_channel,),
                    initializer=offset_initializer,
                    trainable=trainable,
                    collections=collections)

            value = (value - mean) * tf.rsqrt(variance + util.EPSILON) * scale + offset

        if activation_fn is not None:
            with tf.variable_scope(conv_name):
                value = activation_fn(value)

        print('Layer %s, shape=%s, size=%s, stride=%s, learning_mode=%s' % (value.name, value.get_shape(), size, stride, learning_mode))
        return value

    def unit(self, value, name, subsample, out_channel, learning_mode='normal'):
        in_channel = ImageUtil.get_channel(value)

        if subsample:
            stride = (2, 2)
        else:
            stride = (1, 1)

        out_channel_inner = out_channel
        out_channel_outer = 4 * out_channel

        with tf.variable_scope(name):
            if subsample or in_channel != out_channel_outer:
                value1 = self.conv(value, 'res%s_branch1' % name, out_channel=out_channel_outer, stride=stride, norm_name='%s_branch1' % name, learning_mode=learning_mode)
            else:
                value1 = value

            value2 = self.conv(value, 'res%s_branch2a' % name, out_channel=out_channel_inner, stride=stride, norm_name='%s_branch2a' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
            value2 = self.conv(value2, 'res%s_branch2b' % name, out_channel=out_channel_inner, size=(3, 3), norm_name='%s_branch2b' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
            value2 = self.conv(value2, 'res%s_branch2c' % name, out_channel=out_channel_outer, norm_name='%s_branch2c' % name, learning_mode=learning_mode)

            value = tf.nn.relu(value1 + value2)
        return value

    def block(self, value, name, num_units, subsample, out_channel, learning_mode='normal'):
        for num_unit in xrange(num_units):
            value = self.unit(value, '%s%c' % (name, ord('a') + num_unit), subsample=subsample and num_unit == 0, out_channel=out_channel, learning_mode=learning_mode)
        return value

    def softmax(self, value, dim):
        value = tf.exp(value - tf.reduce_max(value, reduction_indices=dim, keep_dims=True))
        value = value / tf.reduce_sum(value, reduction_indices=dim, keep_dims=True)
        return value

    '''
    def test_segment_mean(self, value):
        batch_size = tf.shape(value)[0] / self.num_test_crops
        segment_ids = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), (-1, 1)), (1, self.num_test_crops)), (-1,))
        value = tf.segment_mean(value, segment_ids)
        return value

    def segment_mean(self, value):
        shape = ImageUtil.get_shape(value)

        value = self.case([
            (Net.Phase.TRAIN, lambda: value),
            (Net.Phase.TEST, lambda: self.test_segment_mean(value))])

        value.set_shape((None,) + shape[1:])
        return value
    '''

    def rebatch(self, value):
        num_crops = self.case([
            (Net.Phase.TRAIN, lambda: tf.constant(1, dtype=tf.int32)),
            (Net.Phase.TEST, lambda: tf.constant(self.num_test_crops, dtype=tf.int32))])

        batch_size = tf.shape(value)[0]
        size = ImageUtil.get_size(value)

        value = tf.reshape(value, (batch_size / num_crops, num_crops) + size)
        value.set_shape((None, None) + size)
        return value


class ResNet50(ResNet):
    def __init__(self,
                 learning_rate=Net.LEARNING_RATE,
                 learning_modes=Net.LEARNING_RATE_MODES,
                 learning_rate_decay_steps=Net.LEARNING_RATE_DECAY_STEPS,
                 learning_rate_decay_rate=Net.LEARNING_RATE_DECAY_RATE,
                 weight_decay=Net.WEIGHT_DECAY,
                 gpu_frac=Net.GPU_FRAC,
                 resnet_params_path=ResNet.RESNET_PARAMS_PATH,
                 num_test_crops=ResNet.NUM_TEST_CROPS,
                 is_train=False,
                 is_show=False):

        super(ResNet50, self).__init__(
            learning_rate=learning_rate,
            learning_modes=learning_modes,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_rate=learning_rate_decay_rate,
            weight_decay=weight_decay,
            gpu_frac=gpu_frac,
            resnet_params_path=resnet_params_path,
            num_test_crops=num_test_crops,
            is_train=is_train,
            is_show=is_show)

    def build(self, blob):
        assert len(blob.as_tuple_list()) == 1, 'Must pass in a single pair of image and label'
        (self.image, self.label) = blob.as_tuple_list()[0]

        with tf.variable_scope('1'):
            self.v0 = self.conv(self.image, 'conv1', size=(7, 7), stride=(2, 2), out_channel=64, biased=True, norm_name='_conv1', activation_fn=tf.nn.relu, learning_mode='slow')
            self.v1 = self.max_pool(self.v0, 'max_pool', size=(3, 3), stride=(2, 2))

        self.v2 = self.block(self.v1, '2', num_units=3, subsample=False, out_channel=64, learning_mode='slow')
        self.v3 = self.block(self.v2, '3', num_units=4, subsample=True, out_channel=128, learning_mode='slow')
        self.v4 = self.block(self.v3, '4', num_units=6, subsample=True, out_channel=256, learning_mode='normal')
        self.v5 = self.block(self.v4, '5', num_units=3, subsample=True, out_channel=512, learning_mode='normal')

        with tf.variable_scope('fc'):
            self.v6 = self.avg_pool(self.v5, 'avg_pool', size=(7, 7))
            self.v6_ = tf.squeeze(self.v6, (1, 2))
            self.v7 = self.conv(self.v6, 'fc', out_channel=len(META.class_names), biased=True)
            self.v8 = self.softmax(self.v7, 3)
            self.v8_ = tf.squeeze(self.v8, (1, 2))

        _feat = self.rebatch(self.v6_)
        self.feat = tf.reduce_mean(_feat, 1)
        _prob = self.rebatch(self.v8_)
        self.prob = tf.reduce_mean(_prob, 1)
        _consistency = - tf.reduce_sum(tf.expand_dims(self.prob, 1) * tf.log(_prob), 2)
        self.consistency = tf.exp(- tf.reduce_mean(_consistency, 1))

        self.make_stat()

        if self.is_train:
            self.make_train_op()

        if self.is_show:
            self.make_show()

        self.finalize()

    def train(self, iteration=0, feed_dict=dict(), save_per=-1):
        self.sess.run(self.phase_assign, feed_dict={self.phase: Net.Phase.TRAIN.value})

        train_dict = dict(train=self.train_op)
        show_dict = self.show_dict[Net.Phase.TRAIN]
        summary_dict = dict(summary=self.summary[Net.Phase.TRAIN])

        self.model.train(
            iteration=iteration,
            feed_dict=feed_dict,
            callbacks=[
                dict(fetch=util.merge_dicts(train_dict, show_dict, summary_dict)),
                dict(fetch=show_dict,
                     func=lambda **kwargs: self.model.display(begin='Train', end='\n', **kwargs)),
                dict(interval=5,
                     fetch=summary_dict,
                     func=lambda **kwargs: self.model.summary(summary_writer=self.summary_writer, **kwargs)),
                dict(interval=5,
                     func=lambda **kwargs: self.test(feed_dict=feed_dict)),
                dict(interval=save_per,
                     func=lambda **kwargs: self.model.save(saver=self.saver, saver_kwargs=dict(save_path=self.model_path, global_step=None), **kwargs))])

    def test(self, iteration=1, feed_dict=dict()):
        self.sess.run(self.phase_assign, feed_dict={self.phase: Net.Phase.TEST.value})

        show_dict = self.show_dict[Net.Phase.TEST]
        summary_dict = dict(summary=self.summary[Net.Phase.TEST])

        self.model.test(
            iteration=iteration,
            feed_dict=feed_dict,
            callbacks=[
                dict(fetch=util.merge_dicts(show_dict, summary_dict)),
                dict(fetch=show_dict,
                     func=lambda **kwargs: self.model.display(begin='\033[2K\rTest', end='\n', **kwargs)),
                dict(fetch=summary_dict,
                     func=lambda **kwargs: self.model.summary(summary_writer=self.summary_writer, **kwargs))])

    def online(self, feed_dict=dict(), fetch=dict()):
        self.sess.run(self.phase_assign, feed_dict={self.phase: Net.Phase.TEST.value})

        self.model.test(
            iteration=1,
            feed_dict=feed_dict,
            callbacks=[
                dict(fetch=fetch)])

        return self.model.output_values


class Postprocess(object):
    def __init__(self):
        pass

    def blob(self, values):
        return Blob(values=values)


class Consumer(object):
    BATCH_SIZE = 64
    NUM_TEST_CROPS = 4
    CAPACITY = 64

    def __init__(self,
                 batch_size=BATCH_SIZE,
                 num_test_crops=NUM_TEST_CROPS,
                 capacity=CAPACITY):

        self.batch_size = batch_size
        self.num_test_crops = num_test_crops
        self.capacity = capacity

    def build(self, blob):
        values = blob.values

        test_batch_size = self.batch_size / self.num_test_crops
        self.queue = tf.PaddingFIFOQueue(
            self.capacity,
            shapes=[(None,) + ImageUtil.get_size(value) for value in values],
            dtypes=[value.dtype for value in values])
        enqueue = self.queue.enqueue(values)
        queue_runner = tf.train.QueueRunner(self.queue, [enqueue])
        tf.train.add_queue_runner(queue_runner)

        total_size = tf.Variable(-1, trainable=False, dtype=tf.int32)
        dequeue_size = (total_size - 1) / test_batch_size + 1
        self.total_size = tf.placeholder_with_default(self.capacity * test_batch_size, shape=())
        self.assign = total_size.assign(self.total_size)

        values = prob_list(self.queue.dequeue_many(dequeue_size))
        values_ = list()
        for value in values:
            shape = ImageUtil.get_shape(value)
            value = tf.reshape(value, (-1,) + shape[2:])
            values_.append(value)
        return Blob(values=values_)

    def kwargs(self, total_size):
        return dict(
            feed_dict={self.total_size: total_size},
            fetch=dict(consumer_assign=self.assign))


class Timer(object):
    def __init__(self, message):
        self.message = message
        self.start = time.time()

    def __enter__(self):
        print(self.message)

    def __exit__(self, type, msg, traceback):
        if type:
            print(msg)
        else:
            print('Time: %.3f s' % (time.time() - self.start))
        return False
