import tensorflow as tf
import numpy as np
from baselines import logger
from baselines.common.tf_util import adjust_shape
from baselines.common.input import observation_placeholder
from baselines.a2c.utils import fc

class AEReward:
    def __init__(self, ob_space, sess=None):
        logger.info("Using AEReward")
        self.sess = sess or tf.get_default_session()
        self.X = observation_placeholder(ob_space)
        initializer = tf.variance_scaling_initializer()
        num_inp = ob_space.shape[0]
        num_hid1 = num_inp // 2
        # num_hid2 = 16
        # num_hid3 = num_hid1

        w1 = tf.Variable(initializer([num_inp, num_hid1]), dtype=tf.float32)
        # w2 = tf.Variable(initializer([num_hid1, num_hid2]), dtype=tf.float32)
        # w3 = tf.Variable(initializer([num_hid2, num_hid3]), dtype=tf.float32)
        # w4 = tf.Variable(initializer([num_hid3, num_inp]), dtype=tf.float32)
        w4 = tf.Variable(initializer([num_hid1, num_inp]), dtype=tf.float32)

        # h = tf.layers.flatten(X)
        # h = fc(h, 'mlp_fc{}'.format(i), nh=num_hid1, init_scale=np.sqrt(2))
        # h = fc(h, 'mlp_fc{}'.format(i), nh=num_hid2, init_scale=np.sqrt(2))
        # h = fc(h, 'mlp_fc{}'.format(i), nh=num_hid1, init_scale=np.sqrt(2))

        b1 = tf.Variable(tf.zeros(num_hid1))
        # b2 = tf.Variable(tf.zeros(num_hid2))
        # b3 = tf.Variable(tf.zeros(num_hid3))
        b4 = tf.Variable(tf.zeros(num_inp))

        hid_layer1 = tf.nn.relu(tf.matmul(self.X, w1) + b1)
        # hid_layer2 = tf.nn.relu(tf.matmul(hid_layer1, w2) + b2)
        # hid_layer3 = tf.nn.relu(tf.matmul(hid_layer2, w3) + b3)
        output_layer = tf.nn.relu(tf.matmul(hid_layer1, w4) + b4)

        self.loss = tf.reduce_mean(tf.square(output_layer - self.X))
        self.bonus = tf.reduce_mean(tf.square(output_layer - self.X), 1) # novelty reward as AE loss per observation
        optimizer = tf.train.AdamOptimizer(0.01)
        self.train = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run(session=sess)

    def update(self, batch):
        sess = self.sess

    def get_batch_bonus_and_update(self, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)
        result = sess.run({"bonus": self.bonus, "train": self.train}, feed_dict)
        return result["bonus"]

class RNDReward:
    def __init__(self, ob_space, sess=None):
        logger.info("Using RNDReward")
        self.sess = sess or tf.get_default_session()
        # RND.
        num_inp = ob_space.shape[0]
        num_hid1 = 64 #num_inp // 2
        num_hid_pred = 32
        rep_size = 10
        proportion_of_exp_used_for_predictor_update = 1.

        self.X = observation_placeholder(ob_space)

        # Random target network.
        logger.info("CnnTarget: using shape %s as observation input" % (str(ob_space.shape)))
        xr = self.X
        xr = tf.nn.leaky_relu(fc(xr, 'fc1r', nh=num_hid1, init_scale=np.sqrt(2)))
        X_r = fc(xr, 'fc2r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        xrp = self.X
        xrp = tf.nn.leaky_relu(fc(xr, 'fc1r_pred', nh=num_hid1, init_scale=np.sqrt(2)))
        X_r_hat = tf.nn.relu(fc(xrp, 'fc1r_hat1_pred', nh=num_hid_pred, init_scale=np.sqrt(2)))
        X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=num_hid_pred, init_scale=np.sqrt(2)))
        X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=1)
        # self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        noisy_targets = tf.stop_gradient(X_r)
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
        optimizer = tf.train.AdamOptimizer(0.01)
        self.train = optimizer.minimize(self.aux_loss)

        tf.global_variables_initializer().run(session=sess)

    def get_batch_bonus_and_update(self, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)
        result = sess.run({"bonus": self.int_rew, "train": self.train}, feed_dict)
        return result["bonus"]