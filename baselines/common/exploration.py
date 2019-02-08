import tensorflow as tf
from baselines.common.tf_util import adjust_shape
from baselines.common.input import observation_placeholder
from baselines.a2c.utils import fc

class AEReward:
    def __init__(self, ob_space, sess=None):
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
        self.bonus = tf.clip_by_value(tf.reduce_mean(tf.square(output_layer - self.X), 1), 0, 1, 'clip novelty') # novelty reward as AE loss per observation
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
