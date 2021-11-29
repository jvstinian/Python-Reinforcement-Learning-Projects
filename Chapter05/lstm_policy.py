'''
Created on 31 May 2017

@author: ywz
'''
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from layer import conv2d, linear


class LSTMPolicy:
    
    def __init__(self, input_shape=(84, 84, 4), n_outputs=4, network_type='cnn'):
        
        self.width = input_shape[0]
        self.height = input_shape[1]
        self.channel = input_shape[2]
        self.n_outputs = n_outputs
        self.network_type = network_type
        self.entropy_beta = 0.01
        
        self.x = tf.placeholder(dtype=tf.float32, 
                                shape=(None, self.channel, self.width, self.height))
        self.build_model()
        
    def build_model(self):
        
        self.net = {}
        self.net['input'] = tf.transpose(self.x, perm=(0, 2, 3, 1))
            
        if self.network_type == 'cnn':
            self.net['conv1'] = conv2d(self.net['input'], 16, kernel=(8, 8), stride=(4, 4), name='conv1')
            self.net['conv2'] = conv2d(self.net['conv1'], 32, kernel=(4, 4), stride=(2, 2), name='conv2')
            self.net['feature'] = linear(self.net['conv2'], 256, name='fc1')
        else:
            # MLP for testing
            self.net['fc1'] = linear(self.net['input'], 50, init_b = tf.constant_initializer(0.0), name='fc1')
            self.net['feature'] = linear(self.net['fc1'], 50, init_b = tf.constant_initializer(0.0), name='fc2')
        
        num_units = self.net['feature'].get_shape().as_list()[-1]
        # self.lstm = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=0.0, state_is_tuple=True)
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, forget_bias=0.0, state_is_tuple=True)
        self.init_state = self.lstm.zero_state(batch_size=1, dtype=tf.float32)
        
        step_size = tf.shape(self.x)[:1]
        feature = tf.expand_dims(self.net['feature'], axis=0)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm, feature, 
                                                     initial_state=self.init_state, 
                                                     sequence_length=step_size,
                                                     time_major=False)
        outputs = tf.reshape(lstm_outputs, shape=(-1, num_units))
        self.final_state = lstm_state
        
        self.net['value'] = tf.reshape(linear(outputs, 1, activation=None, name='value',
                                              init_b = tf.constant_initializer(0.0)), 
                                       shape=(-1,))
        
        self.net['logits'] = linear(outputs, self.n_outputs, activation=None, name='logits',
                                    init_b = tf.constant_initializer(0.0))
        
        self.net['policy'] = tf.nn.softmax(self.net['logits'], name='policy')
        self.net['log_policy'] = tf.nn.log_softmax(self.net['logits'], name='log_policy')
        
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    
    def build_gradient_op(self, clip_grad=None):

        self.action = tf.placeholder(dtype=tf.float32, shape=(None, self.n_outputs), name='action')
        self.reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None,), name='advantage')

        value = self.net['value']
        policy = self.net['policy']
        log_policy = self.net['log_policy']
        
        entropy = -tf.reduce_sum(policy * log_policy, axis=1)
        p_loss = -tf.reduce_sum(tf.reduce_sum(log_policy * self.action, axis=1) * self.advantage + 
                                self.entropy_beta * entropy)
        v_loss = 0.5 * tf.reduce_sum((value - self.reward) ** 2)
        total_loss = p_loss + v_loss
        
        self.gradients = tf.gradients(total_loss, self.vars)
        if clip_grad is not None:
            self.gradients, _ = tf.clip_by_global_norm(self.gradients, clip_grad)
        
        # Add summaries
        tf.summary.scalar("policy_loss", p_loss, collections=['policy_network'])
        tf.summary.scalar("value_loss", v_loss, collections=['policy_network'])
        tf.summary.scalar("entropy", tf.reduce_mean(entropy), collections=['policy_network'])
        # tf.summary.scalar("grad_global_norm", tf.global_norm(self.gradients), collections=['policy_network'])
        self.summary_op = tf.summary.merge_all('policy_network')
        
        return self.gradients
    
    def run_initial_state(self, sess):
        return sess.run(self.init_state)
    
    def run_value(self, sess, state, cell, *args):
        feed_dict={self.x: state, self.init_state[0]: cell[0], self.init_state[1]: cell[1]}
        value = sess.run(self.net['value'], feed_dict=feed_dict)
        return value
    
    def run_policy_and_value(self, sess, state, cell, *args):
        feed_dict={self.x: state, self.init_state[0]: cell[0], self.init_state[1]: cell[1]}
        policy, value, final_state = sess.run([self.net['policy'], self.net['value'], self.final_state],
                                              feed_dict=feed_dict)
        return policy, value, final_state
    
    def get_feed_dict(self, states, actions, rewards, advantages, cell, *args):
        feed_dict={self.x: states, self.action: actions, 
                   self.reward: rewards, self.advantage: advantages,
                   self.init_state[0]: cell[0], self.init_state[1]: cell[1]}
        return feed_dict
    
    