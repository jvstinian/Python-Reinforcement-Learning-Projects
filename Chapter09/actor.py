from keras import layers, models, optimizers, regularizers
from keras import backend as K


class Actor:
    
    
  # """Actor (policy) Model. """

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        
        net = layers.Dense(units=16,kernel_regularizer=regularizers.l2(1e-6))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=32,kernel_regularizer=regularizers.l2(1e-6))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        actions = layers.Dense(units=self.action_size, activation='softmax', name = 'actions')(net)
        
        self.model = models.Model(inputs=states, outputs=actions)

        action_gradients = layers.Input(shape=(self.action_size,))
        # action_gradients = K.gradients(actions, states) # TODO

        loss = K.mean(-action_gradients * actions)

        optimizer = optimizers.Adam(lr=.00001)
        self.optimizer = optimizer # TODO 
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        # print('updates_op variables: %s' % (updates_op,))
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[loss], # TODO: Is this correct?
            updates=updates_op)
        
        # TODO: Remove later
        import numpy as np
        example_states = np.random.random_sample( (10,self.state_size) )
        example_action_gradients = np.random.random_sample( (10, self.action_size) )
        print('train_fn output: %s' % (self.train_fn([example_states, example_action_gradients, 0]),))