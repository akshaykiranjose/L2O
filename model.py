import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

class LSTMOpt(tf.keras.models.Model):
    """
    The optimizer defined in the paper with a few changes.
    Architecture is slightly different.
    """
    def __init__(self, **kwargs):
        
        super(LSTMOpt, self).__init__(**kwargs)
        self.lstm1 = LSTM(20, 
                          return_state=True, 
                          return_sequences=True)
        
        self.lstm2 = LSTM(20, 
                          return_state=True)
        
        self.dense1 = Dense(10, 
                            name='10 dim update')
    
        
    def call(self, gradients, states):
        
        if states == None:
            h1,c1,h2,c2 = tf.zeros([128,20]), tf.zeros([128,20]), tf.zeros([128,20]), tf.zeros([128,20])
        else:
            h1,c1,h2,c2 = states
            
        if gradients == None:
            gradients = tf.zeros([128,1,10])
        else:
            gradients = tf.reshape(gradients, [128,1,10])
        
        seq, h1, c1 = self.lstm1(gradients, initial_state = [h1, c1])
        _, h2,c2 = self.lstm2(seq, initial_state = [h2,c2])
        update = self.dense1(h2)
                        
        states = [h1, c1, h2, c2]
                      
        return update[..., None], states

