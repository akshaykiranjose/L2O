import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model

class LSTMOpt(Model):
    
    """
    The optimizer defined here matches the one in the paper to the best of my interpretation.
    """
    
    def __init__(self, **kwargs):
        
        super(LSTMOpt, self).__init__(**kwargs)
        self.lstm1 = LSTM(20, 
                          return_state=True, 
                          return_sequences=True)
        
        self.lstm2 = LSTM(20, 
                          return_state=True)
        
        self.dense1 = Dense(1, 
                            name='1 dimensional update')
    
        
    def call(self, gradients, states):
        
        if states == None:
            h1,c1,h2,c2 = tf.zeros([10,20]), tf.zeros([10,20]), tf.zeros([10,20]), tf.zeros([10,20])
        else:
            h1,c1,h2,c2 = states
            
        if gradients == None:
            gradients = tf.zeros([10,1,1])
        else:
            gradients = tf.reshape(gradients, [10,1,1])
        
        seq, h1, c1 = self.lstm1(gradients, initial_state = [h1, c1])
        seq, h2,c2 = self.lstm2(seq, initial_state = [h2,c2])
        update = self.dense1(seq)
                        
        states = [h1, c1, h2, c2]
                      
        return update, states
    