import tensorflow as tf

# Note: We refer to the code in this link
# (https://github.com/nikhil-kotecha/Generating_Music/blob/master/MyFunctions.py)
# when we try to transform the input data.

class transformLayer(tf.keras.layers.Layer):
    '''
    The very first layer that is used to transform the data into the
    expanded shape.
    Input: [play, articulate] for each note and each timestep
    Output: [pitch, pitch_class, prev_vicinity, context, beat, zero]
    '''

    def __init__(self, input_dim):
        super(transformLayer, self).__init__()
        self.batch_size = input_dim[0]
        self.num_notes = input_dim[2]
        self.num_timesteps = input_dim[1]
        self.out = tf.Variable(initial_value=tf.zeros((self.batch_size, self.num_timesteps, self.num_notes, 80)),
                               trainable=False)

    def gen_pitch(self):
        pitches = tf.squeeze(tf.range(start=0, limit=self.num_notes, delta=1))
        pitch = tf.ones((self.batch_size, self.num_timesteps, 1, self.num_notes), dtype='float32') * tf.cast(pitches,
                                                                                                        tf.float32)
        pitch = tf.transpose(pitch, perm=[0, 3, 1,
                                          2])  # shape = batch_size, num_notes, num_timesteps, 1 with the last axis containing pitch

        pitchclasses = tf.squeeze(pitch % 12, axis=3)  # shape = batch_size, num notes, num_timesteps
        pitch_class = tf.one_hot(tf.cast(pitchclasses, dtype=tf.uint8), depth=12,
                                 dtype=tf.float32)  # shape = batch_size,num_notes, num_timesteps,num_pitchclass=12
        return pitch, pitch_class

    def gen_vicinity(self, input_data):
        input_flatten = tf.transpose(input_data, perm=[0, 2, 1, 3])
            # shape = batch_size,num_timesteps,num_notes,2
        input_flatten = tf.reshape(input_flatten, [self.batch_size * self.num_timesteps, self.num_notes,
                                                   2])  # channel for play and channel for articulate

        input_flatten_p = tf.expand_dims(input_flatten[:, :, 0], axis=2)  # shape=batch_size*num_timesteps,num_notes,1
        input_flatten_a = tf.expand_dims(input_flatten[:, :, 1], axis=2)

        filt_vicinity = tf.expand_dims(tf.eye(25, dtype=tf.float32), axis=1)  # shape = 25,1,25

        vicinity_p = tf.nn.conv1d(input_flatten_p, filt_vicinity, stride=1, padding='SAME')
        vicinity_a = tf.nn.conv1d(input_flatten_a, filt_vicinity, stride=1,
                                  padding='SAME')  # shape=batch_size*num_timesteps,num_notes,25

        vicinity = tf.stack([vicinity_p, vicinity_a], axis=3)  # shape=batch_size*num_timesteps,num_notes,25,2
        vicinity = tf.unstack(vicinity,
                              axis=2)
            # turn into a list of len 25, containing tensors with shpe =batch_size*num_timesteps,num_notes,2
        vicinity = tf.concat(vicinity,
                             axis=2)  # concat the list to a tensor of shape=batch_size*num_timesteps,num_notes,50
        prev_vicinity = tf.reshape(vicinity, shape=[self.batch_size, self.num_timesteps, self.num_notes, 50])
        prev_vicinity = tf.transpose(prev_vicinity, perm=[0, 2, 1, 3])  # shape = batch_size,num_notes,num_timesteps,50

        return prev_vicinity

    def gen_context(self, input_data):

        input_flatten = tf.transpose(input_data, perm=[0, 2, 1, 3])
            # shape = batch_size,num_timesteps,num_notes,2
        input_flatten = tf.reshape(input_flatten, [self.batch_size * self.num_timesteps, self.num_notes,
                                                   2])  # channel for play and channel for articulate
        input_flatten_p = tf.slice(input_flatten, [0, 0, 0],
                                   size=[-1, -1, 1])  # shape=batch_size*num_timesteps,num_notes,1
        input_flatten_p_bool = tf.minimum(input_flatten_p, 1)  # shape=batch_size*num_timesteps,num_notes,1
        filt_context = tf.expand_dims(tf.tile(tf.eye(12, dtype=tf.float32), multiples=[(self.num_notes // 12) * 2, 1]),
                                      axis=1)  # shape=240,1,12

        context = tf.nn.conv1d(input_flatten_p_bool, filt_context, stride=1,
                               padding='SAME')  # shape=shape=batch_size*num_timesteps,num_notes,12
        context = tf.reshape(context, shape=[self.batch_size, self.num_timesteps, self.num_notes, 12])
        context = tf.transpose(context, perm=[0, 2, 1, 3])  # shape = batch_size,num_notes,num_timesteps,12

        return context

    def gen_beat(self, time_init):
        # beat: boolean values of each time index being in [1/4, 1/2,1,2] beats
        timesteps = tf.range(time_init, self.num_timesteps + time_init)  # shape=128
        time = tf.reshape(tf.tile(timesteps, multiples=[self.batch_size * self.num_notes]),
                          # repeat time index for each notes in each batch
                          shape=[self.batch_size, self.num_notes, self.num_timesteps, 1])
        beat = tf.cast(tf.concat([time % 2, time // 2 % 2, time // 4 % 2, time // 8 % 2], axis=-1),
                       dtype=tf.float32)  # shape=batch_size,num_notes,num_timesteps,4

        return beat

    def gen_input(self, input_data, time_init=0):
        """
        Arguments:
            input_data: size = [batch_size x num_notes x num_timesteps x 2]
                (the input data represents that at the previous timestep of what we are trying to predict)
            time_init: integer representing where the 'beat' component begins for the batch.
        Returns:
            Note_State_Expand: size = [batch_size x num_notes x num_timesteps x 80]
        """

        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

        pitch, pitch_class = self.gen_pitch()

        prev_vicinity = self.gen_vicinity(input_data)

        context = self.gen_context(input_data)

        beat = self.gen_beat(time_init)

        # zero
        zero = tf.zeros([self.batch_size, self.num_notes, self.num_timesteps, 1], dtype=tf.float32)

        # Final Vector,shape=batch_size,num_notes,num_timesteps,80
        Input_Expand = tf.concat([pitch, pitch_class, prev_vicinity, context, beat, zero], axis=-1)
        Input_Expand = tf.transpose(Input_Expand, perm=[0, 2, 1, 3])  # change axis of time and note
        Input_Expand = tf.reshape(Input_Expand, shape=[self.batch_size * self.num_notes,
                                                       self.num_timesteps, 80])
        return Input_Expand

    def call(self, inputs):
        self.out = self.gen_input(inputs)
        return self.out


class changeAxisLayer1(tf.keras.layers.Layer):
    '''
    The layer that convert the output of time-aixs LSTM layers to the
    input of the note-axis LSTM layers.
    '''
    def __init__(self, input_dim):
        super(changeAxisLayer1, self).__init__()
        self.out = tf.Variable(initial_value=tf.zeros((10, 128, 10240)), trainable=False)

    def changeAxis(self, t):
        num_notes=128
        batch_size = 10
        num_timesteps= 128
        num_units = 300
        out = tf.reshape(t, shape=[batch_size,num_notes,num_timesteps,num_units])
        out = tf.transpose(out, perm=[0,2,1,3])
        out = tf.reshape(out, shape=[batch_size*num_timesteps,num_notes,num_units])
        return out

    def call(self, inputs):
        self.out = self.changeAxis(inputs)
        return self.out

class changeAxisLayer2(tf.keras.layers.Layer):
    '''
    The layer that convert the output of note-aixs LSTM layers to the
    format of final output that is ready to convert to MIDI files.
    '''
    def __init__(self, input_dim):
        super(changeAxisLayer2, self).__init__()
        self.out = tf.Variable(initial_value=tf.zeros((10, 128, 10240)), trainable=False)

    def changeAxis(self,t):

        num_notes=128
        batch_size = 10
        num_timesteps= 128
        num_units = 50
        out = tf.reshape(t,shape=[batch_size,num_timesteps,num_notes,num_units])
        return out
    def call(self, inputs):
        self.out=self.changeAxis(inputs)
        return self.out

    
class LossAligned(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = y_pred[:,:-1,:,:]
        y_true = y_true[:,1:,:,:]
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)