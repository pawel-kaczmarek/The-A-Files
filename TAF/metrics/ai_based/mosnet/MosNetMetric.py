import os
from numbers import Number

import librosa
import numpy as np
import scipy
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional

from TAF.models.Metric import Metric

# prevent TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MosNetMetric(Metric):

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        # constants
        self.fixed_rate = fs
        self.mono = True
        self.absolute = True

        self.FFT_SIZE = 512
        self.SGRAM_DIM = self.FFT_SIZE // 2 + 1
        self.HOP_LENGTH = 256
        self.WIN_LENGTH = 512

        self.initialize()

        magnitude_original = self.to_magnitude(samples_original)
        magnitude_processed = self.to_magnitude(samples_processed)

        # now call the actual MOSnet
        predict_original = self.model.predict(magnitude_original[None, ...], verbose=0, batch_size=1)
        predict_processed = self.model.predict(magnitude_processed[None, ...], verbose=0, batch_size=1)
        return np.concatenate([predict_original[0][0], predict_processed[0][0]])

    def name(self) -> str:
        return "MOSNet: Deep Learning based Objective Assessment for Voice Conversion"

    def to_magnitude(self, samples):
        # stft. D: (1+n_fft//2, T)
        linear = librosa.stft(
            y=np.asfortranarray(samples),
            n_fft=self.FFT_SIZE,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
            window=scipy.signal.hamming,
        )

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft/2, T)

        # shape in (T, 1+n_fft/2)
        return np.transpose(mag.astype(np.float32))

    def initialize(self):
        _input = keras.Input(shape=(None, 257))

        re_input = layers.Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)

        # CNN
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(re_input)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(conv1)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 3), activation='relu',
                        padding='same'))(conv1)

        conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(conv1)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(conv2)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 3), activation='relu',
                        padding='same'))(conv2)

        conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(conv2)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(conv3)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 3), activation='relu',
                        padding='same'))(conv3)

        conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(conv3)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                        padding='same'))(conv4)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 3), activation='relu',
                        padding='same'))(conv4)

        re_shape = layers.Reshape((-1, 4 * 128), input_shape=(-1, 4, 128))(conv4)

        # BLSTM
        blstm1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3,
                 recurrent_dropout=0.3,
                 recurrent_constraint=max_norm(0.00001)),
            merge_mode='concat')(re_shape)

        # DNN
        flatten = TimeDistributed(layers.Flatten())(blstm1)
        dense1 = TimeDistributed(Dense(128, activation='relu'))(flatten)
        dense1 = Dropout(0.3)(dense1)

        frame_score = TimeDistributed(Dense(1), name='frame')(dense1)

        average_score = layers.GlobalAveragePooling1D(name='avg')(frame_score)

        self.model = Model(outputs=[average_score, frame_score], inputs=_input)

        # weights are in the directory of this file
        pre_trained_dir = os.path.dirname(__file__)

        # load pre-trained weights. CNN_BLSTM is reported as best
        self.model.load_weights(os.path.join(pre_trained_dir, 'cnn_blstm.h5'))
