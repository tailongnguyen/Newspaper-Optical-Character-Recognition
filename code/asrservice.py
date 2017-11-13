
# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from data_generator import DataGenerator
from char_map import index_map
from keras.models import Model,model_from_json

import time
import tensorflow as tf
import keras.backend as K
from keras.layers import (BatchNormalization, Convolution2D, Dense, Input ,LSTM
, Bidirectional, TimeDistributed,Activation,Lambda)
from utils import argmax_decode, load_model,spectrogram_from_file
import soundfile
from flask import Flask,request
import io
from flask import Flask, render_template, jsonify, request, json

app = Flask(__name__)

                  
 

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def relu20(x):
    return K.relu(x,max_value=20)

def compile_asr_model(input_dim=161, output_dim=91, recur_layers=7):
    
    """ Build a recurrent network (CTC) for speech with LSTM units """
    print("Building asr model")
    # Main acoustic input
    acoustic_input = Input(shape=(None, input_dim,1), name='acoustic_input')
    conv2d_1 = Convolution2D(32, kernel_size = (11,41), strides=(2, 2), padding='valid',
                            data_format="channels_last")(acoustic_input)
    conv2d_1 = BatchNormalization(name='bn_conv1_2d')(conv2d_1)
    conv2d_1 =  Activation(relu20,name='conv1_relu20')(conv2d_1)

    
    
    conv2d_2 = Convolution2D(32, kernel_size = (11,21),strides=(1, 2), padding='valid',
                            data_format="channels_last")(conv2d_1)
    conv2d_2= BatchNormalization(name='bn_conv2_2d')(conv2d_2)
    conv2d_2 =  Activation(relu20, name='conv2_relu20')(conv2d_2)

        
    
    conv2d_3 = Convolution2D(96, kernel_size = (11,21),strides=(1, 2), padding='valid',
                            data_format="channels_last")(conv2d_2)
    conv2d_3 = BatchNormalization(name='bn_conv3_2d')(conv2d_3)
    conv2d_3 =  Activation(relu20,name='conv3_relu20')(conv2d_3)

    output = Lambda(function= lambda x : K.squeeze(x,axis=2))(conv2d_3)
    for r in range(recur_layers):
        output = Bidirectional(LSTM(units=768, return_sequences= True,activation='tanh',
                                    implementation=2,
                                    name='blstm_{}'.format(r+1)))(output)
        output = BatchNormalization(name='bn_rnn_{}'.format(r + 1))(output)
        

    network_output = TimeDistributed(Dense(output_dim,activation="linear",name="output"))(output)
    model = Model(inputs=acoustic_input,outputs=network_output)
    return model

def compile_output_fn(model):

    acoustic_input = model.inputs[0]
    network_output = tf.nn.softmax(model.outputs[0])
    
    output_fn = K.function([acoustic_input, K.learning_phase()],
                           [network_output])
    return output_fn

def featurize(audio_clip):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=10, window=20,
            max_freq=8000)


model = compile_asr_model()
model_weights_file = "/data/voice/train-data/ModelAsr/Model-demo/model_final.h5"
model.load_weights(model_weights_file)
test_fn = compile_output_fn(model) 

def predict(test_desc_file):
    # Prepare the data generator
    datagen = DataGenerator()

    b = time.time()
    
    # Test the model
    test_desc_file = test_desc_file
    if test_desc_file != '':
            file = test_desc_file
            inputs = [featurize(file)]
            probs_tf = np.squeeze(test_fn([inputs, True]))

            b = time.time()
            decode , _ = tf.nn.ctc_greedy_decoder(probs_tf,[probs_tf.shape[0]])
            #decode,_ =  tf.nn.ctc_beam_search_decoder(probs_tf,[probs_tf.shape[0]],merge_repeated=True,beam_width=200)
            d = K.get_value(decode[0])
            
            dense_decoded = K.get_value(tf.sparse_tensor_to_dense(d, default_value=-1))
            for seq in   dense_decoded:
                seq = [s for s in seq if s != -1 and s != 0]
                pre = ''.join([index_map[i] for i in seq])
            
            return pre

app.route("/voice",methods=['POST'])
def voiceTotext():
    data = request.form['data']
    pre = predict(data)
    return pre

app.run(host='0.0.0.0',port=2907)