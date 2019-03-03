#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
from __future__ import print_function
import nltk
from util.preprocessing_side_info_real_value import addCharInformation, createMatrices__side_info_real_value, addCasingInformation

from neuralnets.BiLSTM_uni_real_value import BiLSTM_uni
import sys
import numpy as np
import time
import keras.losses
from numpy import newaxis
import tensorflow as tf

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=sess_config))


# if len(sys.argv) < 3:
#     print("Usage: python RunModel.py modelPath inputPath")
#     exit()

#inputPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/input.txt'

# :: Read input ::
# with open(inputPath, 'r') as f:
#     text = f.read()

# :: Load the model ::
modelPath = 'models/chicago/allit/real_value/chicago_442.9203_465.2058_20.h5' 


modelname = 'chicago'
temperature = 1
lstmModel = BiLSTM_uni.loadModel(modelPath, temperature)

# for rhyme_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8, 0.9, 1, 2]:
for s_info in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]:
# for s_info in [32.0, 64.0, 128.0]:
    print('starting')
    i=0
    quatrains = []
    while i < 50:
        text = 'sos'
        generation_mode = 'sample' # 'max' or 'sample'
        predictions_sampled = [[]]
        while True:
            sentences = []
            for sent in nltk.sent_tokenize(text):
                word_token = nltk.word_tokenize(sent)
    #             side_info = [[4]] * len(word_token)
    #             side_info = side_info[:,:,newaxis]
                side_info = [s_info]
    #             print(type(side_info), np.shape(side_info), side_info)
                sentences.append({'tokens': word_token, 'side_info': side_info})
            
            addCharInformation(sentences)
            
            dataMatrix = createMatrices__side_info_real_value(sentences, lstmModel.mappings, True)
                                   
            #addCasingInformation(sentences)
            #print('dataMatrix ', dataMatrix)
            #print(dataMatrix[0]['tokens'])
            # :: Tag the input ::
            tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
            #print('tags ', tags)
            text +=' '+tags[modelname][0][-1]
            if tags[modelname][0][-1] in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags[modelname][0]) == 100:
                print('neuer Text: ', text)
                quatrains.append(text)
                i+=1
                break
    with open('evaluation_files/'+modelname+'/sentiment/real_value/'+modelname+str(s_info), 'w') as file:
        for quatrain in quatrains:
            file.write('%s \n' %quatrain)
    print('wrote a file')
    time.sleep(60)
    




# pos_neg = [0,0]
# for j in range(100):
#     start = 'sos'
#     if j < 50:
#         appendix = '_n'
#         text = start+appendix
#     else:
#         appendix = '_p'
#         text = start+appendix
#     generation_mode = 'sample' # 'max' or 'sample'
#     predictions_sampled = [[]]
#     # :: Prepare the input ::
#     while True:
#         sentences = []
#         for sent in nltk.sent_tokenize(text):
#             word_token = nltk.word_tokenize(sent)
#             for i in range(len(word_token)):
#                 if word_token[i][-2:] != appendix:
#                     word_token[i] = word_token[i]+appendix
#             sentences.append({'tokens': word_token})
#                                
# #         addCharInformation(sentences)
# #         addCasingInformation(sentences)
#         dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
#         # :: Tag the input ::
#         tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
#         text +=' '+tags['textgrid'][0][-1]
#         if tags['textgrid'][0][-1]in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags['textgrid'][0]) == 300:
#             print('neuer Text: ', text)
#             if appendix == '_p':
#                 pos_neg[0] += len(text.split(' '))
#             else:
#                 pos_neg[1] += len(text.split(' '))
#             break
#     #time.sleep(1)
# print(pos_neg, pos_neg[0]/50, pos_neg[1]/50)
