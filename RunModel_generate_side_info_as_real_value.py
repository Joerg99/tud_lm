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
# modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/chicago/real_value_relative_allit/chicago_498.2776_508.7411_65.h5' 
# modelname = 'chicago'

modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/gutentag/sentiment/gutentag_sent_214.2632_207.5308_35.h5' 
modelname = 'gutentag_sent'



i=0
temperature = 1
lstmModel = BiLSTM_uni.loadModel(modelPath, 1)

# for s_info in [-2, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]:
for s_info in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
# for temperature in [0.1, 0.4, 0.6, 0.8, 1]:
# for s_info in [-1, -0.5,  0.0, 0.5, 1]:
    print('starting')
    i=0
    quatrains = []
    while i < 15:
        text = 'sos'
        generation_mode = 'sample' # 'max' or 'sample'
        predictions_sampled = [[]]
        while True:
            sentences = []
            for sent in nltk.sent_tokenize(text):
                word_token = nltk.word_tokenize(sent)
#                 side_info = [[s_info]] * len(word_token)
#                 side_info = side_info[:,:,newaxis]
    #             print(type(side_info), np.shape(side_info), side_info)

            
                ####### use for gutentag_sent #######
                if modelname.startswith('gutentag'):
                    sentences.append({'tokens': word_token, 'side_info': s_info})
                ####### use for all other cases #######
                else: 
                    side_info = [s_info]
                    sentences.append({'tokens': word_token, 'side_info': side_info})
            
            addCharInformation(sentences)
             
            dataMatrix = createMatrices__side_info_real_value(sentences, lstmModel.mappings, True)
                                    
            #addCasingInformation(sentences)
            print('dataMatrix ', dataMatrix)
            #print(dataMatrix[0]['tokens'])
            # :: Tag the input ::
            tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode, modelname, temperature)
            #print('tags ', tags)
            text +=' '+tags[modelname][0][-1]
            if tags[modelname][0][-1] in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags[modelname][0]) == 100:
                print('neuer Text: ', text)
                quatrains.append(text)
                i+=1
                break

#     with open('softmax_eval/softmax_eval_'+str(s_info), 'w') as file:
#         for line in lstmModel.n_largest_recordings_from_softmax:
#             for word in line:
#                 file.write(word+' ')
#             file.write('\n')
#     lstmModel.n_largest_recordings_from_softmax = []
    
    with open('evaluation_files/'+modelname+'/allit_log/'+modelname+'_'+str(s_info), 'w') as file:
        for quatrain in quatrains:
            file.write('%s \n' %quatrain)
    print('wrote a file')


#     time.sleep(60)
    




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
