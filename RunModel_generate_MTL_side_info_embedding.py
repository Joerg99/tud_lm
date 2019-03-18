#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
from __future__ import print_function
import nltk
from util.preprocessing_side_info_embedding import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM_uni import BiLSTM_uni
import sys
import numpy as np
import time
import keras.losses
from _operator import pos
from warnings import catch_warnings
import tensorflow as tf

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=sess_config))


# :: Load the model ::
temperature = 1


modelPath_1 = 'models/chicago/mtl_without_rhyme/chicago_mtl1_292.9635_331.0788_100.h5' 
modelname_1 = 'chicago_mtl1'
lstmModel_1 = BiLSTM_uni.loadModel(modelPath_1, 1)

modelPath_2 = 'models/chicago/mtl_without_rhyme/chicago_mtl2_292.9635_331.0788_100.h5' 
modelname_2 = 'chicago_mtl2'
lstmModel_2 = BiLSTM_uni.loadModel(modelPath_2, 1)

# modelPath_3 = 'models/chicago/mtl/chicago_mtl3_160.6308_164.5199_100.h5' 
# modelname_3 = 'chicago_mtl3'
# lstmModel_3 = BiLSTM_uni.loadModel(modelPath_3, 1)


for run in range(15): ######### number of files with a certain side value 
    
    
    side_info_allit = ['start'] ###### START WITH
    side_info_rhyme = ['start']
    try:
        i=0
        quatrains = []
        while i < 10: ########## number of samples
            text = 'sos'
            generation_mode = 'sample' # 'max' or 'sample'
            predictions_sampled_1 = [[]]
            predictions_sampled_2 = [[]]
            predictions_sampled_3 = [[]]
            
            #this is generating a sequence. 
            while True:
                sentences = []
                for sent in nltk.sent_tokenize(text):
                    word_token = nltk.word_tokenize(sent)
                    
                    sentences.append({'tokens': word_token, 'side_info_allit': side_info_allit}) #, 'side_info_rhyme': side_info_rhyme})
                
                addCharInformation(sentences)
                 
                dataMatrix_1 = createMatrices(sentences, lstmModel_1.mappings, True)
                dataMatrix_2 = createMatrices(sentences, lstmModel_2.mappings, True)
#                 dataMatrix_3 = createMatrices(sentences, lstmModel_3.mappings, True)
                                        
                # :: Tag the input ::
                tags_1 = lstmModel_1.tagSentences_generate(dataMatrix_1, predictions_sampled_1, generation_mode)
                
                tags_2 = lstmModel_2.tagSentences_generate(dataMatrix_2, predictions_sampled_2, generation_mode)
                allit_info_from_model = tags_2[modelname_2][0][-1]
                side_info_allit.append(allit_info_from_model)
                
#                 tags_3 = lstmModel_3.tagSentences_generate(dataMatrix_3, predictions_sampled_3, generation_mode)
#                 rhyme_info_from_model = tags_3[modelname_3][0][-1]
#                 side_info_rhyme.append(rhyme_info_from_model)
                
                
                text +=' '+tags_1[modelname_1][0][-1]
                if tags_1[modelname_1][0][-1] in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags_1[modelname_1][0]) == 100:
                    print(text)
                    quatrains.append(text)
                    print(side_info_allit)
#                     print(side_info_rhyme)
                    
#                     allits_string = ''
#                     for v in side_info_allit:
#                         allits_string += allits_string+' '
#                     
#                     quatrains.append(allits_string)
                    i+=1
                    side_info_allit = ['start']
#                     side_info_rhyme = ['start']
                    break


        with open('evaluation_files/'+modelname_1+'/embedding/mtl_without_rhyme/'+modelname_1+str(run)+'ep1005', 'w') as file:
            for quatrain in quatrains:
                file.write('%s \n' %quatrain)
    except Exception as e:
        print('Exception!!!! ', e)
        continue
    
    
'''   
modelPath = 'models/chicago/mtl/chicago_mtl1_164.2228_176.2986_18.h5' 


modelname = 'chicago_mtl1'
temperature = 1
lstmModel = BiLSTM_uni.loadModel(modelPath, temperature)
# side_info_allit_fix = ['B', 'I', 'O'] * 100
for run in range(1): ######### number of files with a certain side value 
    s_info_allit = 'B'
    s_info_rhyme = '0'
    try:
        i=0
        quatrains = []
        while i < 200: ########## number of samples
            text = 'sos'
            generation_mode = 'sample' # 'max' or 'sample'
            predictions_sampled = [[]]
            
            #this is generating a sequence. 
            while True:
                sentences = []
                for sent in nltk.sent_tokenize(text):
                    word_token = nltk.word_tokenize(sent)
                    
                    side_info_allit = [str(s_info_allit)] * len(word_token)
#                     side_info_allit.extend(['I'] * (len(word_token)-1))
#                     side_info_allit = side_info_allit_fix[:len(word_token)]
#                     print(side_info_allit)
                    side_info_rhyme = [str(s_info_rhyme)] * len(word_token)
                    sentences.append({'tokens': word_token, 'side_info_allit': side_info_allit, 'side_info_rhyme': side_info_rhyme})
                
                
                
                print(sentences)
                
                addCharInformation(sentences)
                 
                dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
                                        
                #addCasingInformation(sentences)
                #print('dataMatrix ', dataMatrix)
                #print(dataMatrix[0]['tokens'])
                # :: Tag the input ::
                tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
                text +=' '+tags[modelname][0][-1]
                if tags[modelname][0][-1] in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags[modelname][0]) == 100:
                    print('neuer Text: ', text)
                    quatrains.append(text)
                    i+=1
                    break
        with open('evaluation_files/'+modelname+'/embedding/'+modelname+str(run), 'w') as file:
            for quatrain in quatrains:
                file.write('%s \n' %quatrain)
    except Exception as e:
        print('Exception!!!! ', e)
        continue
'''
    
# short_verses = []
# for _ in range(10):
#     text = 'sos'
#     generation_mode = 'sample' # 'max' or 'sample'
#     predictions_sampled = [[]]
#     while True:
#         sentences = []
#         for sent in nltk.sent_tokenize(text):
#             word_token = nltk.word_tokenize(sent)
#             side_info = [""] * len(word_token)
#             sentences.append({'tokens': word_token, 'side_info': side_info})
#          
#         #print('sentences: ', sentences)
#          
#         addCharInformation(sentences)
#          
#         dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
#                                 
#         #addCasingInformation(sentences)
#         #print('dataMatrix ', dataMatrix)
#         #print(dataMatrix[0]['tokens'])
#         # :: Tag the input ::
#         tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
#         #print('tags ', tags)
#         text +=' '+tags[modelname][0][-1]
#         if tags[modelname][0][-1] in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags[modelname][0]) == 100:
#             print('neuer Text: ', text)
#             short_verses.append(len(text.split(' '))-1)
#             break
# print(np.mean(short_verses))


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
