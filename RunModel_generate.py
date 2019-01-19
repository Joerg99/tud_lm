#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
from __future__ import print_function
import nltk
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
from neuralnets.BiLSTM_uni import BiLSTM_uni
import sys
import numpy as np
import time
import keras.losses
from _operator import pos


# if len(sys.argv) < 3:
#     print("Usage: python RunModel.py modelPath inputPath")
#     exit()

#inputPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/input.txt'

# :: Read input ::
# with open(inputPath, 'r') as f:
#     text = f.read()

# :: Load the model ::
modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/test/pos_neg_textgrid_563.7473_1185.1209_11.h5' # with perplexity and POS label DOESNT RUN
lstmModel = BiLSTM_uni.loadModel(modelPath, 0.1)

start = 'ich'
appendix = '_p'
text = start+appendix
generation_mode = 'sample' # 'max' or 'sample'

predictions_sampled = [[]]
while True:
    #sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
    sentences = []
    for sent in nltk.sent_tokenize(text):
        word_token = nltk.word_tokenize(sent)
        for i in range(len(word_token)):
            if word_token[i][-2:] != appendix:
                word_token[i] = word_token[i]+appendix
        sentences.append({'tokens': word_token})
                           
    #addCharInformation(sentences)
    #addCasingInformation(sentences)
    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
    # :: Tag the input ::
    tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
    text +=' '+tags['textgrid'][0][-1]
    print('neuer Text: ', text)
    if tags['textgrid'][0][-1] == 'eos_n' or tags['textgrid'][0][-1] == 'eos_p' or  tags['textgrid'][0][-1] == '<eos>' or  tags['textgrid'][0][-1] == 'eos':
        break
    time.sleep(1)



# pos_neg = [0,0]
# for i in range(100):
#     if i < 50:
#         text = 'in_p'
#     else:
#         text = 'in_n'
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
#         #addCharInformation(sentences)
#         #addCasingInformation(sentences)
#         dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
#         # :: Tag the input ::
#         tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
#         text +=' '+tags['textgrid'][0][-1]
#         if tags['textgrid'][0][-1] == 'eos_n' or tags['textgrid'][0][-1] == 'eos_p' or  tags['textgrid'][0][-1] == '<eos>' or  tags['textgrid'][0][-1] == 'eos':
#             print('neuer Text: ', text)
#             if i < 50:
#                 pos_neg[1] += len(text.split(' '))
#             else:
#                 pos_neg[0] += len(text.split(' '))
#             break
#     #time.sleep(1)
# print(pos_neg, pos_neg[0]/50, pos_neg[1]/50)
