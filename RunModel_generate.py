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
modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/250k_pos_neg_lower_64_04/textgrid_914.5579_753.4499_1.h5' # with perplexity and POS label DOESNT RUN
lstmModel = BiLSTM_uni.loadModel(modelPath)

text = 'gott_n'
generation_mode = 'sample' # 'max' or 'sample'

predictions_sampled = [[]]
# :: Prepare the input ::
while True:
    sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
    #print(sentences)
    addCharInformation(sentences)
    addCasingInformation(sentences)
    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
    # :: Tag the input ::
    tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
    #print('returned tags: ', tags)
    text +=' '+tags['textgrid'][0][-1]
    print('neuer Text: ', text)
    if tags['textgrid'][0][-1] == 'eos_n' or tags['textgrid'][0][-1] == 'eos_p' or  tags['textgrid'][0][-1] == '<eos>' :
        break
    time.sleep(1)

# pos_neg = [0,0]
# for i in range(100):
#     if i < 50:
#         text = 'die_p'
#     else:
#         text = 'die_n'
#     generation_mode = 'sample' # 'max' or 'sample'
#     
#     predictions_sampled = [[]]
#     # :: Prepare the input ::
#     while True:
#         sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
#         #print(sentences)
#         addCharInformation(sentences)
#         addCasingInformation(sentences)
#         dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
#         # :: Tag the input ::
#         tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
#         print('returned tags: ', tags)
#         text +=' '+tags['textgrid'][0][-1]
#         #print('neuer Text: ', text)
#         if tags['textgrid'][0][-1] == 'eos_n' or tags['textgrid'][0][-1] == 'eos_p' or  tags['textgrid'][0][-1] == '<eos>' or len(text.split(' ')) == 30:
#             if tags['textgrid'][0][0][-2:] == '_n':
#                 pos_neg[1] += len(text.split(' '))
#             else:
#                 pos_neg[0] += len(text.split(' '))
#             break
#         #time.sleep(1)
# print(pos_neg, pos_neg[0]/50, pos_neg[1]/50)
