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


# if len(sys.argv) < 3:
#     print("Usage: python RunModel.py modelPath inputPath")
#     exit()

#inputPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/input.txt'

# :: Read input ::
# with open(inputPath, 'r') as f:
#     text = f.read()

# :: Load the model ::
modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/test/textgrid_9.6219_309.9963_21.h5' # with perplexity and POS label DOESNT RUN
#modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/stanza_perpLoss_400_drop05_train/textgrid_0.1249_0.1333_31.h5' # with ACC and no POS  RUNS!
lstmModel = BiLSTM_uni.loadModel(modelPath)

text = 'sos_n' #'startseq'
generation_mode = 'sample' # 'max' or 'sample'

predictions_sampled = [[]]
# :: Prepare the input ::
while True:
    sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
    addCharInformation(sentences)
    addCasingInformation(sentences)
    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
    # :: Tag the input ::
    tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
    print('returned tags: ', tags)
    text +=' '+tags['textgrid'][0][-1]
    print('neuer Text: ', text)
    if tags['textgrid'][0][-1] == '<eos>':
        break
    time.sleep(1)
