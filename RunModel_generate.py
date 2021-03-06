#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
from __future__ import print_function
import nltk
from util.preprocessing_normal import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM_uni_simple_generation import BiLSTM_uni
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
modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/gutentag/unconditioned/gutentag_248.2973_249.1088_50.h5' # with perplexity and POS label DOESNT RUN
# modelPath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/chicago/unconditioned/chicago_35.h5' # with perplexity and POS label DOESNT RUN

modelname = 'gutentag'

temperature = 0
lstmModel = BiLSTM_uni.loadModel(modelPath, temperature)
for temp in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]:
    quatrains = []
    i= 0
    while i < 10:
        text = 'sos'
        generation_mode = 'sample' # 'max' or 'sample'
        predictions_sampled = [[]]
        while True:
            #sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
            sentences = []
            for sent in nltk.sent_tokenize(text):
                word_token = nltk.word_tokenize(sent)
                sentences.append({'tokens': word_token})
                                    
            addCharInformation(sentences)
            #addCasingInformation(sentences)
            dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
            # :: Tag the input ::
            tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode, temp)
            text +=' '+tags[modelname][0][-1]
            if tags[modelname][0][-1]in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags[modelname][0]) == 100:
                print('neuer Text: ', text)
                quatrains.append(text)
                i += 1
                break

    with open('evaluation_files/'+modelname+'/temperature/'+modelname+'_'+str(temp), 'w') as file:
        for quatrain in quatrains:
            file.write('%s \n' %quatrain)
    print('wrote a file')    


def run_with_pos_neg_embeddings():
    
    # start = 'in'
    # appendix = '_p'
    # text = start+appendix
    # generation_mode = 'sample' # 'max' or 'sample'
    # 
    # predictions_sampled = [[]]
    # while True:
    #     #sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
    #     sentences = []
    #     for sent in nltk.sent_tokenize(text):
    #         word_token = nltk.word_tokenize(sent)
    #         for i in range(len(word_token)):
    #             if word_token[i][-2:] != appendix:
    #                 word_token[i] = word_token[i]+appendix
    #         sentences.append({'tokens': word_token})
    #                            
    #     #addCharInformation(sentences)
    #     #addCasingInformation(sentences)
    #     dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
    #     # :: Tag the input ::
    #     tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
    #     text +=' '+tags['textgrid'][0][-1]
    #     if tags['textgrid'][0][-1]in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags['textgrid'][0]) == 30:
    #         print('neuer Text: ', text)
    #         break
    #     #time.sleep(1)
    ####
    #### 
    ####
    pos_neg = [0,0]
    for j in range(100):
        start = 'sos'
        if j < 50:
            appendix = '_n'
            text = start+appendix
        else:
            appendix = '_p'
            text = start+appendix
        generation_mode = 'sample' # 'max' or 'sample'
        predictions_sampled = [[]]
        # :: Prepare the input ::
        while True:
            sentences = []
            for sent in nltk.sent_tokenize(text):
                word_token = nltk.word_tokenize(sent)
                for i in range(len(word_token)):
                    if word_token[i][-2:] != appendix:
                        word_token[i] = word_token[i]+appendix
                sentences.append({'tokens': word_token})
                                   
    #         addCharInformation(sentences)
    #         addCasingInformation(sentences)
            dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
            # :: Tag the input ::
            tags = lstmModel.tagSentences_generate(dataMatrix, predictions_sampled, generation_mode)
            text +=' '+tags['textgrid'][0][-1]
            if tags['textgrid'][0][-1]in ['eos_n', 'eos_p', '<eos>', 'eos'] or len(tags['textgrid'][0]) == 300:
                print('neuer Text: ', text)
                if appendix == '_p':
                    pos_neg[0] += len(text.split(' '))
                else:
                    pos_neg[1] += len(text.split(' '))
                break
        #time.sleep(1)
    print(pos_neg, pos_neg[0]/50, pos_neg[1]/50)
