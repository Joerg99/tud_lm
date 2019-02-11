"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging used for multi-task learning.

Author: Nils Reimers
License: Apache-2.0
"""

from __future__ import print_function
from util import BIOF1Validation

import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
# import matplotlib.pyplot as plt 
import math
import numpy as np
import sys
import gc
import time
import os
import random
import logging
import keras.losses
import tensorflow as tf
from .keraslayers.ChainCRF import ChainCRF
import sys
from sklearn.metrics import log_loss
from numpy import newaxis

class BiLSTM_uni:
    def __init__(self, params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.models = None
        self.modelSavePath = None
        self.resultsSavePath = None
        
        # Hyperparameters for the network
        defaultParams = {'dropout': (0.5,0.5), 'classifier': ['Softmax'], 'LSTM-Size': (100,100), 'customClassifier': {},
                         'optimizer': 'adam', 'temperature': 1,
                         'charEmbeddings': None, 'charEmbeddingsSize': 30, 'charFilterSize': 30, 'charFilterLength': 3, 'charLSTMSize': 25, 'maxCharLength': 25,
                         'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1,
                         'earlyStopping': 20, 'miniBatchSize': 32,
                         'featureNames': ['tokens', 'casing'], 'addFeatureDimensions': 10}
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams



    def setMappings(self, mappings, embeddings):
        self.embeddings = embeddings
        self.mappings = mappings

    def setDataset(self, datasets, data):
        self.datasets = datasets
        self.data = data

        # Create some helping variables
        self.mainModelName = None
        self.epoch = 0
        self.learning_rate_updates = {'sgd': {1: 0.3, 20: 0.1, 50: 0.05}}
        self.modelNames = list(self.datasets.keys())
        self.evaluateModelNames = []
        self.labelKeys = {}
        self.idx2Labels = {}
        self.trainMiniBatchRanges = None
        self.trainSentenceLengthRanges = None


        for modelName in self.modelNames:
            labelKey = self.datasets[modelName]['label']
            self.labelKeys[modelName] = labelKey
            self.idx2Labels[modelName] = {v: k for k, v in self.mappings[labelKey].items()}
            
            if self.datasets[modelName]['evaluate']:
                self.evaluateModelNames.append(modelName)
            
            logging.info("--- %s ---" % modelName)
            logging.info("%d train sentences" % len(self.data[modelName]['trainMatrix']))
            logging.info("%d dev sentences" % len(self.data[modelName]['devMatrix']))
            logging.info("%d test sentences" % len(self.data[modelName]['testMatrix']))
            
        
        if len(self.evaluateModelNames) == 1:
            self.mainModelName = self.evaluateModelNames[0]
             
        self.casing2Idx = self.mappings['casing']

        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            logging.info("Pad words to uniform length for characters embeddings")
            all_sentences = []
            for dataset in self.data.values():
                for data in [dataset['trainMatrix'], dataset['devMatrix'], dataset['testMatrix']]:
                    for sentence in data:
                        all_sentences.append(sentence)

            self.padCharacters(all_sentences)
            logging.info("Words padded to %d characters" % (self.maxCharLen))

        
    def buildModel(self):
        self.models = {}
        print('########################')
        print(self.embeddings.shape)
        print(self.embeddings.shape[0])
        print(self.embeddings.shape[1])
        print('########################')
        tokens_input = Input(shape=(None,), dtype='int32', name='words_input')
        tokens = Embedding(input_dim=self.embeddings.shape[0], output_dim=self.embeddings.shape[1], weights=[self.embeddings], trainable=False, name='word_embeddings')(tokens_input)
        
        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        for featureName in self.params['featureNames']:
            if featureName == 'tokens' or featureName == 'characters':
                continue
            
            if featureName == 'side_info':
#                 print('K K K K K K K')
                feature_input = Input(shape=(None, 1), dtype='float32', name=featureName+'_input')
                feature_embedding = TimeDistributed(Dense(1))(feature_input)
                #feature_embedding = Reshape((self.params['miniBatchSize'], 1, 1))(feature_input)
                 
                inputNodes.append(feature_input)
                mergeInputLayers.append(feature_embedding)
                 
            else:                
                feature_input = Input(shape=(None,), dtype='int32', name=featureName+'_input')
                feature_embedding = Embedding(input_dim=len(self.mappings[featureName]), output_dim=self.params['addFeatureDimensions'], name=featureName+'_emebddings')(feature_input)
    
                inputNodes.append(feature_input)
                mergeInputLayers.append(feature_embedding)


        # :: Character Embeddings ::
        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            charset = self.mappings['characters']
            charEmbeddingsSize = self.params['charEmbeddingsSize']
            maxCharLen = self.maxCharLen
            charEmbeddings = []
            for _ in charset:
                limit = math.sqrt(3.0 / charEmbeddingsSize)
                vector = np.random.uniform(-limit, limit, charEmbeddingsSize)
                charEmbeddings.append(vector)

            charEmbeddings[0] = np.zeros(charEmbeddingsSize)  # Zero padding
            charEmbeddings = np.asarray(charEmbeddings)

            chars_input = Input(shape=(None, maxCharLen), dtype='int32', name='char_input')
            mask_zero = (self.params['charEmbeddings'].lower()=='lstm') #Zero mask only works with LSTM
            chars = TimeDistributed(
                Embedding(input_dim=charEmbeddings.shape[0], output_dim=charEmbeddings.shape[1],
                          weights=[charEmbeddings],
                          trainable=True, mask_zero=mask_zero), name='char_emd')(chars_input)

            if self.params['charEmbeddings'].lower()=='lstm':  # Use LSTM for char embeddings from Lample et al., 2016
                charLSTMSize = self.params['charLSTMSize']
#                 chars = TimeDistributed(Bidirectional(LSTM(charLSTMSize, return_sequences=False)), name="char_lstm")(chars)
                chars = TimeDistributed(LSTM(charLSTMSize, return_sequences=False), name="char_lstm")(chars)
            else:  # Use CNNs for character embeddings from Ma and Hovy, 2016
                charFilterSize = self.params['charFilterSize']
                charFilterLength = self.params['charFilterLength']
                chars = TimeDistributed(Conv1D(charFilterSize, charFilterLength, padding='same'), name="char_cnn")(
                    chars)
                chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)

            self.params['featureNames'].append('characters')
            mergeInputLayers.append(chars)
            inputNodes.append(chars_input)
            
        # :: Task Identifier :: 
        if self.params['useTaskIdentifier']:
            self.addTaskIdentifier()
            
            taskID_input = Input(shape=(None,), dtype='int32', name='task_id_input')
            taskIDMatrix = np.identity(len(self.modelNames), dtype='float32')
            taskID_outputlayer = Embedding(input_dim=taskIDMatrix.shape[0], output_dim=taskIDMatrix.shape[1], weights=[taskIDMatrix], trainable=False, name='task_id_embedding')(taskID_input)
        
            mergeInputLayers.append(taskID_outputlayer)
            inputNodes.append(taskID_input)
            self.params['featureNames'].append('taskID')

        if len(mergeInputLayers) >= 2:
            merged_input = concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]
        
        
        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:      
            if isinstance(self.params['dropout'], (list, tuple)):  
#                 shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name='shared_varLSTM_'+str(cnt))(shared_layer)
                shared_layer = LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1])(shared_layer) #, name='shared_varLSTM_'+str(cnt))(shared_layer)
            else:
                """ Naive dropout """
#                 shared_layer = Bidirectional(LSTM(size, return_sequences=True), name='shared_LSTM_'+str(cnt))(shared_layer) 
                shared_layer = LSTM(size, return_sequences=True)(shared_layer)
                #shared_layer = concatenate([shared_layer, feature_input])
                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(Dropout(self.params['dropout']), name='shared_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(shared_layer)
            
            cnt += 1
            
            
        for modelName in self.modelNames:
            output = shared_layer
            
            modelClassifier = self.params['customClassifier'][modelName] if modelName in self.params['customClassifier'] else self.params['classifier']

            if not isinstance(modelClassifier, (tuple, list)):
                modelClassifier = [modelClassifier]
            cnt = 1
            for classifier in modelClassifier:
                n_class_labels = len(self.mappings[self.labelKeys[modelName]])
                print('n_class_labels', n_class_labels)
                if classifier == 'Softmax':
                    temp = self.params['temperature']
                    logits_temperature = Lambda(lambda x : x / temp)(output)
                    
                    output = TimeDistributed(Dense(n_class_labels, activation='softmax'), name=modelName+'_softmax')(logits_temperature) # without temperature input is (output)
                    
                    def my_loss(y_true, y_pred): # my_loss = perplexity
                        perplexity = K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))
                        print(perplexity)
                        return perplexity
                    
                    lossFct = 'sparse_categorical_crossentropy' #my_loss
                elif classifier == 'CRF':
                    output = TimeDistributed(Dense(n_class_labels, activation=None),
                                             name=modelName + '_hidden_lin_layer')(output)
                    crf = ChainCRF(name=modelName+'_crf')
                    output = crf(output)
                    lossFct = crf.sparse_loss
                elif isinstance(classifier, (list, tuple)) and classifier[0] == 'LSTM':
                            
                    size = classifier[1]
                    if isinstance(self.params['dropout'], (list, tuple)): 
#                         output = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name=modelName+'_varLSTM_'+str(cnt))(output)
                        output = LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1])(output)#, name=modelName+'_varLSTM_'+str(cnt))(output)
                    else:
                        """ Naive dropout """ 
#                         output = Bidirectional(LSTM(size, return_sequences=True), name=modelName+'_LSTM_'+str(cnt))(output) 
                        output = LSTM(size, return_sequences=True)(output)#, name=modelName+'_LSTM_'+str(cnt))(output) 
                        if self.params['dropout'] > 0.0:
                            output = TimeDistributed(Dropout(self.params['dropout']), name=modelName+'_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(output)                    
                else:
                    assert(False) #Wrong classifier
                    
                cnt += 1
                
            # :: Parameters for the optimizer ::
            optimizerParams = {}
            if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
                optimizerParams['clipnorm'] = self.params['clipnorm']
            
            if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
                optimizerParams['clipvalue'] = self.params['clipvalue']
            
            if self.params['optimizer'].lower() == 'adam':
                opt = Adam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'nadam':
                opt = Nadam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'rmsprop': 
                opt = RMSprop(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adadelta':
                opt = Adadelta(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adagrad':
                opt = Adagrad(**optimizerParams)
            elif self.params['optimizer'].lower() == 'sgd':
                opt = SGD(lr=0.1, **optimizerParams)
            
            
            model = Model(inputs=inputNodes, outputs=[output])
            model.compile(loss=lossFct, optimizer=opt)
            
            model.summary(line_length=125)
            #logging.info(model.get_config())
            #logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))
            
            self.models[modelName] = model
        


    def trainModel(self):
        self.epoch += 1
        
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:       
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            for modelName in self.modelNames:            
                K.set_value(self.models[modelName].optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch]) 
                
        loss_all_batches_train = []
        loss_all_batches_test = []
        loss_all_batches_dev = []
        
        #train
        for batch in self.minibatch_iterate_dataset('trainMatrix'):
            for modelName in self.modelNames:         
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                nnInput[1] = nnInput[1][:,:,newaxis]
#                 print('training input nnInput side info: ', np.shape(nnInput[1]) ,type(nnInput[1]))
                self.models[modelName].train_on_batch(nnInput, nnLabels)  

        #evaluate
        for batch in self.minibatch_iterate_dataset('trainMatrix'):
            for modelName in self.modelNames:         
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                nnInput[1] = nnInput[1][:,:,newaxis]
                loss_all_batches_train.append(np.exp(self.models[modelName].test_on_batch(nnInput, nnLabels)))  #or .evaluate(....)
        print('train loss in epoch:', self.epoch, np.mean(loss_all_batches_train))
        
        for batch in self.minibatch_iterate_dataset('testMatrix'):
            for modelName in self.modelNames:         
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                nnInput[1] = nnInput[1][:,:,newaxis]
                loss_all_batches_test.append(np.exp(self.models[modelName].test_on_batch(nnInput, nnLabels)))  #or .evaluate(....)
        print('test loss in epoch:', self.epoch, np.mean(loss_all_batches_test))
        
        for batch in self.minibatch_iterate_dataset('devMatrix'):
            for modelName in self.modelNames:         
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                nnInput[1] = nnInput[1][:,:,newaxis]
                loss_all_batches_dev.append(np.exp(self.models[modelName].test_on_batch(nnInput, nnLabels)))  #or .evaluate(....)
        print('dev loss in epoch:', self.epoch, np.mean(loss_all_batches_dev))
        
        return np.mean(loss_all_batches_train), np.mean(loss_all_batches_test), np.mean(loss_all_batches_dev)


    def minibatch_iterate_dataset(self, matrixName,  modelNames = None):
        """ Create based on sentence length mini-batches with approx. the same size. Sentences and 
        mini-batch chunks are shuffled and used to the train the model """
        self.trainSentenceLengthRanges = None
        if self.trainSentenceLengthRanges == None:
            """ Create mini batch ranges """
            self.trainSentenceLengthRanges = {}
            self.trainMiniBatchRanges = {}            
            for modelName in self.modelNames:
                trainData = self.data[modelName][matrixName]
                trainData.sort(key=lambda x:len(x['tokens'])) #Sort train matrix by sentence length
                trainRanges = []
                oldSentLength = len(trainData[0]['tokens'])            
                idxStart = 0
                
                #Find start and end of ranges with sentences with same length
                for idx in range(len(trainData)):
                    sentLength = len(trainData[idx]['tokens'])
                    
                    if sentLength != oldSentLength:
                        trainRanges.append((idxStart, idx))
                        idxStart = idx
                    
                    oldSentLength = sentLength
                
                #Add last sentence
                trainRanges.append((idxStart, len(trainData)))
                
                
                #Break up ranges into smaller mini batch sizes
                miniBatchRanges = []
                for batchRange in trainRanges:
                    rangeLen = batchRange[1]-batchRange[0]

                    bins = int(math.ceil(rangeLen/float(self.params['miniBatchSize'])))
                    binSize = int(math.ceil(rangeLen / float(bins)))
                    
                    for binNr in range(bins):
                        startIdx = binNr*binSize+batchRange[0]
                        endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                        miniBatchRanges.append((startIdx, endIdx))
                      
                self.trainSentenceLengthRanges[modelName] = trainRanges
                self.trainMiniBatchRanges[modelName] = miniBatchRanges
                
        if modelNames == None:
            modelNames = self.modelNames
            
        #Shuffle training data
        for modelName in modelNames:      
            #1. Shuffle sentences that have the same length
            x = self.data[modelName][matrixName]
            for dataRange in self.trainSentenceLengthRanges[modelName]:
                for i in reversed(range(dataRange[0]+1, dataRange[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    j = random.randint(dataRange[0], i)
                    x[i], x[j] = x[j], x[i]
                
            #2. Shuffle the order of the mini batch ranges       
            random.shuffle(self.trainMiniBatchRanges[modelName])
     
        
        #Iterate over the mini batch ranges
        if self.mainModelName != None:
            rangeLength = len(self.trainMiniBatchRanges[self.mainModelName])
        else:
            rangeLength = min([len(self.trainMiniBatchRanges[modelName]) for modelName in modelNames])

        
        batches = {}
        for idx in range(rangeLength):
            batches.clear()
            
            for modelName in modelNames:   
                trainMatrix = self.data[modelName][matrixName]
                dataRange = self.trainMiniBatchRanges[modelName][idx % len(self.trainMiniBatchRanges[modelName])] 
                labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])
                labels = np.expand_dims(labels, -1)
                
                
                batches[modelName] = [labels]
                
                for featureName in self.params['featureNames']:
                    inputData = np.asarray([trainMatrix[idx][featureName] for idx in range(dataRange[0], dataRange[1])])
                    batches[modelName].append(inputData)
            
            yield batches   
            

        
    def storeResults(self, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.resultsSavePath = open(resultsFilepath, 'w')
        else:
            self.resultsSavePath = None
        
    def fit(self, epochs):
        if self.models is None:
            self.buildModel()

        total_train_time = 0
        max_dev_score = {modelName:0 for modelName in self.models.keys()}
        max_test_score = {modelName:0 for modelName in self.models.keys()}
        no_improvement_since = 0
        
        train_scores_plotting = []
        test_scores_plotting = []
        dev_scores_plotting = []
        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("\n--------- Epoch %d -----------" % (epoch+1))
            
            start_time = time.time() 
            train_score, test_score, dev_score = self.trainModel()
            
            train_scores_plotting.append(train_score)
            test_scores_plotting.append(test_score)
            dev_scores_plotting.append(dev_score)
            
            with open('models/test/plot_'+str(epoch)+'_'+self.evaluateModelNames[0], 'w') as file:
                for i in range(len(train_scores_plotting)):
                    file.write(str(train_scores_plotting[i])+' ')
                file.write('\n')
                for j in range(len(dev_scores_plotting)):
                    file.write(str(dev_scores_plotting[j])+' ')
                file.write('\n')
                for k in range(len(test_scores_plotting)):
                    file.write(str(test_scores_plotting[k])+' ')
                file.write('\n')
            
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            
            
            start_time = time.time() 
            for modelName in self.evaluateModelNames:
                logging.info("-- %s --" % (modelName))
#                 train_score, dev_score, test_score = 0, 0, 0 #self.computeScore(modelName, self.data[modelName]['trainMatrix_eval'], self.data[modelName]['devMatrix'], self.data[modelName]['testMatrix'])
#                 train_scores_plotting.append(train_score)
#                 test_scores_plotting.append(test_score)
#                 dev_scores_plotting.append(dev_score)
#                 if dev_score > max_dev_score[modelName]:
#                     max_dev_score[modelName] = dev_score
#                     max_test_score[modelName] = test_score
#                     no_improvement_since = 0
# 
#                 else:
#                     no_improvement_since += 1
                    
                #Save the model alle 4 Epochen
                if self.modelSavePath != None and epoch % 1 == 0:
                    self.saveModel(modelName, epoch, dev_score, test_score)
#                     plt.plot(train_scores_plotting, label='train')
#                     plt.plot(test_scores_plotting, label='test')
#                     plt.plot(dev_scores_plotting, label='dev')
#                     plt.legend()
#                     plt.savefig('plot_'+str(epoch)+'_'+modelName)
#                     plt.clf()

                        
                        
                if self.resultsSavePath != None:
                    self.resultsSavePath.write("\t".join(map(str, [epoch + 1, modelName, dev_score, test_score, max_dev_score[modelName], max_test_score[modelName]])))
                    self.resultsSavePath.write("\n")
                    self.resultsSavePath.flush()
                
#                 logging.info("\nScores from epoch with best Perplexity on Dev:\n  Dev-Perplexity: %.4f\n  Test-Score %.4f" % (max_dev_score[modelName], max_test_score[modelName]))
#                 logging.info("")
                
            logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
#             if self.params['earlyStopping']  > 0 and no_improvement_since >= self.params['earlyStopping']:
#                 logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
#                 break
            
            
    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)): # index = Satz
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)
        
        return sentenceLengths
            

    def tagSentences_generate(self, sentences, predictions_sampled, generation_mode):
        # Pad characters
        if 'characters' in self.params['featureNames']:
            self.padCharacters(sentences)
        
#         print('sentences', sentences)
        labels = {}
        for modelName, model in self.models.items(): # modelname = textgrid, immer nur ein model drin
            paddedPredLabels = self.predictLabels_generate(model, sentences, predictions_sampled, generation_mode)
            
            predLabels = []
            for idx in range(len(sentences)): ###### immer nur ein satz
                unpaddedPredLabels = []
                for tokenIdx in range(len(sentences[idx]['tokens'])):
                    if sentences[idx]['tokens'][tokenIdx] != 0:  # Skip padding tokens
                        unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])


                predLabels.append(unpaddedPredLabels) # list mit indexen, 
            
            # CONVERT PREDLABEL INDEX TO LABEL
            idx2Label = self.idx2Labels[modelName]
            labels[modelName] = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]
            #print('labels ', labels)
        return labels
    
    def predictLabels_generate(self, model, sentences, predictions_sampled, generation_mode):
        predLabels = [None]*len(sentences)
        sentenceLengths = self.getSentenceLengths(sentences) # sentenceLengths for speed....
                
        for indices in sentenceLengths.values():   
            nnInput = []                  
            for featureName in self.params['featureNames']: # tokens and pos
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
#                 print('sentences[idx][featureName]', sentences[0][featureName])
                
                nnInput.append(inputData)
            
#             print('nnInput:', nnInput)
            predictions = model.predict(nnInput, verbose=False)
            
            #generation_mode = 'sample'   # 'max' oder 'sample'
            if generation_mode == 'sample':
                ########### for sampling
                predicted = np.random.choice(len(predictions[0][-1]), p=predictions[0][-1])
                predictions_sampled[0].append(predicted)
#                 print('neu: ', predictions_sampled)
                predIdx = 0
                for idx in indices:
                    predLabels[idx] = predictions_sampled[predIdx]    
                    predIdx += 1   
            
            if generation_mode == 'max':
                ########### for argmax
                predictions = predictions.argmax(axis=-1)
                predIdx = 0
                for idx in indices:
                    predLabels[idx] = predictions[predIdx]    
                    predIdx += 1 
                
        return predLabels
    
    def tagSentences(self, sentences):
        # Pad characters
        if 'characters' in self.params['featureNames']:
            self.padCharacters(sentences)

        labels = {}
        for modelName, model in self.models.items():
            paddedPredLabels = self.predictLabels(model, sentences)
            predLabels = []
            for idx in range(len(sentences)):
                unpaddedPredLabels = []
                for tokenIdx in range(len(sentences[idx]['tokens'])):
                    if sentences[idx]['tokens'][tokenIdx] != 0:  # Skip padding tokens
                        unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

                predLabels.append(unpaddedPredLabels)

            idx2Label = self.idx2Labels[modelName]
            labels[modelName] = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]

        return labels
    
    def predictLabels(self, model, sentences):
        predLabels = [None]*len(sentences)
        sentenceLengths = self.getSentenceLengths(sentences)
        for indices in sentenceLengths.values():   
            nnInput = []                  
            for featureName in self.params['featureNames']:
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
                nnInput.append(inputData)
            predictions = model.predict(nnInput, verbose=False)
            predictions= predictions.argmax(axis=-1) #argmax returns index, in [[i1, i2, ...]]          
            
            
            predIdx = 0
            for idx in indices:
                predLabels[idx] = predictions[predIdx]    
                predIdx += 1   
        
        return predLabels
   
    def computeScore(self, modelName, trainMatrix, devMatrix, testMatrix):
        print('Perplexity Evaluation')
        return self.computePerplexityScores(modelName,trainMatrix, devMatrix, testMatrix)   
#         print('Accuracy Evaluation')
#         return self.computeAccScores(modelName,trainMatrix, devMatrix, testMatrix)   

    def computeAccScores(self, modelName, trainMatrix, devMatrix, testMatrix):
        train_acc = self.computeAcc(modelName, trainMatrix)
        dev_acc = self.computeAcc(modelName, devMatrix)
        test_acc = self.computeAcc(modelName, testMatrix)
        logging.info("Train-Data: Accuracy: %.4f" % (train_acc))
        logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
        logging.info("Test-Data: Accuracy: %.4f" % (test_acc))
        return train_acc, dev_acc, test_acc 
    
    def computeAcc(self, modelName, sentences):
        correctLabels = [sentences[idx][self.labelKeys[modelName]] for idx in range(len(sentences))]
        print('correctLabels', len(correctLabels), correctLabels[0:5])
        predLabels = self.predictLabels(self.models[modelName], sentences) 
        print('predLabels', predLabels[:5])
        numLabels = 0
        numCorrLabels = 0
        for sentenceId in range(len(correctLabels)):
            for tokenId in range(len(correctLabels[sentenceId])):
                numLabels += 1
                if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
                    numCorrLabels += 1
        return numCorrLabels/float(numLabels)
    
    def computePerplexityScores(self, modelName, trainMatrix, devMatrix, testMatrix):
        train_acc = self.compute_perplexity(modelName, trainMatrix) # compute_perplexity
        dev_acc = self.compute_perplexity(modelName, devMatrix) # compute_perplexity
        test_acc = self.compute_perplexity(modelName, testMatrix) # compute perplexity
        logging.info("Train-Data: Perplexity: %.4f" % (train_acc))
        logging.info("Dev-Data: Perplexity: %.4f" % (dev_acc))
        logging.info("Test-Data: Perplexity: %.4f" % (test_acc))
        return train_acc, dev_acc, test_acc   
    
    def compute_perplexity(self, modelName, sentences):
        all_predictions, all_labels = self.predictLabels_for_perplexity_evaluation(self.models[modelName], sentences)
        
        return self.numpy_perplexity(all_predictions, all_labels, modelName,  sentences)
        
######### OPTION 1 model.evaluate should be the best option... but not working :( 
#         print('k')
#         model = self.loadModel('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/test/textgrid_1072.2056_1595.8857_3.h5', 1)
#         print(type(model))
#         print(model.models)
#         model = model.models['textgrid']
#         print('kk')
#         model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        
#         
#         
#         # idea is to run following code on the model compiled above... but it doesnt work
#         for i in range(len(all_labels)):
#             all_labels[i] = all_labels[i][:,:, np.newaxis]
#         perplexity = []
#           
#         for i in range(len(all_labels)): #range(10,15): 
#             #start = time.time()
#             xentropy = K.sparse_categorical_crossentropy(tf.keras.backend.cast(all_labels[i], dtype='float32'), tf.keras.backend.cast(all_predictions[i], dtype='float32')) #tf.convert_to_tensor(all_labels[i]), tf.convert_to_tensor(all_predictions[i]))
#             perplexity.append(K.eval(K.pow(2.0, xentropy)))
#             #print('time for one set of sentences. ', time.time()- start)
#         #average for each datapoint
#         for i in range(len(perplexity)):
#             perplexity[i] = np.average(perplexity[i], axis=1)
#             perplexity[i] = np.average(perplexity[i])
#         
#         p = np.mean(perplexity)
#         model = None
#         gc.collect()
# 
#         print(p)
#         return p
#########


######### OPTION 3 Working but memory problem
#         calculate perplexity for each sentence length and each datapoint and append to list
#         # add an axis to fit tensor shape used for Option 3
#         for i in range(len(all_labels)):
#             all_labels[i] = all_labels[i][:,:, np.newaxis]
#         perplexity = []
#           
#         for i in range(len(all_labels)): #range(10,15): 
#             xentropy = K.sparse_categorical_crossentropy(tf.keras.backend.cast(all_labels[i], dtype='float32'), tf.keras.backend.cast(all_predictions[i], dtype='float32')) #tf.convert_to_tensor(all_labels[i]), tf.convert_to_tensor(all_predictions[i]))
#             perplexity.append(K.eval(K.pow(2.0, xentropy)))
#         #average for each datapoint
#         for i in range(len(perplexity)):
#             perplexity[i] = np.average(perplexity[i], axis=1)
#             perplexity[i] = np.average(perplexity[i])
#           
#         
#         return np.mean(perplexity)
     
    
    def numpy_perplexity(self, all_predictions, all_labels, modelName, sentences):
        #correctLabels = [sentences[idx][self.labelKeys[modelName]] for idx in range(len(sentences))]
        
#         print('preds ', np.shape(all_predictions), ' labels ', np.shape(all_labels))
#         print('preds ', np.shape(all_predictions[4]), ' labels ', np.shape(all_labels[4]))
        #print(all_labels[4])
        oh_labels = np.zeros_like(all_predictions)
        
        
        all_labels_oh = []
        # k = Matrix in Liste
        # i = 
        for k in range(len(all_labels)):
            
            oh_labels = np.zeros_like(all_predictions[k])
            #print(np.shape(oh_labels))
            
            for i in range(len(all_labels[k][0])): # i waehlt einen Satz aus 
                for j in range(len(all_labels[k][0][i])):
                    oh_labels[i][j][all_labels[k][0][i][j]] = 1
            
            all_labels_oh.append(oh_labels)
        
            
        def cross_entropy(predictions, targets, epsilon=1e-12):
            """
            Computes cross entropy between targets (encoded as one-hot vectors)
            and predictions. 
            Input: predictions (N, k) ndarray
                   targets (N, k) ndarray        
            Returns: scalar
            """
            predictions = np.clip(predictions, epsilon, 1. - epsilon)
            N = predictions.shape[0]
            ce = (-np.sum(targets*np.log(predictions+1e-9))/N) / predictions.shape[1]
            return ce
        
        perplexities = []
        for i in range(len(all_predictions)):
            perplexities.append(np.exp(cross_entropy(all_predictions[i], all_labels_oh[i])))
        #print('manually calculated perplexity: ', np.mean(perplexities))
        return np.mean(perplexities)

    def predictLabels_for_perplexity_evaluation(self, model, sentences):
        
        sentenceLengths = self.getSentenceLengths(sentences)
        all_preds_softmax = []
        all_preds_softmax_labels = []
        for indices in sentenceLengths.values():   
            nnInput = []
            nnInput_labels = []                  
            for featureName in self.params['featureNames']:
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
                inputData_labels = np.asarray([sentences[idx]['POS'] for idx in indices])
                nnInput.append(inputData)
                nnInput_labels.append(inputData_labels)
                
            predictions_softmax = model.predict(nnInput, verbose=False)
            all_preds_softmax.append(predictions_softmax)
            all_preds_softmax_labels.append(nnInput_labels)
        return all_preds_softmax, all_preds_softmax_labels
    
    def computeF1Scores(self, modelName, devMatrix, testMatrix):
        #train_pre, train_rec, train_f1 = self.computeF1(modelName, self.datasets[modelName]['trainMatrix'])
        #print "Train-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (train_pre, train_rec, train_f1)
        
        dev_pre, dev_rec, dev_f1 = self.computeF1(modelName, devMatrix)
        logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (dev_pre, dev_rec, dev_f1))
        
        test_pre, test_rec, test_f1 = self.computeF1(modelName, testMatrix)
        logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))
        
        return dev_f1, test_f1
    
    def computeF1(self, modelName, sentences):
        labelKey = self.labelKeys[modelName]
        model = self.models[modelName]
        idx2Label = self.idx2Labels[modelName]
        
        correctLabels = [sentences[idx][labelKey] for idx in range(len(sentences))]
        predLabels = self.predictLabels(model, sentences) 

        labelKey = self.labelKeys[modelName]
        encodingScheme = labelKey[labelKey.index('_')+1:]
        
        pre, rec, f1 = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'O', encodingScheme)
        pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'B', encodingScheme)
        
        if f1_b > f1:
            logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
            pre, rec, f1 = pre_b, rec_b, f1_b
        
        return pre, rec, f1
    
    def padCharacters(self, sentences):
        """ Pads the character representations of the words to the longest word in the dataset """
        #Find the longest word in the dataset
        maxCharLen = self.params['maxCharLength']
        if maxCharLen <= 0:
            for sentence in sentences:
                for token in sentence['characters']:
                    maxCharLen = max(maxCharLen, len(token))
          

        for sentenceIdx in range(len(sentences)):
            for tokenIdx in range(len(sentences[sentenceIdx]['characters'])):
                token = sentences[sentenceIdx]['characters'][tokenIdx]

                if len(token) < maxCharLen: #Token shorter than maxCharLen -> pad token
                    sentences[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0,maxCharLen-len(token)), 'constant')
                else: #Token longer than maxCharLen -> truncate token
                    sentences[sentenceIdx]['characters'][tokenIdx] = token[0:maxCharLen]
    
        self.maxCharLen = maxCharLen
        
    def addTaskIdentifier(self):
        """ Adds an identifier to every token, which identifies the task the token stems from """
        taskID = 0
        for modelName in self.modelNames:
            dataset = self.data[modelName]
            for dataName in ['trainMatrix', 'devMatrix', 'testMatrix']:            
                for sentenceIdx in range(len(dataset[dataName])):
                    dataset[dataName][sentenceIdx]['taskID'] = [taskID] * len(dataset[dataName][sentenceIdx]['tokens'])
            
            taskID += 1


    def saveModel(self, modelName, epoch, dev_score, test_score):
        import json
        import h5py

        if self.modelSavePath == None:
            raise ValueError('modelSavePath not specified.')

        savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch+1)).replace("[ModelName]", modelName)

        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(savePath):
            logging.info("Model "+savePath+" already exists. Model will be overwritten")

        self.models[modelName].save(savePath, True)

        with h5py.File(savePath, 'a') as h5file:
            h5file.attrs['mappings'] = json.dumps(self.mappings)
            h5file.attrs['params'] = json.dumps(self.params)
            h5file.attrs['modelName'] = modelName
            h5file.attrs['labelKey'] = self.datasets[modelName]['label']




    @staticmethod
    def loadModel(modelPath, temperature):
        import h5py
        import json
        from .keraslayers.ChainCRF import create_custom_objects

        def my_loss(y_true, y_pred):
            perplexity = K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))
            #print(perplexity)
            return perplexity

        model = keras.models.load_model(modelPath, custom_objects={'my_loss': my_loss})#create_custom_objects())

        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            params = json.loads(f.attrs['params'])
            modelName = f.attrs['modelName']
            labelKey = f.attrs['labelKey']
        
        params['temperature'] = temperature
        
        bilstm = BiLSTM_uni(params)
        bilstm.setMappings(mappings, None)
        bilstm.models = {modelName: model}
        bilstm.labelKeys = {modelName: labelKey}
        bilstm.idx2Labels = {}
        bilstm.idx2Labels[modelName] = {v: k for k, v in bilstm.mappings[labelKey].items()}
        return bilstm
