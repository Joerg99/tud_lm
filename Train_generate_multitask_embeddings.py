# This script trains the BiLSTM-CRF architecture for part-of-speech tagging using
# the universal dependency dataset (http://universaldependencies.org/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys



########## EMBEDDING #############
from neuralnets.BiLSTM_uni import BiLSTM_uni # side info as embedding
from util.preprocessing_side_info_embedding import perpareDataset, loadDatasetPickle

########## REAL VALUE #############
# from neuralnets.BiLSTM_uni_real_value import BiLSTM_uni
# from util.preprocessing_side_info_real_value import perpareDataset, loadDatasetPickle


import keras
import tensorflow as tf

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=sess_config))



# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################


# my_datasets = {
#     'chicago_mtl1':
#         {'columns': {1:'tokens', 2:'POS', 3:'side_info'},  #3: allit density , 4: rhyme density , 5: allit density_norm, 6: rhyme density_norm
#          'label': 'POS',
#          'evaluate': True,
#          'commentSymbol': None},
#     'chicago_mtl2':
#         {'columns': {1:'tokens', 3:'BIO', 4:'side_info'},  #3: allit density , 4: rhyme density , 5: allit density_norm, 6: rhyme density_norm
#          'label': 'BIO',
#          'evaluate': True,
#          'commentSymbol': None}
# }
my_datasets = {
    'chicago_mtl1': # lm
        {'columns': {1:'tokens', 2:'POS', 3:'side_info_allit', 5:'side_info_rhyme'},  #3: allit density , 4: rhyme density , 5: allit density_norm, 6: rhyme density_norm
         'label': 'POS',
         'evaluate': True,
         'commentSymbol': None},
    'chicago_mtl2': # alliteration
        {'columns': {1:'tokens', 4:'BIO', 3:'side_info_allit', 5:'side_info_rhyme'},  #3: allit density , 4: rhyme density , 5: allit density_norm, 6: rhyme density_norm
         'label': 'BIO',
         'evaluate': True,
         'commentSymbol': None},
    'chicago_mtl3': # reim
        {'columns': {1:'tokens', 6:'RHYME', 3:'side_info_allit', 5:'side_info_rhyme'},  #3: allit density , 4: rhyme density , 5: allit density_norm, 6: rhyme density_norm
         'label': 'RHYME',
         'evaluate': True,
         'commentSymbol': None}
}

print(my_datasets.keys())

# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
# embeddingsPath = 'embedding_textgrid_300_lower.bin' #_pos_neg.bin'
embeddingsPath = 'embedding_chicago_300_lower.bin'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, my_datasets) # Set reducePretrainedEmbeddings = True and padOneTokenSentence = False

######################################################
#
# The training of the network starts here
#
######################################################
#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters

##### for perplexity add 'POS' to featureNames #######

params = {'featureNames': ['tokens', 'side_info_allit', 'side_info_rhyme'], 'classifier': ['Softmax'], 'optimizer': 'adam', 'LSTM-Size': [64], 'dropout': (0.2), 'charEmbeddings': 'LSTM'} #,'charEmbeddings': 'LSTM'}
# params = {'featureNames': ['tokens', 'side_info'], 'classifier': ['Softmax'], 'optimizer': 'adam', 'LSTM-Size': [64], 'dropout': (0.2), 'charEmbeddings': 'LSTM'} #,'charEmbeddings': 'LSTM'}

model = BiLSTM_uni(params)
model.setMappings(mappings, embeddings)
model.setDataset(my_datasets, data)
model.storeResults('results/textgrid_results.csv') #Path to store performance scores for dev / test
model.modelSavePath = "models/chicago/mtl/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Path to store models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
model.fit(epochs=101)

