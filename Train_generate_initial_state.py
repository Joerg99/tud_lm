# This script trains the BiLSTM-CRF architecture for part-of-speech tagging using
# the universal dependency dataset (http://universaldependencies.org/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM_uni_initial_state import BiLSTM_uni

from util.preprocessing import perpareDataset, loadDatasetPickle



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

# datasets = {
#     'unidep_pos_german':                            #Name of the dataset
#         {'columns': {1:'tokens', 3:'POS'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
#          'label': 'POS',                     #Which column we like to predict
#          'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
#          'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
# }

my_datasets = {
    'textgrid':
        {'columns': {1:'tokens', 2:'POS'}, # 3:'sentiment'
         'label': 'POS',
         'evaluate': True,
         'commentSymbol': None}
}


# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
# embeddingsPath = 'komninos_english_embeddings.gz'
embeddingsPath = 'embedding_textgrid_300.bin'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, my_datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters

##### for perplexity add 'POS' to featureNames #######
params = {'featureNames': ['tokens', 'casing', 'POS'], 'classifier': ['Softmax'],'charEmbeddings': None, 'optimizer': 'adam', 'LSTM-Size': [100], 'dropout': (0.4)}

model = BiLSTM_uni(params)
model.setMappings(mappings, embeddings)
model.setDataset(my_datasets, data)
model.storeResults('results/textgrid_results.csv') #Path to store performance scores for dev / test
model.modelSavePath = "models/init_state/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Path to store models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
model.fit(epochs=101)

