import torch
from torch.utils.data import Dataset
import utils
import os
# import ipdb
import glob
import time
import random
import sys
from PIL import Image
import numpy as np

import glob
# load image and convert to and from NumPy array
from numpy import asarray
# End Imports
# ----------------------------------------------------

# File Global Variables and datapath setting
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path,"Data")

data_path_train = os.path.join(dir_path, "Data", "fruits-360_dataset", "fruits-360", "Training")
data_path_test = os.path.join(dir_path, "Data", "fruits-360_dataset", "fruits-360", "Test")
sys.path.insert(0, data_path)


# =================================================================================================================
# Class Declaration
# =================================================================================================================
class Tweets(Dataset):
    
    def __init__(self, randInputOrder = None, skipPerCent = None,  dataDir=None, csv_file=None, setSize = 0.8):
        
        allData, allTargets =  self._load_data()
        setLowerLimit = 0 if skipPerCent is None else int(allData.shape[0]* skipPerCent)
        setUpperLimit = setLowerLimit + int(allData.shape[0] * setSize)
        print("Lower Limit: {}, Upper Limit: {}".format(setLowerLimit, setUpperLimit))
        # Randomly select indeces for split
        randomizedOrder = torch.randperm(allData.shape[0]) if randInputOrder is None else randInputOrder
        self.randOrder = randomizedOrder
        # Select the first samples for training
        setIdxs = randomizedOrder[setLowerLimit:setUpperLimit]
        self.data = allData[setIdxs]
        self.target = allTargets[setIdxs]
        self.N = self.data.shape[0]
        self.dataDim = self.data.shape[1]
    # --------------------------------------------------------------------------------

   
    # --------------------------------------------------------------------------------
    def _transform_data_to_tensor():
        temp = torch.ones_like(self.data[0][0])
    # --------------------------------------------------------------------------------
    def __getitem__(self, index):
        """ DESCRIPTION: Mandatory function for a PyTorch dataset to implement. This is called by the dataloader iterator to return
                         batches of data for training.
        """
        data   = self.data[index]
        target = self.target[index]
        # In this framework dataloaders should return: data, target, misc
        return [data, target]

    # --------------------------------------------------------------------------------

    def __len__(self):
        return self.N
    

    def _create_word2vec_reps(tweetFile = './Data/training-Obama-Romney-tweets_corrected2_normalized_no_stop_words.txt', labelFile = './Data/training-Obama-Romney-tweets_corrected2_normalized_no_stop_words_labels.txt'):


        file1 = open(tweetFile, 'r') 
        lines = file1.readlines()

        # Get the average word 2 vec representation of all tweets. Unknown words are omitted.
        m = word2vec_rep(lines)

        # Save reps to disk for future use
        saveFile = './Data/avg_w2v_rep.npy'
        np.save(saveFile,m)
        return m

    # --------------------------------------------------------------------------------

    def _load_data(self, savedRepsFile=None, labelFile ='./Data/training-Obama-Romney-tweets_corrected2_normalized_no_stop_words_labels.txt' ):
        """ DESCRIPTION: This function will load data and tranform them into pytorch tensors. By default it loads already the representations found
                         in the Dta folder. If the file does not exist,  the function will load the preprocessed tweet file and compute the representation
                         then save them for future use. FInally, if a different argument is given in the savedRepsFile variable, then the function will load that instead
                         
           ARGUMENTS: savedRepsFile: (path or str): name of the file containing worv-vector representation. Must be in numpy format
                      labelFile: (array in txt): txt file containg the label of each tweet
                      
           RETUNRS:   m: tensor holdign the data. Dinemnsions numOFData x vector length
                      targets: tensor holding the target for each tweet. Dimensions: numOfData x 1 
        """
        if savedRepsFile == None:
            
            if os.path.exists('./Data/avg_w2v_rep.npy'):
                 m = np.load('./Data/avg_w2v_rep.npy')
            else:
                # Read all lines of tweet file, store them as list of strings
                file1 = open(tweetFile, 'r') 
                lines = file1.readlines()

                # Get the average word 2 vec representation of all tweets. Unknown words are omitted.
                # Function to load word vectors pre-trained on Google News
                m = word2vec_rep(lines)

                # Save reps to disk for future use
                saveFile = './Data/avg_w2v_rep.npy'
                np.save(saveFile,m)
        else:
            np.load(savedRepsFile)
            
        targets = np.loadtxt(labelFile)
        targets += 1 # Need to be non negative. Tweets class has -1 fro negative sentiment label

        return torch.from_numpy(m).type(torch.float32), torch.from_numpy(targets).type(torch.long)

# End of Tweets class
# ====================================================================================================================

# Arguments: None
# Returns: w2v (dict)
# Where, w2v (dict) is a dictionary with words as keys (lowercase) and vectors as values
def load_w2v():
    with open('./Data/w2v.pkl', 'rb') as fin:
        return pkl.load(fin)
    
def get_tokens(doc):
	tokens = re.split(r"[^A-Za-z0-9-']", doc)
	tokens = list(filter(len, tokens))
	return tokens


def word2vec_rep(docs):
    '''  DESCRIPTION:   Function to get word2vec averge representations. Input is a collection of documents. Docs are tokenized and for each line the average representation
                        is computed from its constituent tokens. Unknownwords are skipped.
    
         ARGUMENTS: docs: A list of strings, each string represents a document
         
         RETURNS:   mat (numpy.ndarray) of size (len(docs), dim) mat is a two-dimensional numpy array containing vector representation for ith document (in input list docs) in ith row
                    dim represents the dimensions of word vectors, here dim = 300 for Google News pre-trained vectors
    
    '''
    # Declare variables
    freqs = Counter()
    docFreqs = [Counter() for d in docs]
    docTokens = [[] for d in docs]
    voc = []

    # Build Vocabulary
    for i,d in enumerate(docs):
        docTokens = get_tokens(d.lower())                          # get tokens
        #isWord = [t in stopwords for t in tokens]       # see which word is stop word
        #docTokens[i] = [t for (t,i) in zip(tokens,isWord) if not i]# eliminate stopwords from furhter consideration    
        #docTokens[i] = [t for (t,i) in zip(tokens,isWord) if not i]# eliminate stopwords from furhter consideration    
        docFreqs[i] = Counter(docTokens)                   # count the freqs os this docs' tokens 
        freqs += Counter(docTokens)                        # Add the doc's freqs to total freqs


    # Dummy matrix
    dim = 300
    mat = np.zeros((len(docs), dim))
    w2v = load_w2v()

    # Create a sorted vocabulary out of the unique terms. Declare the matrix that hold the per doc
    # bag-of-word represeantions in terms of appeared token freqeuencies!
    voc = list(freqs.keys())
    voc.sort()

    # Build averaged representations
    cnt = 0
    embedding = np.zeros(dim)
    for i in range(len(docs)):
        for j, v in enumerate(voc):
            if v in w2v:
                embedding += (w2v[v] * docFreqs[i][v])
                cnt += docFreqs[i][v]
            # else: # word not in w2v, just consider it as adding 0's
                # cnt += 1
        cnt = cnt if cnt > 0 else 1
        mat[i] = embedding / cnt # average the summed embeddings
        cnt = 0
        embedding.fill(0) 


    return mat

   



#=================================================================
# Main
# ================================================================
def main():
    
    dSet  = Tweets(setSize=0.7)
    dSet2 = Tweets(randInputOrder = dSet.randOrder, skipPerCent = 0.1,setSize=0.1)
    dSet3 = Tweets(randInputOrder = dSet.randOrder, skipPerCent = 0.8,setSize=0.1)
    
    print(dSet.data.shape)
    print(dSet2.data.shape)
    print(dSet3.data.shape)
    print(dSet2.__getitem__(5)[0], dSet2.__getitem__(5)[1])
    print(dSet3.__getitem__(5)[0], dSet3.__getitem__(5)[1])
    # --------------------

if __name__ == "__main__":
    main()
