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
from collections import Counter
import glob
import pickle as pkl
import re
# load image and convert to and from NumPy array
from numpy import asarray
# End Imports
# ----------------------------------------------------

# File Global Variables and datapath setting
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path,"Data")

sys.path.insert(0, data_path)


def_tweet_file = os.path.join(data_path,'training-Obama-Romney-tweets_corrected2_normalized_no_stop_words.txt')
def_label_file = os.path.join(data_path, 'training-Obama-Romney-tweets_corrected2_normalized_no_stop_words_labels.txt')

# =================================================================================================================
# Class Declaration
# =================================================================================================================
class Tweets(Dataset):
    
    def __init__(self, tweetFile= def_tweet_file, labelFile=def_label_file, randInputOrder = None, skipPerCent = None,  dataDir=None, csv_file=None, setSize = 0.8, repType = 'avgReps', targetSize = 0):
        
        allData, allTargets =  self._load_data(tweetFile = tweetFile, labelFile = def_label_file, repType = repType, targetSize = 0)
        setLowerLimit = 0 if skipPerCent is None else int(allData.shape[0]* skipPerCent)
        setUpperLimit = setLowerLimit + int(allData.shape[0] * setSize)
        print("Lower Limit: {}, Upper Limit: {}".format(setLowerLimit, setUpperLimit))
        if randInputOrder is not False:
            # Randomly select indeces for split
            randomizedOrder = torch.randperm(allData.shape[0]) if randInputOrder is None else randInputOrder
            self.randOrder = randomizedOrder
            # Select the first samples for training
            setIdxs = randomizedOrder[setLowerLimit:setUpperLimit]
            self.data = allData[setIdxs]
            self.target = allTargets[setIdxs]
        else:
            self.data= allData
            self.target = allTargets
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

    def _load_data(self, tweetFile = def_tweet_file, savedRepsFile=None, labelFile = def_label_file, repType = 'avgReps', targetSize = 0):
        """ DESCRIPTION: This function will load data and tranform them into pytorch tensors. By default it loads already the representations found
                         in the Dta folder. If the file does not exist,  the function will load the preprocessed tweet file and compute the representation
                         then save them for future use. FInally, if a different argument is given in the savedRepsFile variable, then the function will load that instead
                         
           ARGUMENTS: savedRepsFile: (path or str): name of the file containing worv-vector representation. Must be in numpy format
                      labelFile: (array in txt): txt file containg the label of each tweet
                      repType: (str avgReps | tweetReps): Selector for average representations per tweet, or tweet bundle reps.
                      
           RETUNRS:   m: tensor holdign the data. Dinemnsions numOFData x vector length
                      targets: tensor holding the target for each tweet. Dimensions: numOfData x 1 
        """
        # Get base path and add keywords to check if representation file already exists; comptue it otherwise
        rootPath = tweetFile.split('.')[0]
        repFile = 'avg_w2v_rep.npy' if repType == 'avgReps' else ('tweet_w2v_rep.npy' if repType == 'tweetReps' else '')
        repFile = '_'.join((rootPath, repFile))
        
        # IF not ready rep are given check if default exist
        if savedRepsFile == None:
            if os.path.exists(repFile):
                 m = np.load(repFile)
            else:
                print('Creating File ', repFile)
                # Read all lines of tweet file, store them as list of strings
                with open(tweetFile) as f:
                    lines = f.read().splitlines() 

                # Get the average word 2 vec representation of all tweets. Unknown words are omitted.
                # Function to load word vectors pre-trained on Google News
                if repType == 'avgReps':
                    m = word2vec_rep(lines)
                elif repType == 'tweetReps':
                    m = tweet_summary_reps(lines, targetSize = 0)
                else:
                    print("Invalid representation load selector. Choose either avgReps of tweetReps")
                    return -1
                # Save reps to disk for future use
                saveFile = '_'.join((rootPath, repType.split('R')[0] +'_w2v_rep.npy'))
                np.save(saveFile,m)
                np.savetxt(saveFile.split('.')[0]+'.txt', m)
        else:
            np.load(savedRepsFile)
        # If data is not for just prediction    
        if labelFile is not None:    
            targets = np.loadtxt(labelFile)
            targets += 1 # Need to be non negative. Tweets class has -1 fro negative sentiment label
        else:
            targets = np.ones_like(m[:,0]) *-1 # dummy targets for prediction; used to work with the above code
        return torch.from_numpy(m).type(torch.float32), torch.from_numpy(targets).type(torch.long)

# End of Tweets class
# ====================================================================================================================

def tweet_summary_reps(lines, lenSizeType = 'maxVal', targetDesnity = 0.7, targetSize = 0):
    lengths = []
    splitLines = []
    totalLines = 0
    for l in lines:
        splitLine = l.split(' ')
        splitLine = [p for p in splitLine if p != '']
        lineLen = len(splitLine)
        totalLines += 1
        lengths.append(lineLen)
        splitLines.append(splitLine)
        
    occurence_count = Counter(lengths) 
    # Find most frequent length. Structure is (top k most common tuples (val, freq)[choose tuple][val or freq])
    freqs = occurence_count.most_common()
    occurs, avgLen = 0, 0
    maxVal = 0
    # Pick the average size derived from the vals whose total frequency is 70% of the overall data
    for i, t in enumerate(freqs):
        occurs += t[1]
        avgLen += float(t[0] * t[1])
        maxVal = t[0] if t[0] > maxVal else maxVal
        #print(occurs, avgLen, targetDesnity (def 0.7)* totalLines, i, t[0], t[1])
        if occurs>= targetDesnity * totalLines:
            break
            
    mode = int(avgLen/ occurs)
    print('At {} density tweet length mode: {}, max val: {}'.format(targetDesnity, mode, maxVal))
    if targetSize == 0:
        targetSize= maxVal if lenSizeType == 'maxVal' else mode
    # Turn all tweets to worv2vec reps. Pad all shorter and cut all large tweets to targetSize.
   
    cnt, flag  = 0, 0    
    # Declare rep Matrix. Dimension of word2vec is 300. So, the matrix should be numOfTweets * targetSize
    dim = 300
    mat = np.ones((len(lines), dim* targetSize))* (-2)
    w2v = load_w2v()
    embedding = np.zeros(dim)
    #print(type(w2v))
     # Build  representations. if a tweet is longer than targetSize cut it to target size
    for i, l in enumerate(splitLines):
        if len(l) > targetSize:
            l = l[:targetSize]
        #if i< 30:
            #print(l)
        for j, w in enumerate(l):
            # if word is known add its represention, otherwise treat it as 0.              
            #if i< 30:
                #print(w)
            if w in w2v:
                mat[i, j*300: (j+1)*300] = w2v[str(w)]
    
    print("Tweet_word2vec size: {}".format(mat.shape))

    return mat

# --------------------------------------------


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
    
    dSet  = Tweets(setSize=0.7, repType= 'tweetReps')
    #dSet2 = Tweets(randInputOrder = dSet.randOrder, skipPerCent = 0.1,setSize=0.1)
    #dSet3 = Tweets(randInputOrder = dSet.randOrder, skipPerCent = 0.8,setSize=0.1)
    
    print(dSet.data.shape)
    #print(dSet2.data.shape)
    #print(dSet3.data.shape)
    #print(dSet2.__getitem__(5)[0], dSet2.__getitem__(5)[1])
    #print(dSet3.__getitem__(5)[0], dSet3.__getitem__(5)[1])
    # --------------------

if __name__ == "__main__":
    main()
