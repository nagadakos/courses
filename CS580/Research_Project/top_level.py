import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import sys as sys
from classifier_template import ClassifierFrame
from autoencoder_template import AutoEncoderFrame
import embedding_nets as eNets
from tweet_loader import  Tweets
from plotter import display_tensor_image
import os
import argparse
from utils import MSE_KLD_CompoundLoss
dir_path   = os.path.dirname(os.path.realpath(__file__))
tools_path = os.path.join(dir_path, "../../Code/")
sys.path.insert(0, tools_path)

#  Global Parameters
# Automatically detect if there is a GPU or just use CPU.
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========================================================================================================================
# Functions and Network Template
# ========================================================================================================================
def load_data(dataPackagePath = None, bSize = 32, trainSize =0.8, testSize = None, valSize = 0):
    # bundle common args to the Dataloader module as a kewword list.
    # pin_memory reserves memory to act as a buffer for cuda memcopy 
    # operations
    testSize = 1 - trainSize if testSize is None else testSize
    comArgs = {'shuffle': True,'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Data Loading -----------------------
    # ******************
    # 
    # ******************
    
    # Load  PyTorch data set
    trainSet = Tweets(setSize=trainSize)
    testSet = Tweets(randInputOrder = trainSet.randOrder, skipPerCent = trainSize, setSize=testSize)
    # Create a PyTorch Dataloader
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = bSize, **comArgs )
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = bSize, **comArgs)
    # End of DataLoading -------------------


    return trainLoader, testLoader
# --------------------------------------------------------------------------------------------------------

def parse_args():
    ''' Description: This function will create an argument parser. This will accept inputs from the console.
                     But if no inputs are given, the default values listed will be used!
        

    '''
    parser = argparse.ArgumentParser(prog='Fashion MNIST Network building!')
    # Tell parser to accept the following arguments, along with default vals.
    parser.add_argument('--lr',    type = float,metavar = 'lr',   default='0.001',help="Learning rate for the oprimizer.")
    parser.add_argument('--m',     type = float,metavar = 'float',default= 0.9,     help="Momentum for the optimizer, if any.")
    parser.add_argument('--bSize', type = int,  metavar = 'bSize',default=32,     help="Batch size of data loader, in terms of samples. a size of 32 means 32 images for an optimization step.")
    parser.add_argument('--epochs',type = int,  metavar = 'e',    default=8   ,   help="Number of training epochs. One epoch is to perform an optimization step over every sample, once.")
    parser.add_argument('--debug', type = bool,  metavar = 'debug',default=False,  help="Sets debug mode. Training, testing will orceed for only 1 batch and stop.")
    parser.add_argument('--trClss',type = bool, metavar = 'trClss',default=True,  help="Enables Training and evaluation of a classifier")
    parser.add_argument('--trEnc', type = bool, metavar = 'trEnc', default=True,  help="Enable training and evaluation of autoencoders")
    # Parse the input from the console. To access a specific arg-> dim = args.dim
    args = parser.parse_args()
    lr, m, bSize, epochs, debug, trainClassifier, trainEncoder = args.lr, args.m, args.bSize, args.epochs, args.debug, args.trClss, args.trEnc
    # Sanitize input
    m = m if (m>0 and m <1) else 0 
    lr = lr if lr < 1 else 0.1
    # It is standard in larger project to return a dictionary instead of a myriad of args like:
    # return {'lr':lr,'m':m,'bSize':bbSize,'epochs':epochs}
    return lr, m , bSize, epochs, debug, trainClassifier, trainEncoder

# ================================================================================================================================
# Execution
# ================================================================================================================================
def main():
    
    # Handle command line input and load data
    # Get keyboard arguments, if any! (Try the dictionary approach in the code aboe for some practice!)
    lr, m , bSize, epochs, debug, trainClassifier, trainEncoder = parse_args()
    # Load data, initialize model and optimizer!
    # Use this for debugg, loads a tiny amount of dummy data!
    if debug:
        trainLoader, testLoader = load_data(dataPackagePath = os.path.join(dir_path, 'Data','dummy.npz'),  bSize=bSize)
        fitArgs = dict(epochs = 1, earlyStopIdx = 1, earlyTestStopIdx = 1)
    else:
        trainLoader, testLoader = load_data(bSize=bSize)
        fitArgs = dict(epochs = epochs, earlyStopIdx = 0, earlyTestStopIdx = 0)
    # ---|
    
    # ********************
    # Classify Tweets!
    # ********************
    
    trainClassifier = True
    if trainClassifier:
        print("Top level device is :{}".format(device))
        # Declare your model and other parameters here
        embeddingNetKwargs = dict(device=device)
        #embeddingNet = eNets.ANET(**embeddingNetKwargs).to(device)
        #embeddingNet = eNets.MultiLayerPerceptron(**embeddingNetKwargs).to(device)
        embeddingNet = eNets.SimpleConvolutional(**embeddingNetKwargs).to(device)
        loss = embeddingNet.propLoss #nn.CrossEntropyLoss() # or use embeddingNet.propLoss (which should bedeclared at your model; its the loss function you want it by default to use)
        fitArgs['lossFunction'] = loss
        # ---|

        # Bundle up all the stuff into dicts to pass them to the template, this are mostly for labellng purposes: ie how to label the saved model, its plots and logs.
        templateKwargs = dict(lr=lr, momnt=m, optim='SGD', loss = str(type(loss)).split('.')[-1][:-2], targetApp = 'Tweet_Classification')
        kwargs = dict(templateKwargs=templateKwargs, encoderKwargs=embeddingNetKwargs)
        # ---|

        # Instantiate the framework with the selected architecture, labeling options etc 
        model = ClassifierFrame(embeddingNet, **kwargs)
        optim = optm.SGD(model.encoder.parameters(), lr=lr, momentum=m)
        #optim = optm.Adam(model.encoder.parameters(), lr=lr)
        # ---|

        print("######### Initiating {} Network training #########\n".format(model.descr))
        print("Parameters: lr:{}, momentum:{}, batch Size:{}, epochs:{}".format(lr,m,bSize,epochs))
        model.fit(trainLoader, testLoader, optim, device, **fitArgs)

        # Final report
        model.report()
   
    
# Define behavior if this module is the main executable. Standard code.
if __name__ == '__main__':
    main()
