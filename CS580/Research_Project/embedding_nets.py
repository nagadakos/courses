import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
# Python Imaging Library
import PIL
import numpy as np
import sys as sys
import utils
from utils import Print

# =========================================================================================================================================================================================
# TWEETS NETWORKS
# =========================================================================================================================================================================================

class SimpleLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        # Boiler plate code. Any init should declare the following
        self.descr = 'SimpleLSTM'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        """
        Initialize the model by setting up the layers.
        """
        super(SimpleLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
    
    # ====================================================================
    
class MultiLayerPerceptron(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [1, 300],
                  dims = {'nodes':[300, 100, 32, 3], 'kSizes':[5,5,3,3,3,3], 'strides':[1,1,2,2,2,2], 'linear1':[200,120]},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(MultiLayerPerceptron, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'MultiLayerPerceptron'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        dataChannels1 = '1d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        self.numOfDenseLayers = len(dims['nodes'])
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        self.layers = []
        dims['nodes'][0] = dataSize[1] # Get the input dataSize for first layer
        # Layers Declaration
        for i in range(0,  self.numOfDenseLayers-1):
            self.layers += [nn.Linear(dims['nodes'][i], dims['nodes'][i+1])]
            self.layers += [nn.ReLU()]

        self.layers = nn.ModuleList(self.layers)
        
        # Device handling CPU or GPU 
        self.device = device
          
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of {}:".format(self.descr))
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
    
class SimpleConvolutional(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [1,300], verbose = True,
                  dims = {'nodes':[50, 40, 30, 30], 'kSizes':[3,3,3,3], 'strides':[1,1,1,1], 'linearSizes':[200], 'targetSize': 3, 'latentDim':50, 'poolKSizes': [3,3,3,3], 'poolStrides':[2,2,2,2]},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(SimpleConvolutional, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'SimpleConvolutional'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        self.numOfConvLayers = len(dims['nodes'])
        self.latentDim = dims['latentDim']
        self.verbose = verbose
        self.device = device
        self.targetSize = dims['targetSize']
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        dataChannels1 = '1d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h, w = [], []
        h.append(dataSize[1])
        for i in range(1,self.numOfConvLayers+1):
            hi =  utils.comp_conv_dimensions(dataChannels1, h[i-1], 1, dims['kSizes'][i-1], stride = dims['strides'][i-1])
            #print("Dims after conv layer {}: {}".format(i, hi))
            hi =  utils.comp_conv_dimensions(dataChannels1, hi, 1, dims['poolKSizes'][i-1], stride = dims['poolStrides'][i-1])
            #print("Dims after pool layer {}: {}".format(i, hi))
            h.append(hi)
        
        self.linearSize = dims['nodes'][-1] * h[-1]
        #print("Dims for linear layaer: " + str(self.linearSize))
        self.h= h
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.convLayers  = [nn.Conv1d(dataSize[0], dims['nodes'][0], kernel_size=dims['kSizes'][0], stride = dims['strides'][0])]
        self.convLayers += [nn.MaxPool1d(dims['poolKSizes'][0], stride = dims['poolStrides'][0], dilation = 1 ,padding = 0)]
        for i in range(1,self.numOfConvLayers):
            self.convLayers += [nn.Conv1d(dims['nodes'][i-1], dims['nodes'][i], kernel_size=dims['kSizes'][i], stride = dims['strides'][i])]
            self.convLayers += [nn.MaxPool1d(dims['poolKSizes'][i], stride = dims['poolStrides'][i], dilation = 1 ,padding = 0)]
        #self.convLayers += [nn.Conv1d(dims['nodes'][i-1], dims['nodes'][i], kernel_size=dims['kSizes'][i], stride = dims['strides'][i]) for i in range(1,self.numOfConvLayers)]
        self.convLayers = nn.ModuleList(self.convLayers)
        self.linear = nn.Linear(self.linearSize, dims['linearSizes'][0])
        self.toClass = nn.Linear(dims['linearSizes'][0], dims['targetSize'])
        #print(self.convLayers)
        # Device handling CPU or GPU 
        self.device = device
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = x.unsqueeze(1) 
        #print(x.shape)    
        for i, layer in enumerate(self.convLayers):
            x = F.relu(layer(x))
            #print("Layer {}, shape: {}".format(i, x.shape))
        x = x.reshape(-1, self.linearSize)
        x = F.relu(self.linear(x))
        x = F.relu(self.toClass(x))
        return x
    
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        with torch.no_grad():
            preds = F.softmax(self.forward(x)).max(dim=1) 
        return preds
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of {}:".format(self.descr))
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
# ===============================================================================================================================================================================
# ENCODER NETS
# ===============================================================================================================================================================================

class BasicEncoder (nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                 dims2 = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':200}},
                 dims = {'nodes':[32,64,64,64], 'kSizes':[3,3,3,3], 'strides':[2,2,2,2], 'latentDim':200},
                 decDims = {'nodes':[64,64,64,32], 'kSizes':[3,3,3,3], 'strides':[2,2,2,2], 'latentDim':200},
                 decoder = None, verbose = False,
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(BasicEncoder, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'Basic_Encoder'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.trainMethod = 'label'
        self.classMethod = 'label'
        self.propLoss = nn.MSELoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        self.numOfConvLayers = len(dims['nodes'])
        self.latentDim = dims['latentDim']
        self.verbose = verbose
        self.device = device
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h, w = [], []
        h.append(dataSize[1])
        w.append(dataSize[2])
        for i in range(1,self.numOfConvLayers+1):
            hi, wi = utils.comp_conv_dimensions(dataChannels1, h[i-1], w[i-1], dims['kSizes'][i-1], stride = dims['strides'][i-1])
            h.append(hi)
            w.append(wi)
            print("Dims conv1 for linear layaer: {} {}".format(hi,wi))
        
        self.linearSize = dims['nodes'][-1] * h[-1]*w[-1]
        print("Dims for linear layaer: " + str(self.linearSize))
        self.h, self.w = h,w
        self.latentDim = dims['latentDim']
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.convLayers  = [nn.Conv2d(dataSize[0], dims['nodes'][0], kernel_size=dims['kSizes'][0], stride = dims['strides'][0])]
        self.convLayers += [nn.Conv2d(dims['nodes'][i-1], dims['nodes'][i], kernel_size=dims['kSizes'][i], stride = dims['strides'][i]) for i in range(1,self.numOfConvLayers)]
        self.convLayers = nn.ModuleList(self.convLayers)
        #self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        #self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.toLatentSpace = nn.Linear(self.linearSize, dims['latentDim'])
        
        # Device handling CPU or GPU 
        self.device = device
        
        if decoder is None:
            self.decoder = nn.Sequential(
                Print(self.verbose,div=True),
                nn.Linear(self.latentDim, self.linearSize),
                nn.ReLU(),
                Print(enable=self.verbose,div=True),
                utils.shapedUnFlatten(dims['nodes'][-1], self.h[-1], self.w[-1]),
                Print(enable=self.verbose,div=True),
                nn.ConvTranspose2d(dims['nodes'][-1], dims['nodes'][-2], kernel_size=dims['kSizes'][-1], stride=dims['strides'][-1], output_padding=0),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-2], dims['nodes'][-3], kernel_size=dims['kSizes'][-2], stride=dims['strides'][-2], output_padding=1),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-3], dims['nodes'][-4], kernel_size=dims['kSizes'][-3], stride=dims['strides'][-3], output_padding=0),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-4], dataSize[0], kernel_size=dims['kSizes'][-4], stride=dims['strides'][-4], output_padding=1),
                nn.Sigmoid(),
                Print(enable=self.verbose),
            )
        else:
            self.decoder = decoder
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        encodedX = self.encode(x)
        decodedX = self.decode(encodedX)
        return [x, decodedX]
    # ------------------
    def encode(self,x):
        for i in range(len(self.convLayers)):
            x = F.relu(self.convLayers[i](x))
            #print(x.shape)
        x = x.reshape(-1, self.linearSize)
        x = self.toLatentSpace(x)
        return x
    # ------------------
    def decode(self, x):
        decX = self.decoder(x) 
        return decX
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------
    def generate(self, sample=None, bSize=32,  **kwargs):
        """ DESCRIPTION: This function will generate an image based on an input ssample, usually encoded from the forward sample.
                         Sample can be a list of samples, where their average will be used to generate the final image. 
            ARGUMENTS: sample (list of tensors): Hold the samples to be used to generate new fruits from, can be any number
                                                 and it should be in ether in [batachSize,channels, h,w] or(channels,h,w) format .
                                                 If NONE, function will generate new data from sampling the latent space.
                       bSize (int): Batch size then the function generates data by sampling from latent space only. IS NOT USED when
                                    generating from input images.
            RETURNS: genData (tensor): Returned tensor shame shape as the original input, the decoded output that is either:
                                       a) The average represnetation of the provided samples, or if NO sample is given,
                                       b) The decoded output of a random latent space sample
        """
        if sample is not None:
            # Turn input to list, if not already
            if not isinstance(sample, list):
                sample = [sample]
            # If input is not in [batachSize,channels, h,w] format, and its just an image like (channels,h,w) add the batch dimension
            if len(sample[0].shape) < 4:
                for i in range(len(sample)):
                    sample[i] = sample[i].unsqueeze(0)
            # Preassign a tensor to hold the sum of the samples representations, the average of which will be used to generate a new fruit!
            genData = torch.zeros((sample[0].shape[0], self.latentDim)).to(self.device) 
            # Get a representation for all samples, average it and decode this average to get a new synthesized attempt!
            for i in range(len(sample)):
                genData += self.encode(sample[i])
            genData = self.decode(genData/len(sample))
            latentSample = 0
        else: #if latent sample is required
            latentSample = torch.rand((bSize, self.latentDim)).to(self.device)
            genData = self.decode(latentSample)
            
        return genData, latentSample
    # ------------------
    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
    
class BasicVAEEncoder (nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100], VAE = True,
                 dims2 = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':200}},
                 dims = {'nodes':[32,64,64,64], 'kSizes':[3,3,3,3], 'strides':[2,2,2,2], 'latentDim':200},
                 decDims = {'nodes':[64,64,64,32], 'kSizes':[3,3,3,3], 'strides':[2,2,2,2], 'latentDim':200},
                 decoder = None, encoder= None, verbose = False,
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(BasicVAEEncoder, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'Basic_VAE_Encoder'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.trainMethod = 'label'
        self.classMethod = 'label'
        self.VAE = VAE
        if self.VAE:
            self.propLoss = utils.BCE_KLD_CompoundLoss()
        else:
            self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        self.numOfConvLayers = len(dims['nodes'])
        self.latentDim = dims['latentDim']
        self.verbose = verbose
        self.device = device
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h, w = [], []
        h.append(dataSize[1])
        w.append(dataSize[2])
        for i in range(1,self.numOfConvLayers+1):
            hi, wi = utils.comp_conv_dimensions(dataChannels1, h[i-1], w[i-1], dims['kSizes'][i-1], stride = dims['strides'][i-1])
            h.append(hi)
            w.append(wi)
            #print("Dims conv1 for linear layaer: {} {}".format(hi,wi))
        
        self.linearSize = dims['nodes'][-1] * h[-1]*w[-1]
        #print("Dims for linear layaer: " + str(self.linearSize))
        self.h, self.w = h,w
        self.latentDim = dims['latentDim']
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.convLayers  = [nn.Conv2d(dataSize[0], dims['nodes'][0], kernel_size=dims['kSizes'][0], stride = dims['strides'][0])]
        self.convLayers += [nn.Conv2d(dims['nodes'][i-1], dims['nodes'][i], kernel_size=dims['kSizes'][i], stride = dims['strides'][i]) for i in range(1,self.numOfConvLayers)]
        self.convLayers = nn.ModuleList(self.convLayers)
        self.mu     = nn.Linear(self.linearSize, dims['latentDim'])
        self.logvar = nn.Linear(self.linearSize, dims['latentDim'])
        
        # Device handling CPU or GPU 
        self.device = device
        
        if decoder is None:
            self.decoder = nn.Sequential(
                Print(self.verbose,div=True),
                nn.Linear(self.latentDim, self.linearSize),
                nn.ReLU(),
                Print(enable=self.verbose,div=True),
                utils.shapedUnFlatten(dims['nodes'][-1], self.h[-1], self.w[-1]),
                Print(enable=self.verbose,div=True),
                nn.ConvTranspose2d(dims['nodes'][-1], dims['nodes'][-2], kernel_size=dims['kSizes'][-1], stride=dims['strides'][-1], output_padding=0),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-2], dims['nodes'][-3], kernel_size=dims['kSizes'][-2], stride=dims['strides'][-2], output_padding=1),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-3], dims['nodes'][-4], kernel_size=dims['kSizes'][-3], stride=dims['strides'][-3], output_padding=0),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-4], dataSize[0], kernel_size=dims['kSizes'][-4], stride=dims['strides'][-4], output_padding=1),
                nn.Sigmoid(),
                Print(enable=self.verbose),
            )
        else:
            self.decoder = decoder
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        mu, logvar  = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decodedX = self.decode(z)
        return [x, decodedX, mu, logvar]
    # ------------------
    def encode(self,x):
        for i in range(len(self.convLayers)):
            x = F.relu(self.convLayers[i](x))
            #print(x.shape)
        x = x.reshape(-1, self.linearSize)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    # ------------------
    def decode(self, x):
        decX = self.decoder(x) 
        return decX
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            # Turn logvar into std
            std = logvar.mul(0.5).exp_()
            # Sample from standard normal distribution of dimension same as mu
            esp = torch.randn(mu.size(),device=self.device)
            # Reparametrize into z.
            z = mu + std * esp
            return z
        else:
            return mu
    # ------------------
    def generate(self, sample=None, bSize=128,  **kwargs):
        """ DESCRIPTION: This function will generate an image based on an input ssample, usually encoded from the forward sample.
                         Sample can be a list of samples, where their average will be used to generate the final image. 
            ARGUMENTS: sample (list of tensors): Hold the samples to be used to generate new fruits from, can be any number
                                                 and it should be in ether in [batachSize,channels, h,w] or(channels,h,w) format .
                                                 If NONE, function will generate new data from sampling the latent space.
                       bSize (int): Batch size then the function generates data by sampling from latent space only. IS NOT USED when
                                    generating from input images.
            RETURNS: genData (tensor): Returned tensor shame shape as the original input, the decoded output that is either:
                                       a) The average represnetation of the provided samples, or if NO sample is given,
                                       b) The decoded output of a random latent space sample
        """
        self.eval()
        if sample is not None:
            # Turn input to list, if not already
            if not isinstance(sample, list):
                sample = [sample]
            # If input is not in [batachSize,channels, h,w] format, and its just an image like (channels,h,w) add the batch dimension
            if len(sample[0].shape) < 4:
                for i in range(len(sample)):
                    sample[i] = sample[i].unsqueeze(0)
            # Preassign a tensor to hold the sum of the samples representations, the average of which will be used to generate a new fruit!
            genData = torch.zeros((sample[0].shape[0], self.latentDim)).to(self.device) 
            # Get a representation for all samples, average it and decode this average to get a new synthesized attempt!
            for i in range(len(sample)):
                genData += self.reparameterize(*self.encode(sample[i]))
            genData = self.decode(genData/len(sample))
            latentSample = 0
        else: #if latent sample is required
            latentSample = torch.rand((bSize, self.latentDim)).to(self.device)
            genData = self.decode(latentSample)
            
        return genData, latentSample
    # ------------------
    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of :")

    # ====================================================================
