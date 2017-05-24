import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim



NUM_HIDDEN = 128
DROPOUT_RATE = 0.5
EMBEDDING_SIZE = 300
BATCH_SIZE = 1
NUM_FILTERS = 1
NUM_CLASSES = 5
NUM_LAYERS = 3
OUT_CHANNELS = 100


class Net(nn.Module):
    def __init__(self, maxSentenceLength, weights=None):
        
        super(Net, self).__init__()
        self.inputSize = EMBEDDING_SIZE
        self.weights = None

        self.train = True
        self.setLayers()


    def setLayers(self):
        self.LSTM = nn.LSTM(input_size=self.inputSize, hidden_size=NUM_HIDDEN, bidirectional=True, 
                        num_layers=NUM_LAYERS, dropout=DROPOUT_RATE, bias=True)
     
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=OUT_CHANNELS, kernel_size=3, bias=True), 
                                        nn.BatchNorm2d(OUT_CHANNELS),
                                        nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # 299 because embedding dim is with a kernel_stride of 3 results in
        # and 27 because max sentence length is 56 and after the convlayer and maxpool it is 27
        self.fcLayer = nn.Linear(342900, NUM_CLASSES, bias=True) 

        self.softmax = nn.LogSoftmax()



    def forward(self, in1, hidden):

        output, hidden = self.LSTM(in1, hidden)
        dimSize = output.size()
        
        output = output.resize(BATCH_SIZE, NUM_FILTERS, dimSize[0], dimSize[2]) # to make it 4d to perform 2 d convolutions

        output = self.convLayer(output)

        output = self.maxpool(output)


        output = output.view(-1, 342900)

        output = self.fcLayer(output)

        return self.softmax(output), hidden