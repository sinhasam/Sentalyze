import numpy as np 
import torch 
import torch.nn.functional as F
from torch.autograd import Variable

os.chdir('..')
from data import Data 



LEARNING_RATE = 1e-3
EPOCH = 10000
NUM_CLASSES = 5
NUM_HIDDEN = 128 # commonly chosen hyperparam
DROPOUT_RATE = 0.5
EMBEDDING_SIZE = 5



class Net(nn.Module):
    def __init__(self, maxLength, batchSize, embedDim, inputSize, weights = None):
        super(Net, self).__init__()
        self.input = Variable(torch.tensor())
        self.batchSize = batchSize
        self.inputSize = inputSize
        self.embedDim = embedDim
        self.maxLength = maxLength
        
        self.setBLSTM()

        if not weights:
           self.setBLSTM()

        else: 
            weights = weights


    def setBLSTM(self):
        self.lstm = nn.LSTM(input_size=self.inputSize,hidden_size=NUM_HIDDEN,bidirectional=True,
                        dropout=DROPOUT_RATE, bias=True)
        
        