import torch
import os
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from model import Net
import torch.nn as nn
import sys

from preprocessing import Processing

from data import Data

dtype = torch.FloatTensor

NUM_EPOCH = 100
LEARNING_RATE = 1e-3
NUM_CLASSES = 5
EMBEDDING_SIZE = 300
BATCH_SIZE = 1
NUM_HIDDEN = 128
NUM_LAYERS = 3

data = Data("imdb_data", "dictionary.txt", "GoogleNews-vectors-negative300.bin", "sentiment_labels.txt")
numSentences = data.numSentences

model = Net(data.maxLength, EMBEDDING_SIZE)

criterion = nn.NLLLoss() # dont need cross entropy since we do softmax in the model

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)


# (sequence_length, batch_size, input_size)
# (num_layers*2, batch, hidden_size)
hiddenVariable = nn.Parameter(torch.randn(NUM_LAYERS*2, BATCH_SIZE, NUM_HIDDEN)) # "*2" is for num_directions which = 2 since it is bidirectional
# (num_layers*2, batch, hidden_size)
cellVariable = nn.Parameter(torch.randn(NUM_LAYERS*2, BATCH_SIZE, NUM_HIDDEN))

for epoch in range(NUM_EPOCH):
	print("On epoch " + str(epoch))
	hidden = (hiddenVariable, cellVariable)
	for sentenceCount in range(numSentences):
		print(sentenceCount)
		if sentenceCount % 200 == 0:
			print("sentenceCount " + str(sentenceCount))
		model.zero_grad()

		sentence, label, sentenceLen = data.getNextSentence()

		if sentenceLen < 3:
			print('invalid sentence')
			continue

		label = Variable(torch.LongTensor([label]))

		

		sentence = np.resize(sentence, (data.maxLength, BATCH_SIZE, EMBEDDING_SIZE))
		sentence = torch.from_numpy(sentence)
		sentence = sentence.type(new_type=dtype)
		sentence = Variable(sentence)

		logProb, hidden = model.forward(sentence, hidden)

		loss = criterion(logProb, label)

		loss.backward(retain_variables=True)
		optimizer.step()
	

