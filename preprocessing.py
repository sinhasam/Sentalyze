from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np


EMBEDDING_SIZE = 300 

class Processing(object):
	def __init__(self, modelFilename, maxLength):
		print("Loading " + modelFilename)
		self.word2Vec = KeyedVectors.load_word2vec_format(modelFilename, binary=True)
		print("Done loading")
		self.maxLength = maxLength


	def sentence2vec(self, *args): # maybe multiple sentence? not sure of the use case
		for arg in args:
			sentence = str(arg)
			numWords = len(sentence)
			word2sentence = np.zeros((1, self.maxLength*EMBEDDING_SIZE))
			word2sentenceMatrix = np.zeros((self.maxLength, EMBEDDING_SIZE))
			for count, word in enumerate(sentence):
				try:
					word2sentenceMatrix[count] = self.word2Vec[word]
				except KeyError as e:
					continue
			word2sentenceVec = word2sentenceMatrix.reshape(1,-1)

		return word2sentenceVec, count