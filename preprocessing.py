from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np




class Processing(object):
	
	EMBEDDING_SIZE = 300 # to be decided

	def __init__(self, relPath, modelFilename, dataFileName, maxLength):
		print("Loading " + modelFilename)
		self.word2Vec = KeyedVectors.load_word2vec_format(modelFilename, binary = True)
		print("Done loading")
		self.dataFile = dataFileName
		self.maxLength = maxLength


	def sentence2vec(self, *args): # maybe multiple sentence? not sure of the use case
		for arg in args:
			sentence = str(arg)
			for wordCount, word in enumerate(sentence):
				word2sentence = np.zeros((EMBEDDING_SIZE, self.maxLength))
			word2sentence[:,wordCount] = self.word2Vec[word]
		return word2sentence, wordCount
