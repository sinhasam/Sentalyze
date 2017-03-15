import os
import re
from preprocessing import Processing
from queue import Queue

class Data(object):
	def __init__(self, relPath, dataFileName, modelFileName):
		self.dataFileName = dataFileName
		self.modelFileName = modelFileName
		self.relPath = relPath
		self.queue = Queue()
		self.prepareData()
		self.index = 0
		self.processing = Processing(relPath, self.dataFileName, self.modelFileName, self.maxLength)


	def __str__(self):
		return str(self.maxLength)


	def prepareData(self):
		self.maxLength = 0
		os.chdir(self.relPath)
		with open(self.fileName) as file:
			for count, line in enumerate(file):
				match = re.match("(.*)\|([0-9]*$)", line)
				line = match.group(1)
				line = line.split()
				self.maxLength = max(self.maxLength, len(line))
				self.queue.put(line)
			self.numSentences = count


	def getNextSentence(self):
		nextLine = self.queue.get()
		vecSentence, sentenceLen = self.processing.sentence2vec(nextLine)
