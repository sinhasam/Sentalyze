import os
import re



class Data(object):
	def __init__(self, relPath, fileName):
		self.fileName = fileName
		self.relPath = relPath
		self.findLongestSentence()


	def findLongestSentence(self):
		self.maxLength = 0
		os.chdir(self.relPath)
		with open(self.fileName) as file:
			for count, line in enumerate(file):
				match = re.match("(.*)\|([0-9]*$)", line)
				line = match.group(1)
				line = line.split()
				self.maxLength = max(self.maxLength, len(line))


	def __str__(self):
		return str(self.maxLength)



Data("imdb_data/", "dictionary.txt")
