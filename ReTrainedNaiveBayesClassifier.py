'''
This python module extends the NaiveBayesClassifier with additional functionality of retrain with new document
'''
from nltk import NaiveBayesClassifier;
from nltk import MaxentClassifier;
from nltk import classify;
from collections import defaultdict
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs

class ReTrainedNaiveBayesClassifier (NaiveBayesClassifier):
	
	def __init__(self, label_probdist, feature_probdist, feature_freqdist=None, feature_values=None, label_freqdist=None, fnames=None):
		self._feature_freqdist = feature_freqdist;
		self._feature_values = feature_values;
		self._label_freqdist = label_freqdist;
		self._fnames = fnames
		self._labels = list(label_probdist.samples())
		super(ReTrainedNaiveBayesClassifier, self).__init__(label_probdist, feature_probdist);
                                                  
	@classmethod
	def train(cls, labeled_featuresets, estimator=ELEProbDist):
		"""
		:param labeled_featuresets: A list of classified featuresets,
		i.e., a list of tuples ``(featureset, label)``.
		"""
		label_freqdist = FreqDist()
		feature_freqdist = defaultdict(FreqDist)
		feature_values = defaultdict(set)
		fnames = set()

		# Count up how many times each feature value occurred, given
		# the label and featurename.
		for featureset, label in labeled_featuresets:
			label_freqdist[label] += 1
			for fname, fval in featureset.items():
				# Increment freq(fval|label, fname)
				feature_freqdist[label, fname][fval] += 1
				# Record that fname can take the value fval.
				feature_values[fname].add(fval)
				# Keep a list of all feature names.
				fnames.add(fname)

		# If a feature didn't have a value given for an instance, then
		# we assume that it gets the implicit value 'None.'  This loop
		# counts up the number of 'missing' feature values for each
		# (label,fname) pair, and increments the count of the fval
		# 'None' by that amount.
		for label in label_freqdist:
			num_samples = label_freqdist[label]
			for fname in fnames:
				count = feature_freqdist[label, fname].N()
				# Only add a None key when necessary, i.e. if there are
				# any samples with feature 'fname' missing.
				if num_samples - count > 0:
					feature_freqdist[label, fname][None] += num_samples - count
					feature_values[fname].add(None)

		# Create the P(label) distribution
		label_probdist = estimator(label_freqdist)

		# Create the P(fval|label, fname) distribution
		feature_probdist = {}
		for ((label, fname), freqdist) in feature_freqdist.items():
			probdist = estimator(freqdist, bins=len(feature_values[fname]))
			feature_probdist[label, fname] = probdist

		return cls(label_probdist, feature_probdist, feature_freqdist, feature_values, label_freqdist, fnames)

		def retrain(self, labeled_featuresets, estimator = ELEProbDist):
			for featureset, label in labeled_featuresets:
				self._label_freqdist[label] += 1
				for fname, fval in featureset.items():
					self._feature_freqdist[label, fname][fval] += 1
					self._feature_values[fname].add(fval)

			for label in self._label_freqdist:
				num_samples = self._label_freqdist[label]
				for fname in self._fnames:
					count = self._feature_freqdist[label, fname].N()
					# Only add a None key when necessary, i.e. if there are
					# any samples with feature 'fname' missing.
					if num_samples - count > 0:
						self._feature_freqdist[label, fname][None] += num_samples - count
						self._feature_values[fname].add(None)

			for ((label, fname), freqdist) in self._feature_freqdist.items():
				probdist = estimator(freqdist, bins=len(self._feature_values[fname]))
				self._feature_probdist[label, fname] = probdist
			
			self._label_probdist = estimator(self._label_freqdist)
			self._labels = list(self._label_probdist.samples())
