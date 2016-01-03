import numpy as np


#Load dataset from file

ifile = open('seeds.tsv')

features = []
labels = []

for line in ifile:
  token = line.strip().split('\t')
  features.append([ float(tv) for tv in token[:-1]])
  labels.append(token[-1])

features = np.array(features)
labels = np.array(labels)

print (features.shape)
print (features[:6])
print (labels.shape)
print (labels[:6])


