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

#KNN Model

def learn_model(k,features,labels):
  return k,features.copy(),labels.copy()


#Finding the Class of max nieghbors ( for KNN )
def plurality(xs):
  from collections import defaultdict
  counts = defaultdict(int)
  for x in xs:
    counts[x] += 1
  maxv = max(counts.values())
  for k,v in counts.items():
    if v == maxv:
      return k



def apply_model(features,model):
  k,train_feats,labels = model  
  results = []
  for f in features:
    label_dist = []
    for t,ell in zip(train_feats,labels):
      label_dist.append( (np.linalg.norm(f-t), ell) )

