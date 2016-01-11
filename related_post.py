import scipy as sp
import os
import sys
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem

DIR = r"data/toy"
posts = [open(os.path.join(DIR,f)).read() for f in os.listdir(DIR)]

new_post = 'imaging databases'
#stemming
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
  def build_analyzer(self):
    analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
    return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english',charset_error='ignore')

X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape
print('#sample is %d, #features is %d' % (num_samples,num_features))	

new_post_vec = vectorizer.transform([new_post])

def dist_norm(v1,v2):
  norm_v1 = v1/sp.linalg.norm(v1.toarray())
  norm_v2 = v2/sp.linalg.norm(v2.toarray())
  delta=norm_v1-norm_v2
  return sp.linalg.norm(delta.toarray())


#Cal doc Similarity with bag of words

best_doc = None
best_dist = sys.maxint
best_i = None

for i in range(0, num_samples):
  post = posts[i]
  if post == new_post:
    continue
  post_vec = X_train.getrow(i)
  d = dist_norm(post_vec, new_post_vec)
  print "post %i with dist =%.2f : %s" % (i,d,post)
  if d < best_dist:
    best_dist = d
    best_i = i
    best_doc = post

print ("Best post is %i with dist =%.2f" % (best_i,best_dist))


# Stemming with nltk


