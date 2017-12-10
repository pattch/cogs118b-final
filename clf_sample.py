import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

fname = 'reducedsample.txt'
if len(sys.argv) > 1:
    fname = sys.argv[1]

def process_line(l):
    return [int(n) for n in l.strip().split()]

def process_words(wl):
    return wl.strip().split()

with open(fname) as f:
    dim = int(next(f))
    words = process_words(next(f))
    dat = [process_line(l) for l in f]

dat = np.array(dat)
# Get x
x = dat[:,:dim]
x = np.array(x,dtype=np.float64)
# Get y
y = dat[:,dim:]

for sample in x:
    # Divide by the sum of word counds
    sample /= sum(sample)
    # Scale by the IDF vector
    sample *= idf

for i in range(x.shape[0]):
    n = norm[i]
    if sum(n):
        x[i] /= n
print(x.shape,y.shape)

print('Fitting Classifier: Random Forest Classifier')
clf = RFC(n_estimators=128,max_depth=50,random_state=0)
clf.fit(x,y)
print('Accuracy:',clf.score(x,y))