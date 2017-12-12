import sys
import numpy as np
from itertools import count
from load_lyrics import load_bag_of_words as load
from scipy import sparse
from sklearn.model_selection import train_test_split

fname,outfile = 'sample.txt','reducedsample.txt'
print_statistics = False
dimensionality = 5000

if len(sys.argv) > 1:
    fname = sys.argv[1]
if len(sys.argv) > 2:
    print_statistics = ('count' in [c.lower() for c in sys.argv])
if len(sys.argv) > 2:
    for arg in sys.argv:
        if 'dim' in arg.lower():
            dimensionality = int(arg.split('=',1)[1])
        if 'out' in arg.lower():
            outfile = arg.split('=',1)[1]

print(fname,outfile,dimensionality)

# Load all of the songs
x,y,genres,words = load(fname)

print(x.shape)

# Sum up all of the words
def word_counts(x_):
    s = np.zeros(len(words))
    for i in range(x_.shape[0]):
        s += x_[i]
    return np.array(s).reshape((len(words),))

# Calculate important word metric:
def calculate_metric(x_):
    return word_counts(x_)

s = calculate_metric(x)
total = sum(s)
# Pair the counts with the indices, s is now a list of tuples
s = list(enumerate(s))
# Sort tuples by word count, preserves the original indices
s = sorted(s,key= lambda tup: tup[1],reverse=True)

if print_statistics:
    count = 0
    quar,half,thrq,nine,ninf,ninn,nnnn = 0,0,0,0,0,0,0
    for i,tup in enumerate(s):
        idx,c = tup
        count += c
        if not quar and (count / total) > 0.25:
            quar = i

        if not half and (count / total) > 0.5:
            half = i

        if not thrq and (count / total) > 0.75:
            thrq = i

        if not nine and (count / total) > 0.9:
            nine = i

        if not ninf and (count / total) > 0.95:
            ninf = i

        if not ninn and (count / total) > 0.99:
            ninn = i

        if not nnnn and (count / total) > 0.999:
            nnnn = i

    print('Covering 25% of all word counts takes',quar,'total unique words')
    print('Covering 50% of all word counts takes',half,'total unique words')
    print('Covering 75% of all word counts takes',thrq,'total unique words')
    print('Covering 90% of all word counts takes',nine,'total unique words')
    print('Covering 95% of all word counts takes',ninf,'total unique words')
    print('Covering 99% of all word counts takes',ninn,'total unique words')
    print('Covering 99% of all word counts takes',nnnn,'total unique words')

s = s[:dimensionality]
x_ = sparse.lil_matrix((x.shape[0],dimensionality),dtype=np.int8)
words_ = [i for i,c in s]
words_ref = [words[i] for i in words_]
assert len(words_) == dimensionality

for i in range(x.shape[0]):
    for transformed_index,previous_index in enumerate(words_):
        x_[i,transformed_index] = x[i,previous_index]

x_ = x_.toarray()
with open(outfile,'w',encoding='utf-8') as f:
    l = ' '.join(words_ref)
    f.write(str(dimensionality) + '\n')
    f.write(l + '\n')
    for i in range(x_.shape[0]):
        l = ' '.join([str(count) for count in x_[i]]) + ' ' + ' '.join([str(genre) for genre in y[i]])
        f.write(l + '\n')
