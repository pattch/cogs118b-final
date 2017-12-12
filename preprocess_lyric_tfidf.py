import sys
import numpy as np
from itertools import count
from load_lyrics import load_bag_of_words as load
from scipy import sparse
from scipy.sparse import *

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
# TODO: Support only loading songs with genres from HMM
x,y,genres,words = load(fname)

print("X Shape",x.shape)

# Sum up all of the words
def word_counts(x_):
    s = np.zeros(len(words))
    for i in range(x_.shape[0]):
        s += x_[i]
    return np.array(s).reshape((len(words),))

# 'Documents' are genres, have a |W| * |G| vector to keep track of occurences in order to calculate weighting vector
def idf(wc,y_):
    # Calculate IDF Vector
    idf = np.zeros((len(words),y_.shape[1])).astype(np.dtype('bool'))
    for i in range(1,wc.shape[0]):
        sample = wc[i].toarray().reshape((wc.shape[1],))
        # print('Sample',sample.shape,sample)
        cat = y_[i]
        # np.nonzero returns tuple of np arrays, first 0 access gets the first
        # nonzero list from cat, which is the only dimension, the next 0 access gets
        # the first (and should be only) index in cat that is nonzero
        cat_idx = np.nonzero(cat)[0][0]
        # print('Cat IDX',cat_idx,np.nonzero(cat))
        nonzero_word_indices = np.nonzero(sample)[0]
        # print('NWI',nonzero_word_indices)
        for wi in nonzero_word_indices:
            # Mark word as present in the appropriate position
            # Marks that a given word was present for a given category
            idf[wi][cat_idx] = True
    # Convert boolean values back to 1s and 0s
    idf = idf.astype(np.dtype('float64'))
    # print('IDF',idf,idf.shape)
    # Number of Genres Considered
    cats = y.shape[1]
    # Calculate IDF here -- log of N / num categories with word
    idf_num = np.array([np.log2(cats / sum(word_in_cat)) if sum(word_in_cat) else 0 for word_in_cat in idf])
    # print('IDF',idf_num,sum(idf_num))
    return idf_num.reshape((len(words),))

def tf(w):
    if w.nnz > 0:
        return np.array(w / max(w))
    return w.toarray()

def calculate_tf_idf(wc,y_):
    idf_ = idf(wc,y_)
    # for w in wc:
    #     w = tf(w) * idf_
    for i in range(wc.shape[0]):
        wc[i] = tf(wc[i]) * idf_
    return wc

# Calculate important word metric:
def calculate_metric(x_,y_):
    # Get Word Counts
    wc = word_counts(x_)
    tf_idf_vec = calculate_tf_idf(x_,y_)
    print("TFIDF Shape",tf_idf_vec.shape)
    tf_idf_vec = tf_idf_vec.toarray()
    # print(tf_idf_vec)
    return np.array(tf_idf_vec)

xtf = calculate_metric(x,y)
s = word_counts(xtf)
total = sum(s)
# Pair the counts with the indices, s is now a list of tuples
s = list(enumerate(s))
# Sort tuples by word count, preserves the original indices
s = sorted(s,key=lambda tup: tup[1],reverse=True)
s = s[:dimensionality]
x_ = sparse.lil_matrix(xtf.shape,dtype=np.dtype('float64'))
x_ = x_[:,0:dimensionality]
print('Final Shape',x_.shape)
words_ = [i for i,c in s]
words_ref = [words[i] for i in words_]
print('Number of words in Reduced Space:',len(words_))
assert len(words_) == dimensionality

for i in range(x.shape[0]):
    for transformed_index,previous_index in enumerate(words_):
        x_[i,transformed_index] = xtf[i,previous_index]
print('Writing to file',outfile)
with open(outfile,'w',encoding='utf-8') as f:
    l = ' '.join(words_ref)
    f.write(str(dimensionality) + '\n')
    f.write(l + '\n')
    for i in range(x_.shape[0]):
        c = x_[i].toarray().reshape((dimensionality,))
        # print(sum(c))
        l = ' '.join([str(count) for count in c]) + ' ' + ' '.join([str(genre) for genre in y[i]])
        f.write(l + '\n')
print('Finished writing.')