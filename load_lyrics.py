import numpy as np
import pandas

# Load the input file as numpy array with a list of genres and words
def load(fname):
    genres,words = set(),set()
    with open(fname) as f:
        dat = []
        for l in f:
            p = l.strip().split(':')
            dat.append(p)
            genres.add(p[0])
            for word in p[1].split(' '):
                words.add(word)
        dat = np.array(dat)
    genres,words = list(genres),list(words)
    return (dat,genres,words)

# Load the input file with a one-hot representation for the genre labels
def load_one_hot_lyrics(fname):
    dat,genres,words = load(fname)
    x,y = dat[:,1],dat[:,0]
    y = np.array(pandas.get_dummies(y))

    return (x,y,genres,words)

# Load the input file with one-hot representation for genres and word count vectors
def load_bag_of_words(fname):
    x,y,genres,words = load_one_hot_lyrics(fname)

    # Build dict from word -> index
    wd = {}
    num_words = len(words)
    for i in range(num_words):
        word = words[i]
        wd[word] = i

    # Calculate word counts for each of the lyrics
    x_ = []
    for lyric in x:
        l = [0] * num_words
        for word in lyric.split():
            l[wd[word]] += 1
        x_.append(l)

    x_ = np.array(x_)
    return (x_,y,genres,words)
