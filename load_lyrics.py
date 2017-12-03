import numpy as np
import pandas
from scipy import sparse

# Load the input file as numpy array with a list of genres and words
def load(fname,verbose=False):
    if verbose:
        with open(fname) as f:
            for i,l in enumerate(f):
                pass
        word_count = i+1
        bar = progressbar.ProgressBar(maxval=word_count,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    genres,words = set(),set()
    with open(fname) as f:
        dat = []
        for i,l in enumerate(f):
            p = l.strip().split(':')
            dat.append(p)
            genres.add(p[0])
            for word in p[1].split(' '):
                words.add(word)

            if verbose and i % 1000 == 0:
                bar.update(i+1)
        dat = np.array(dat)
    bar.finish()
    genres,words = list(genres),list(words)
    return (dat,genres,words)

# Load the input file with a one-hot representation for the genre labels
def load_one_hot_lyrics(fname,verbose=False):
    dat,genres,words = load(fname,verbose)
    x,y = dat[:,1],dat[:,0]
    y = np.array(pandas.get_dummies(y))

    return (x,y,genres,words)

# Load the input file with one-hot representation for genres and word count vectors
def load_bag_of_words(fname,raw=False,verbose=False):
    x,y,genres,words = load_one_hot_lyrics(fname,verbose)

    # Build dict from word -> index
    wd = {}
    num_words = len(words)
    for i in range(num_words):
        word = words[i]
        wd[word] = i

    print('about to start building song vectors')
    # Calculate word counts for each of the lyrics
    x_ = sparse.lil_matrix((x.shape[0],num_words),dtype=np.int8)
    for i in range(10000):
        lyric = x[lyric]
        for word in lyric.split():
            idx = wd[word]
            x_[i,idx] += 1

    return (x_,y,genres,words)
