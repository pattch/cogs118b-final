import csv, re, sys
import spacy
from spacy.lemmatizer import Lemmatizer
nlp = spacy.load('en')
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import progressbar

pattern = re.compile('[\W_]+')
infile = 'lyrics100.csv'
outfile = 'lyrics100cleaned.csv'
lemmatizer = Lemmatizer(LEMMA_INDEX,LEMMA_EXC,LEMMA_RULES)

if len(sys.argv) > 1:
    infile = sys.argv[1]
    outfile = infile.split('.')
    outfile = outfile[0] + 'cleaned.' + outfile[1]
print('Processing',infile)

with open(infile,'r',encoding='utf-8') as f:
    for i,l in enumerate(f):
        pass
word_count = i+1
bar = progressbar.ProgressBar(maxval=word_count,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

def process_lyrics(l):
    genre = l[-2].lower()
    lyrics_ = l[-1].lower()

    # Remove words that only contained garbage characters, lemmatize and remove stop words
    lyrics = ' '.join([pattern.sub('',lemmatizer.lookup(word)).lower() for word in lyrics_ if word and not nlp.vocab[word].is_stop])
    # lyrics = [pattern.sub('',token.lemma_) for token in nlp(lyrics_) if not token.is_stop]
    lyrics = ' '.join([word for word in lyrics if word])
    if lyrics:
        # Format for writing to file
        return genre + ':' + lyrics

with open(infile,'rt',encoding='utf-8') as f:
    next(f)
    data = csv.reader(f,delimiter=',',quotechar='"')
    lyrics = []
    for i,l in enumerate(data):
        pl = process_lyrics(l)
        if pl:
            lyrics.append(pl)
        if i % 1000 == 0:
            bar.update(i+1)
    bar.finish()

print('Finished Processing. Writing to',outfile)
print(len(lyrics))
with open(outfile,'w',encoding='utf-8') as f:
    for l in lyrics:
        # print(l)
        f.write(l + '\n')
