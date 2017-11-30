import csv, re, sys

pattern = re.compile('[\W_]+')
infile = 'lyrics100.csv'
outfile = 'lyrics100cleaned.csv'
if len(sys.argv) > 1:
    infile = sys.argv[1]
    outfile = infile.split('.')
    outfile = outfile[0] + 'cleaned.' + outfile[1]
print('Processing',infile)

def process_lyrics(l):
    genre = l[-2].lower()
    lyrics = l[-1]
    lyrics = ' '.join([pattern.sub('',word).lower() for word in re.split('\s+',lyrics)])
    return genre + ':' + lyrics
    # return ':'.join(l)

with open(infile) as f:
    next(f)
    data = csv.reader(f,delimiter=',',quotechar='"')
    lyrics = [process_lyrics(l) for l in data]

print('Finished Processing. Writing to',outfile)
print(len(lyrics))
with open(outfile,'w') as f:
    for l in lyrics:
        # print(l)
        f.write(l + '\n')
