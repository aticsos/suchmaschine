import pickle
import nltk
import numpy


def Bag_of_words(fragen):
    ps = nltk.PorterStemmer()
    with open("voc.pickel", "rb") as pickle_voc:
        voc = pickle.load(pickle_voc)
    V = numpy.zeros((len(fragen), len(voc)), dtype=int)
    for x in range(len(fragen)):
        tokens = nltk.word_tokenize(fragen[x])
        words = [word for word in tokens if word.isalpha()]
        words2 = [ps.stem(word) for word in words]
        for y in range(len(voc)): 
            for w in words2:
                if w == voc[y]:
                    V[x][y] = 1
    return V
