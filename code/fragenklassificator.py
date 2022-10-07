import nltk
import numpy
import pickle
from sklearn.linear_model import LogisticRegression
from csv import reader
import bag_of_words

all_q = []
traincsv = '_train.csv'
with open(traincsv, 'r') as read_obj:
    csv_reader = reader(read_obj, delimiter=';')
    train = list(csv_reader)
lines= len(list(csv_reader))
Y = numpy.empty([lines], dtype=int)
ps = nltk.PorterStemmer()

id = {
    "NUM:date":0,
    "HUM:ind":1,
    "LOC:other":2,
    "NUM:count":3,
    "ENTY:other":4,
    "ENTY:cremat":5,
    "HUM:gr":6,
    "LOC:country":7,
    "LOC:city":8,
    "ENTY:animal":9,
    "ENTY:food":10,
}
def Voc():
    # Erstellung des Vokabular aus den Trainingsfragen
    res = []
    
    for i in range(len(train)):
        all_q.append(train[i][1])
        Y[i] = id.get(train[i][3])
        tokens = nltk.word_tokenize(train[i][1])
        words = [word for word in tokens if word.isalpha()]
        words2 = [ps.stem(word) for word in words]
        [res.append(word) for word in words2]
    res = list(dict.fromkeys(res))
    pickel_out = open("voc.pickel", "wb")
    pickle.dump(res, pickel_out)
    pickel_out.close()

    return all_q


def Train(all_q):
    #Klassifikation mit LogisticRegression
    
    X = bag_of_words.Bag_of_words(all_q)
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    pickel_out = open("logreg.pickel", "wb")
    pickle.dump(logreg, pickel_out)
    pickel_out.close()
    print("logreg.pickel created...")


if __name__ == "__main__":
    all_q = Voc()
    Train(all_q)

