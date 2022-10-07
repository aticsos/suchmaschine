from _csv import reader
import suchmachine
import voc              # Datei wurde nicht mitgeschickt, da train.py eine Sackgasse war...
import bag_of_words     # Datei wurde nicht mitgeschickt, da train.py eine Sackgasse war...
import nltk
import torch
import rezi_net
import numpy
from difflib import SequenceMatcher

def train(net):
    # Training des Neuronalen Netz mit Fragen und Antwortsätzen als Bag-of-Words

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # Klassengewichte / class weigths
    loss = torch.nn.CrossEntropyLoss()
    target = torch.zeros(1)
    # nur eine voc bauen da sonst mehere wörter meherfach drankommen
    voc_fragen, fragen =  voc.Voc("q")
    voc_antworten, antworten = voc.Voc("a")
    # fragen randomizieren
    for epoch in range(10):
        batch = makebatch(fragen,50)
        for x in batch:
            f = fragen[x]
            text = suchmachine.searchmachine(f)
            fb = bag_of_words.bag_of_words_frage(voc_fragen, f)
            for t in text:
                saetze = nltk.sent_tokenize(t)
                for s in saetze:
                    # workaround damit Sätze mit Quotes oder anderen Sonderzeichen, ignoriert werden
                    if SequenceMatcher(None, s, antworten[x]).ratio() >= 0.7:
                        target[0] = 1
                    else:
                        target[0] = 0

                    sb = bag_of_words.bag_of_words_satz(voc_antworten, s)
                    input1 = numpy.concatenate([fb, sb], axis=1)
                    input1 = torch.from_numpy(input1)
                    y_ = net(input1.float())
                    output = loss(y_, target.long()) #target berechnen
                    output.backward()
                    optimizer.step()
                    print("loss:",output)
        print("epoche:", epoch)

        val(net,voc_fragen,voc_antworten)

def val(net, f_voc, a_voc):
    # Validate

    validcsv = '_valid.csv'

    fragen = []
    antworten = []
    target = torch.zeros(1)
    loss = torch.nn.CrossEntropyLoss()

    with open(validcsv, 'r') as read_obj:  # CSV
        csv_reader = reader(read_obj, delimiter=';')
        train = list(csv_reader)

    for i in range(len(train)):
        fragen.append(train[i][1])
        antworten.append(train[i][5])
    for x in range(len(fragen)):
        f = fragen[x]
        text = suchmachine.searchmachine(f)
        fb = bag_of_words.bag_of_words_frage(f_voc, f)
        for t in text:

            saetze = nltk.sent_tokenize(t)
            for s in saetze:
                if SequenceMatcher(None, s, antworten[x]).ratio() >= 0.7:
                    target[0] = 1
                else:
                    target[0] = 0

                sb = bag_of_words.bag_of_words_satz(a_voc, s)
                input1 = numpy.concatenate([fb, sb], axis=1)
                input1 = torch.from_numpy(input1)
                y_ = net(input1.float())
                output = loss(y_,target.long())
                answers = torch.argmax(y_, dim=1)  # answers = schätzung
                print("output:",loss)
        print(x)


def makebatch(fragen, batchsize):
    tmp = []
    while 1:
        x = numpy.random.randint(len(fragen))
        if x not in tmp:
            tmp.append(x)
            if len(tmp) >= batchsize:
                break
    return tmp


if __name__ == "__main__":
    net = rezi_net.rezi_net()
    train(net)