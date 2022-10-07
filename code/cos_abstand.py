import nltk
import numpy as np
import spacy
import scipy
import re
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')

def cos_abstand(model,text,frage,antwort,richtige):
    ranks = []
    res = []
    for t in text:
            saetze = nltk.sent_tokenize(t)
            for j in range(len(saetze)):
                satz = saetze[j]
                vector_1 = np.nanmean([model[word] for word in preprocess(satz, model)],axis=0)
                vector_2 = np.nanmean([model[word] for word in preprocess(frage, model)],axis=0)
                cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
                ranks.append((satz,round((1-cosine)*100,2)))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    for x in ranks:
        res.append(x[0])
        if res == 20:
            break

    for s in res:
        if s == antwort:
            richtige += 1
    return res, richtige

def preprocess(raw_text, model):
    # nur Wörter behalten
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # zu lowercase konvertieren
    words = letters_only_text.lower().split()

    # stopwords entfernen
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))
    cleaned_words = list(set([w for w in cleaned_words if w in model]))

    return cleaned_words



