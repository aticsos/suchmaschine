import suchmachine
import spacy
import numpy
import cos_abstand
import getsentences
import wandb
import pickle
import bag_of_words
import csv


def baseline():
    labels = {
        0: "DATE",
        1: "PERSON",
        2: "LOC",
        3: "QUANTITY",
        4: "other",
        5: "WORK_OF_ART",
        6: "ORG",
        7: "NORP",
        8: "GPE",
        9: "Animal",
        10: "Food"
    }
    '''
    wandb.init(project="rezi_net", entity="rezi")
    wandb.config.update({
        "Daten satz": "valid",
        "methode": "cosinus",
        "texte aus elesticsearch": 1,
        "embdings methode": "gloVe",
        "gloVe datei": 2200000,
        "eleasticsearch quarry typ": "multi_match",
    })
    '''
    nlp = spacy.load("en_core_web_sm")
    gloveFile = "glove.840B.300d.txt"
    fragencsv = 'test_all_blacked.csv'
    antwort_satze = []
    csvfile = open('ReZi1.csv', 'w')
    res = []
    richtige = 0
    teilantwort = ""
    hits = 0
    total_anwser_length = 0
    print("Loading Glove Model")
    model = {}
    f = open(gloveFile, "r", encoding="utf8")
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = numpy.asarray(values[-300:], dtype='float32')
        model[word] = coefs
    f.close()

    print("Done.", len(model), " words loaded!")
    fragen, antworsaetze, doc_ids, antwort = getsentences.getsent(fragencsv)

    bow = bag_of_words.Bag_of_words(fragen)
    with open("logreg.pickel", "rb") as pickle_net:
        klassificator = pickle.load(pickle_net)
    y_pred = klassificator.predict(bow)

    # Iteration durch die Fragen des CSV
    # Rückgabe der elasticseach Artikel
    # Ermittlung des Cosinusabstandes zu den Antwortsätzen
    # Ranking der Antwortsätze
    
    for x in range(len(fragen)):
        frage = fragen[x]
        antwortsatz = antworsaetze[x]
        text, hits = suchmachine.frageJedeZeile(frage, doc_ids[x], hits)
        antwort_satze, richtige = cos_abstand.cos_abstand(model, text, frage, antwortsatz, richtige)

        # Zuordnung zu den entsprechenden NERs
        label = labels[y_pred[x]]
        for antwort_satz in antwort_satze:
            antwort_satz = nlp(antwort_satz)
            if label == "Animal":
                for token in antwort_satz:
                    if len(res) == 10:
                        continue
                    if (token.pos_ == "NOUN" or token.pos_ == "PROPN") \
                        and (token.tag_ == "NNS" or token.tag_ == "NNP") \
                        and token.text not in res:
                        res.append(token.text)
            elif label == "Food":
                for token in antwort_satz:
                    if len(res) == 10:
                        continue
                    if token.pos_ == "NOUN" and token.tag_ == "NN" and token.text not in res:
                        res.append(token.text)
                        print(token.text, token.pos_, token.tag_)
            elif label == "other":
                for ent in antwort_satz.ents:
                    if len(res) == 10:
                        continue
                    if ent.text not in res:
                        res.append(ent.text)
            else:
                for ent in antwort_satz.ents:
                    if len(res) == 10:
                        continue
                    if ent.label_ == label and ent.text not in res:
                        res.append(ent.text)

        total_anwser_length += len(res)

        # Speichern der Ermittelten Antworten in die CSV

        if len(res) == 10:
            for t in res:
                teilantwort += t + ";"
            csvfile.write(teilantwort)
            csvfile.write('\n')
        else:
            #falls wir keine passenden Antwortgefunden haben
            #nehmen wir einfach die 10 ersten NER
            for answers in antwort_satze:
                answers = nlp(answers)
                for ent in answers.ents:
                    if len(res) == 10:
                        continue
                    elif ent.text not in res:
                        res.append(ent.text)
            for t in res:
                teilantwort += t + ";"
            csvfile.write(teilantwort)
            csvfile.write('\n')
        teilantwort = ""
        res = []

        '''wandb.log({"tmp_richtige": richtige / (x + 1)})
    wandb.log({"richtige_sätze": 100 * (richtige / len(fragen))})
    wandb.log({"elesticsearch_hits": hits})
    wandb.log({"avarge answer length": total_anwser_length/len(fragen)})
    wandb.finish()'''
    print("finally finished")




if __name__ == "__main__":
    baseline()
