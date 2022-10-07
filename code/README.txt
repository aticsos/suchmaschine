-------------------------------------------
Readme-Datei für den ReZi Question-Answerer
-------------------------------------------

Vorraussetzungen:

ElasticSearch:
python -m pip install elasticsearch

Glove-Datei:
nlp.stanford.edu/data/glove.840B.300d.zip
herunterladen und .TXT im Ordner ablegen

Spacy:
pip install -U pip setuptools wheel
pip install -U spacy

NLTK:
install --user -U nltk

NUMPY:
pip install --user -U numpy

SKLEARN:
pip3 install -U scikit-learn

PYTORCH:
pip install torch


Installationsanweisungen:

1. falls Sie das System neu trainierten möchten,
	fahren Sie mit 1.1 fort. wenn sie das bereits trainierte
	System nutzen möchten, fahren Sie mit 1.2 fort
   1.1: Erstellen von logreg.pickel und voc.pickel.
	starten Sie hierzu die fragenklassifikator.py mit einer Trainings-CSV Ihrer Wahl.
	diese muss _train.csv heißen, oder Sie ändern den Pfad der fragenklassifikator.py
   1.2: nutzen Sie die die Dateien logreg.pickel und voc.pickel
	in diesem Ordner. (Diese wurden auch am Constest-Tag genutzt)


2. Alle Dateien einen Ordner ablegen inkl. Test-csv-Dateien

3. Wenn nötig Pfadvariablen in der baseline.py für glove-,
   und csv-File anpassen

-----------------------------------------------

Ausührung des codes:

1. baseline.py ausführen

2. möglicherweise verrutschen in der CSV die zeilen, in der abgabe wurde dises problem händich gefixed,
   eine zeile ist verrutscht wenn sie mit einem semikolon anfängt


-----------------------------------------------

Codestruktur:

baseline.py:		Datei beinhaltet das Grundgerüst und den NER-Teil des Projekts
suchmachine.py:		ElasticSeach Indexierung und Suche
cos_abstand.py:		Ermittlung des Cosinus-Abstandes zwischen zwei Sätzen mithilfe von Glove-Embeddings
getsentences.py:	Extrahiert die Fragen aus der CSV-Datei
bag_of_words.py:	Bekommt Fragesätze übergeben und gibt eine Bagofwords Matrix für alle Fragen zurück

fragenklassifikator.py: Dieser Code erstellt die Dateien: voc.pickel und logreg.pickel.
voc.pickel:		Vokabular aus den Trainingsfragen
logreg.pickel:		Eine Datei welcher mit Sklearn/LogisticRegression trainiert worden ist,
			aus einer Frage den richtigen Antworttyp zu finden.


Versuch die den richtigen Antwortsatz mit einem Neuronalen Netz zu finden:
train.py:		Training des Neuronalen Netz mit Fragen und Antwortsätzen als Bag-of-Words
rezi_net.py:		Aufbau des Neuronalen Neztes



