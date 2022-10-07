from _csv import reader
def getsent(datei):
    frage = []
    antwortsatz = []
    id = []
    antwort = []

    with open(datei, 'r') as read_obj:  # CSV
        csv_reader = reader(read_obj, delimiter=';')
        train = list(csv_reader)
    for i in range(len(train)):
        frage.append(train[i][1])
        antwortsatz.append(train[i][5])
        id.append((train[i][4]))
        antwort.append((train[i][3]))
    return frage, antwortsatz, id, antwort