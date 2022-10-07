import json
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import csv
from csv import reader

es = Elasticsearch()

#############################
#### IDEXIEREN  #############
#############################
def searchmachine():
    pfad = 'wikibase.jsonl'

    lines = []

    # falls der index noch nicht exestiert lade die daten ein
    if not es.indices.exists(index="database"):  # Alle Lininen in die Liste lines[]

        mapping = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "doc_id": {
                        "type": "text",
                        "index": "false"
                    },
                    "title": {
                        "type": "text",
                        "index": "true"
                    },
                    "text": {
                        "type": "text",
                        "index": "true"
                    }
                }
            }
        }

        es.indices.create(index="database", body=mapping)

        with open(pfad) as a:  # type(a) = file
            for line in a:
                lines.append(json.loads(line))

        def indexer():                          # generator -> jede Linie vorbereiten und mit yield ausgeben
            for line in lines:
                yield {
                    "_index": "database",       # yield gibt hier json / dict zurueck
                    "text": line["text"],
                    "title": line["title"],
                    "doc_id": line["doc_id"]
                }

        helpers.bulk(es, indexer(), request_timeout=600)  # Indexieren


def frageJedeZeile(frage, doc_id,treffer):
    searchmachine()
    return2base = []
    hits = 0

    es.indices.flush()
    es.indices.refresh(index="database")

    q = {
        "query": {
            "match": {
                "text": {
                    "query": frage,
                    "operator": "or",
                    "fuzziness": 0,
                }
            }
        }
    }
    res = es.search(index='database', size=20, body=q)
    for i in range(1):
        return2base.append(res['hits']['hits'][i]["_source"]["text"])
    for i in range(1):
        if res['hits']['hits'][i]["_source"]["doc_id"] == doc_id:
            treffer = treffer + 1
    return return2base, treffer


if __name__ == "__main__":
    searchmachine()

