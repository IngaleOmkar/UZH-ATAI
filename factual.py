from transformers import pipeline
import torch
from sklearn.model_selection import train_test_split
import rdflib
import os
import numpy as np
import csv
from transformers import BertTokenizer, BertForSequenceClassification
import json

class Factual:
    def __init__(self):
                # define some prefixes
        self.WD = rdflib.Namespace('http://www.wikidata.org/entity/')
        self.WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
        self.DDIS = rdflib.Namespace('http://ddis.ch/atai/')
        self.RDFS = rdflib.namespace.RDFS
        self.SCHEMA = rdflib.Namespace('http://schema.org/')

        base_dir = os.path.dirname(__file__)

        self.graph = rdflib.Graph().parse(os.path.join(base_dir, '..', 'data', 'ddis-movie-graph.nt'), format='turtle')

        # load the embeddings
        self.entity_emb = np.load(os.path.join(base_dir, '..', 'data', 'entity_embeds.npy'))
        self.relation_emb = np.load(os.path.join(base_dir, '..', 'data', 'relation_embeds.npy'))

        # load the dictionaries
        with open(os.path.join(base_dir, '..', 'data', 'entity_ids.del'), 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open(os.path.join(base_dir, '..', 'data', 'relation_ids.del'), 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(self.RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        wikidata_attributes = {
            "revenue": "box office",
            "director": "director",
            "actor": "cast member",
            "release_date": "publication date",
            "genre": "genre",
            "rating": "IMDb ID",
            #"budget": "budget",
            "producer": "producer",
            "screenwriter": "screenwriter",
            #"birth_date": "date of birth",
            "birth_place": "place of birth"
        }

        self.tag2uri = {
            tag:self.lbl2ent[label] for tag, label in wikidata_attributes.items()
        }

        self.model = BertForSequenceClassification.from_pretrained("models/movie_tag_model")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        with open('factual/id2tag.json') as json_file:
            self.id2label = json.load(json_file)

    def classify_query(self, query, model, tokenizer):
        encoded_dataset = tokenizer(query, padding=True, truncation=True, return_tensors="pt")

        # Perform the classification
        model.eval()
        with torch.no_grad():
            output = model(**encoded_dataset)
            predicted_label = torch.argmax(output.logits, dim=1).item()

        # Map the predicted label to the corresponding tag
        tag = self.id2label[str(predicted_label)]

        return tag

    def answer_query(self, query, entities, predicates):

        # Classify the query
        tag = self.classify_query(query, self.model, self.tokenizer)

        # Retrieve the corresponding Wikidata attribute
        uri = self.tag2uri[tag]

        if(len(entities) == 0):
            return "I'm sorry, I couldn't understand the query."
        else:
            en_uri = self.lbl2ent[entities[0]]

            # TODO: replace triplets

            # try:
            #     if(tag in ["rating", "revenue", "budget", "release_date"] ):
            #         print(triplets[(en_uri, uri)])
            #     else:
            #         ans = triplets[(en_uri, uri)]
            #         ans_labels = [uri_to_label[label] for label in ans]
            #         print(", ".join(ans_labels))
            # except:
            #     return "I'm sorry, I couldn't find the answer to your question."
        

        result = ""

        raise NotImplementedError()