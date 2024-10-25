# imports
import csv
import numpy as np
import os
import rdflib
from sklearn.metrics import pairwise_distances

class Embeddings:
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

        self.rel2lbl = {rel: str(lbl) for rel, lbl in self.graph.subject_objects(self.RDFS.label) if str(rel).startswith(self.WDT)}
        self.lbl2rel = {lbl: rel for rel, lbl in self.rel2lbl.items()}

    def find(self, ent_label, pred_label):
        pred_label = pred_label.lower()
        print(f"ent label: {ent_label}, pred label: {pred_label}")

        ent = self.lbl2ent[ent_label]
        pred = self.lbl2rel[pred_label]

        print(f"ent: {ent}, pred: {pred}")

        ent_id = self.ent2id[ent]
        rel_id = self.rel2id[pred]

        print(f"ent id: {ent_id}")
        print(f"rel id: {rel_id}")

        ent_vec = self.entity_emb[ent_id]
        pred_vec = self.relation_emb[rel_id]

        print(f"ent_vec: {ent_vec}, pred_vec: {pred_vec}")

        lhs = ent_vec + pred_vec
        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
        most_likely = dist.argsort()

        most_likely = most_likely[:3]

        print(most_likely)

        most_likely_labels = []

        for x in most_likely:
            print(f"in id to ent: {x in self.id2ent}")
            print(f"in id to rel: {x in self.id2rel}")

            if (x in self.id2ent):
                most_likely_labels.append(self.ent2lbl[self.id2ent[x]])
            elif (x in self.id2rel):
                most_likely_labels.append(self.rel2lbl[self.id2rel[x]])
            else:
                raise KeyError("Not found.")

        print(most_likely_labels)

        return most_likely_labels
    
