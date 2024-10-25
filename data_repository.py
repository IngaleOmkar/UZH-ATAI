import pickle
import json
import numpy as np
import os
import rdflib
import csv

class DataRepository:
    def __init__(self):

        # define some prefixes
        self.WD = rdflib.Namespace('http://www.wikidata.org/entity/')
        self.WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
        self.DDIS = rdflib.Namespace('http://ddis.ch/atai/')
        self.RDFS = rdflib.namespace.RDFS
        self.SCHEMA = rdflib.Namespace('http://schema.org/')

        print("========== Initializing Data Repository ==========")

        # Loading graph
        print("========== Loading graph ==========")
        self.graph = rdflib.Graph().parse('14_graph.nt', format='turtle')

        print("========== Loading data for factual QA ==========")
        self.triplets, self.uri_to_label, self.label_to_uri, self.label_list = pickle.load(open("factual/formatted_data.pkl", "rb"))
        with open('factual/id2tag.json') as json_file:
            self.id2label = json.load(json_file)

        print("========== Loading data for embeddings ==========")
        self.entity_emb = np.load('Embeddings/ddis-graph-embeddings/entity_embeds.npy')
        self.relation_emb = np.load('Embeddings/ddis-graph-embeddings/relation_embeds.npy')

        with open('Embeddings/ddis-graph-embeddings/entity_ids.del', 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open('Embeddings/ddis-graph-embeddings/relation_ids.del', 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(self.RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        self.rel2lbl = {rel: str(lbl) for rel, lbl in self.graph.subject_objects(self.RDFS.label) if str(rel).startswith(self.WDT)}
        self.lbl2rel = {lbl: rel for rel, lbl in self.rel2lbl.items()}

        print("========== Data Repository initialized ==========")

    def get_graph(self):
        return self.graph
    
    def get_triplets(self):
        return self.triplets

    def get_uri_to_label(self):
        return self.uri_to_label
    
    def get_label_to_uri(self):
        return self.label_to_uri
    
    def get_label_list(self):
        return self.label_list

    def get_id2label(self):
        return self.id2label
    
    def get_entity_emb(self):
        return self.entity_emb
    
    def get_relation_emb(self):
        return self.relation_emb

    def get_ent2id(self):
        return self.ent2id
    
    def get_id2ent(self):
        return self.id2ent

    def get_rel2id(self):
        return self.rel2id
    
    def get_id2rel(self):
        return self.id2rel

    def get_ent2lbl(self):
        return self.ent2lbl
    
    def get_lbl2ent(self):
        return self.lbl2ent

    def get_rel2lbl(self):
        return self.rel2lbl
    
    def get_lbl2rel(self):
        return self.lbl2rel
