# imports
import csv
import numpy as np
import os
import rdflib
from sklearn.metrics import pairwise_distances
from responder import Responder

class EmbeddingsResponder(Responder):
    def __init__(self, data_repository, entity_extractor, intent_classifier):
        super().__init__(data_repository, entity_extractor, intent_classifier)

        self.graph = self.data_repository.get_graph()
        self.entity_emb = self.data_repository.get_entity_emb()
        self.relation_emb = self.data_repository.get_relation_emb()
        self.ent2id = self.data_repository.get_ent2id()
        self.id2ent = self.data_repository.get_id2ent()
        self.rel2id = self.data_repository.get_rel2id()
        self.id2rel = self.data_repository.get_id2rel()
        self.ent2lbl = self.data_repository.get_ent2lbl()
        self.lbl2ent = self.data_repository.get_lbl2ent()
        self.rel2lbl = self.data_repository.get_rel2lbl()
        self.lbl2rel = self.data_repository.get_lbl2rel()

    
    def answer_query(self, query):
        try: 
            entities, predicates = self.entity_extractor.extract_all(query)
            ent_label, pred_label = entities[0], predicates[0]
            
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
        
        except Exception as e:
            raise Exception(e)
    
