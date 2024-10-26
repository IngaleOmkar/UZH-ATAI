# imports
import numpy as np
from sklearn.metrics import pairwise_distances
from responder import Responder
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import re

class EmbeddingsResponder(Responder):
    def __init__(self, data_repository, entity_extractor, intent_classifier, emb_intent_classifier):
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
        self.intent_classifier = intent_classifier
        self.extractor = entity_extractor
        self.emb_intent_classifier = emb_intent_classifier

    def get_rel_vec(self, rel_label):
        if (rel_label not in self.lbl2rel):
            raise Exception("Not found.")
        
        rel = self.lbl2rel[rel_label]

        if (rel not in self.rel2id):
            raise Exception("Not found.")

        rel_id = self.rel2id[rel]        
        rel_vec = self.relation_emb[rel_id]
        return rel_vec

    def get_ent_vec(self, ent_label):
        if (ent_label not in self.lbl2ent):
            raise Exception("Not found.")
        
        ent = self.lbl2ent[ent_label]

        if (ent not in self.ent2id):
            raise Exception("Not found.")
        
        ent_id = self.ent2id[ent]

        ent_vec = self.entity_emb[ent_id]
        return ent_vec

    def answer_query(self, query):
        try: 

            entities, predicates = self.entity_extractor.extract_all(query)

            vectors = []

            for ent in entities:
                query = query.replace(ent, "")

            rel_emb, dist_emb = self.emb_intent_classifier.classify_query(query)
            rel_mlp, dist_mlp = self.intent_classifier.classify_query(query)

            try:
                rel_mlp_vec = self.get_rel_vec(rel_mlp)
            except Exception as e:
                dist_mlp = float("inf")

            try:
                rel_emb_vec = self.get_rel_vec(rel_emb)
            except Exception as e:
                dist_emb = float("inf")

            if (dist_mlp != float("inf") or dist_emb != float("inf")):
                if (dist_emb < dist_mlp):
                    vectors.append(rel_emb_vec)
                else:
                    vectors.append(rel_mlp_vec)

            
            if (len(entities) == 0):
                raise Exception("Answer not found.")
            
            ent_vec = self.get_ent_vec(entities[0])

            vectors.append(ent_vec)

            lhs = sum(vectors)
            dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
            most_likely = dist.argsort()

            most_likely = most_likely[:3]

            most_likely_labels = []

            for x in most_likely:
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
    
