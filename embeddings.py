# imports
import numpy as np
from sklearn.metrics import pairwise_distances
from responder import Responder
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import re

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
        self.intent_classifier = intent_classifier
        self.extractor = entity_extractor

        self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rel_lbl_emb = self.data_repository.get_rel_lbl_emb()

    def get_rel_vec(self, rel_label):
        rel_label = rel_label.lower()
        if (rel_label not in self.lbl2rel):
            raise KeyError("Label not found.")
        
        rel = self.lbl2rel[rel_label]
        rel_id = self.rel2id[rel]        
        rel_vec = self.relation_emb[rel_id]

        return rel_vec

    def get_ent_vec(self, ent_label):
        if (ent_label not in self.lbl2ent):
            raise KeyError("Label not found.")
        
        ent = self.lbl2ent[ent_label]
        ent_id = self.ent2id[ent]
        ent_vec = self.entity_emb[ent_id]

        return ent_vec
    
    def get_nlp_embedding(self, text):
        return self.nlp_model.encode(text)

    
    def find_closest_key(self, tag):
        tag_embedding = self.get_nlp_embedding(tag)
        closest_key = None
        closest_similarity = float("inf")
        
        for key, key_embedding in self.rel_lbl_emb.items():
            similarity = cosine(tag_embedding, key_embedding)
            if similarity < closest_similarity:
                closest_similarity = similarity
                closest_key = key
        
        return closest_key, 1 - closest_similarity

    def answer_query(self, query):
        try: 

            entities, predicates = self.entity_extractor.extract_all(query)

            vectors = []

            dict_lbl, dist = self.find_closest_key(predicates[0])
            print(f"key: {dict_lbl}")

            rel_vec = self.get_rel_vec(dict_lbl)

            vectors.append(rel_vec)

            for ent in entities:
                try:
                    vectors.append(self.get_ent_vec(ent))
                    print(f"entity vector found for {ent}")
                except Exception as e:
                    print(f"entity {ent} not found, {e}")
        
            
        
            # ent_label, pred_label = entities[0], predicates[0]
            
            # pred_label = pred_label.lower()
            # print(f"ent label: {ent_label}, pred label: {pred_label}")

            # ent = self.lbl2ent[ent_label]
            # pred = self.lbl2rel[pred_label]

            # print(f"ent: {ent}, pred: {pred}")

            # ent_id = self.ent2id[ent]
            # rel_id = self.rel2id[pred]

            # print(f"ent id: {ent_id}")
            # print(f"rel id: {rel_id}")

            # ent_vec = self.entity_emb[ent_id]
            # pred_vec = self.relation_emb[rel_id]

            # print(f"ent_vec: {ent_vec}, pred_vec: {pred_vec}")

            lhs = np.sum(vectors, axis=0)
            # lhs = ent_vec + pred_vec
            dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
            most_likely = dist.argsort()

            most_likely = most_likely[:3]

            print(most_likely)

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
    
