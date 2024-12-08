from sklearn.metrics import pairwise_distances
from responder import Responder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from formatter import FormatHelper

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
            return (False, "Not Found.")
        
        rel = self.lbl2rel[rel_label]

        if (rel not in self.rel2id):
            return (False, "Not Found.")

        rel_id = self.rel2id[rel]        
        rel_vec = self.relation_emb[rel_id]
        return rel_vec
    
    def get_ent_vec(self, ent_label):
        if (ent_label not in self.lbl2ent):
            return (False, "Not Found.")
        
        ent = self.lbl2ent[ent_label]

        if (ent not in self.ent2id):
            return (False, "Not Found.")
        
        ent_id = self.ent2id[ent]

        ent_vec = self.entity_emb[ent_id]
        return ent_vec
    
    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)


        tag_mlp, uri_mlp = self.intent_classifier.classify_query(query)
        tag_emb, uri_emb = self.emb_intent_classifier.classify_query(" ".join(query.split(entities[0])))


        if(len(entities) == 0):
            return (False, "I'm sorry, I couldn't understand the query.")
        else:
            # en_uri = self.label_to_uri[entities[0]]

            entity_vec = self.get_ent_vec(entities[0])

            try:
                rel_emb_emb = self.get_rel_vec(tag_emb)
                emb_sum = entity_vec + rel_emb_emb

                # find the closest entity 
                idx = np.argmin(pairwise_distances(emb_sum.reshape(1, -1), self.entity_emb))
                ans = self.ent2lbl[self.id2ent[idx]]
                print(ans)
                return (True, ans)

            except:
                try:
                    rel_emb_mlp = self.get_rel_vec(tag_mlp)
                    emb_sum = entity_vec + rel_emb_mlp

                    # find the closest entity
                    idx = np.argmin(pairwise_distances(emb_sum.reshape(1, -1), self.entity_emb))
                    ans = self.ent2lbl[self.id2ent[idx]]
                    print(ans)
                    return (True, ans)
                except:
                    return (False, "I'm sorry, I couldn't find the answer to your question.")

