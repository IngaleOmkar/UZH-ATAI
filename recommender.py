from sklearn.metrics import pairwise_distances
from responder import Responder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Recommender():
        
    def __init__(self, data_repository, entity_extractor):
        self.data_repository = data_repository
        self.entity_extractor = entity_extractor
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

    def get_ent_vec(self, ent_label):
        if (ent_label not in self.lbl2ent):
            raise Exception("Label not found.")
        
        ent = self.lbl2ent[ent_label]

        if (ent not in self.ent2id):
            raise Exception("Entity not found.")
        
        ent_id = self.ent2id[ent]

        ent_vec = self.entity_emb[ent_id]
        return ent_vec

    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)

        if(len(entities) == 0):
            raise Exception("I'm sorry, I couldn't understand the query.")
        else:
            entity_vecs = []
            for ent in entities:
                try: 
                    entity_vecs.append(self.get_ent_vec(ent))
                except Exception as e:
                    print(f"entity {ent} could not be found in embedding: {e}")
                    

            entity_vec = np.mean(entity_vecs, axis=0)

            try:
                distances = pairwise_distances(entity_vec.reshape(1, -1), self.entity_emb)
                closest_indices = np.argsort(distances[0])[:20]
                ents = [self.ent2lbl[self.id2ent[id]] for id in closest_indices]
                print(ents)

                final_ents = []

                for ent in ents:
                    if ent not in entities:
                        final_ents.append(ent)

                print(final_ents)

                if len(final_ents) >= 3:
                    return final_ents[:3]
                else:
                    return final_ents
            except:
                raise Exception("I'm sorry, I couldn't find the answer to your question.")
