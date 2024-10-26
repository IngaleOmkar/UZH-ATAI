from responder import Responder
import re
from intent_classifier import EmbeddingBasedIntentClassifier

class FactualResponder(Responder):
    def __init__(self, data_repository, entity_extractor, intent_classifier):
        super().__init__(data_repository, entity_extractor, intent_classifier)
        self.label_to_uri = self.data_repository.get_label_to_uri()
        self.triplets = self.data_repository.get_triplets()
        self.uri_to_label = self.data_repository.get_uri_to_label()

        self.split = False

        if(type(self.intent_classifier) == EmbeddingBasedIntentClassifier):
            self.split = True


    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)

        if(not self.split):
            # Classify the query
            tag, uri = self.intent_classifier.classify_query(query)
        else:
            tag, uri = self.intent_classifier.classify_query(" ".join(query.split(entities[0])))
            print(tag, uri)

        if(len(entities) == 0):
            raise Exception("I'm sorry, I couldn't understand the query.")
        else:
            en_uri = self.label_to_uri[entities[0]]
            try:
                if(tag in ["rating", "revenue", "budget", "release_date"] ):
                   return self.triplets[(en_uri, uri)]
                else:
                    ans = self.triplets[(en_uri, uri)]
                    ans_labels = [self.uri_to_label[label] for label in ans]
                    return ", ".join(ans_labels)
            except:
                raise Exception("I'm sorry, I couldn't find the answer to your question.")
