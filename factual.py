from responder import Responder
from formatter import FormatHelper


class FactualResponder(Responder):
    def __init__(self, data_repository, entity_extractor, mlp_intent_classifier, emb_intent_classifier):
        super().__init__(data_repository, entity_extractor, mlp_intent_classifier)
        self.label_to_uri = self.data_repository.get_label_to_uri()
        self.triplets = self.data_repository.get_triplets()
        self.uri_to_label = self.data_repository.get_uri_to_label()

        self.emb_intent_classifier = emb_intent_classifier
        self.mlp_intent_classifier = mlp_intent_classifier


    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)

        tag_mlp, uri_mlp = self.mlp_intent_classifier.classify_query(query)
        tag_emb, uri_emb = self.emb_intent_classifier.classify_query(" ".join(query.split(entities[0])))

        print("MLP: ", tag_mlp, uri_mlp)
        print("EMB: ", tag_emb, uri_emb)

        if(len(entities) == 0):
            return (False, "I'm sorry, I couldn't understand the query.")
        else:
            en_uri = self.label_to_uri[entities[0]]
            print("Entities: ", entities)

            
            # both models agree
            try:
                ans = self.triplets[(en_uri, uri_emb)]
                print("ans", ans)
                print(type(ans))
                if type(ans) == str:
                    if ans in self.uri_to_label:
                        return (True, self.uri_to_label[ans])
                    return (True, ans)
                else:
                    answer_string = ""
                    for uri in ans:
                        if uri in self.uri_to_label:
                            print(answer_string)
                            answer_string += self.uri_to_label[uri] + ", "
                        else:
                            answer_string += uri + ", "
                    return (True, answer_string[:-2])
            except:
                pass

            # if embedding model fails, try mlp model
            try:
                ans = self.triplets[(en_uri, uri_mlp)]
                print("ans", ans)
                print(type(ans))
                if type(ans) == str:
                    if ans in self.uri_to_label:
                        return (True, self.uri_to_label[ans])
                    return (True, ans)
                else:
                    answer_string = ""
                    for uri in ans:
                        if uri in self.uri_to_label:
                            print(answer_string)
                            answer_string += self.uri_to_label[uri] + ", "
                        else:
                            answer_string += uri + ", "
                    return (True, answer_string[:-2])
            except:
                pass

            return (False, "I'm sorry, I couldn't find the answer to your question.")