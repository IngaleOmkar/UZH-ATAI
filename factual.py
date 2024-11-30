from responder import Responder
from CrowdsourceChecker import CrowdsourceChecker

class FactualResponder(Responder):
    def __init__(self, data_repository, entity_extractor, mlp_intent_classifier, emb_intent_classifier):
        super().__init__(data_repository, entity_extractor, mlp_intent_classifier)
        self.label_to_uri = self.data_repository.get_label_to_uri()
        self.triplets = self.data_repository.get_triplets()
        self.uri_to_label = self.data_repository.get_uri_to_label()

        self.emb_intent_classifier = emb_intent_classifier
        self.mlp_intent_classifier = mlp_intent_classifier

        self.crowdsource_checker = CrowdsourceChecker(self.data_repository)

    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)

        tag_mlp, uri_mlp = self.mlp_intent_classifier.classify_query(query)
        tag_emb, uri_emb = self.emb_intent_classifier.classify_query(" ".join(query.split(entities[0])))

        if(len(entities) == 0):
            raise Exception("I'm sorry, I couldn't understand the query.")
        else:
            en_uri = self.label_to_uri[entities[0]]

            crowd_answer = ""
            ent_rel_tuple = ("wd:" + str(en_uri).split('/')[-1], "wdt:" + str(tag_emb).split('/')[-1])
            print(f"exists in crowd data: {self.crowdsource_checker.exists_in_crowd_data(ent_rel_tuple)}")
            if (self.crowdsource_checker.exists_in_crowd_data(ent_rel_tuple)):
                kappa = self.crowdsource_checker.get_interrater_agreement(ent_rel_tuple)
                reject_votes, support_votes = self.crowdsource_checker.get_votes(ent_rel_tuple)
                crowd_answer = f"\n[Crowd, inter-rater agreement {kappa}, The answer distribution for this specific task was {support_votes} support vote(s), {reject_votes} reject vote(s)]"

            if(tag_mlp == tag_emb):
                # both models agree
                try:
                    if(tag_mlp in ["rating", "revenue", "budget", "release_date"] ):
                        return self.triplets[(en_uri, uri_mlp)].join(crowd_answer)
                    else:
                        ans = self.triplets[(en_uri, uri_mlp)]
                        ans_labels = [self.uri_to_label[label] for label in ans]
                        return ", ".join(ans_labels).join(crowd_answer)
                except:
                    pass
            
            
            if tag_mlp != tag_emb:
                # models disagree,  try embedding model first
                try:
                    if(tag_emb in ["rating", "revenue", "budget", "release_date"] ):
                        return self.triplets[(en_uri, uri_emb)].join(crowd_answer)
                    else:
                        ans = self.triplets[(en_uri, uri_emb)]
                        ans_labels = [self.uri_to_label[label] for label in ans]
                        return ", ".join(ans_labels)
                except:
                    pass

                # if embedding model fails, try mlp model
                try:
                    if(tag_mlp in ["rating", "revenue", "budget", "release_date"] ):
                        return self.triplets[(en_uri, uri_mlp)].join(crowd_answer)
                    else:
                        ans = self.triplets[(en_uri, uri_mlp)].join(crowd_answer)
                        ans_labels = [self.uri_to_label[label] for label in ans]
                        return ", ".join(ans_labels)
                except:
                    pass

            raise Exception("I'm sorry, I couldn't find the answer to your question.")