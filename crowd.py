from responder import Responder
from intent_classifier import EmbeddingBasedIntentClassifier

class CrowdsourceResoponder(Responder):
    def __init__(self, data_repository, entity_extractor, intent_classifier):
        super().__init__(data_repository, entity_extractor, intent_classifier)
        self.data_repository = data_repository
        self.answer_dict = {}
        self.label_to_uri = self.data_repository.get_label_to_uri()
        self.uri_to_label = self.data_repository.get_uri_to_label()

        self.crowd_data = data_repository.crowd_data

        # new intent classifier with additional categories
        self.intent_classifier = EmbeddingBasedIntentClassifier(data_repository)

        for batch_id, batch in self.crowd_data.items():
            for task_id, task in batch['tasks'].items():
                ent_rel_tuple = (task['Entity'], task['Relation'])
                self.answer_dict[ent_rel_tuple] = {}
                self.answer_dict[ent_rel_tuple]['Result'] = task['Majority Element'] == 1.0
                self.answer_dict[ent_rel_tuple]['Support Votes'] = task['Votes'][0]
                self.answer_dict[ent_rel_tuple]['Reject Votes'] = task['Votes'][1]       
                self.answer_dict[ent_rel_tuple]['Kappa'] = batch['Kappa'] 
                self.answer_dict[ent_rel_tuple]['Answer'] = task['Answer']           

    def exists_in_crowd_data(self, entity, relation):
        return ((entity, relation) in self.answer_dict)

    def get_interrater_agreement(self, entity, relation):
        return self.answer_dict[(entity, relation)]['Kappa']

    def get_votes(self, entity, relation):
        return self.answer_dict[(entity, relation)]['Reject Votes'], self.answer_dict[(entity, relation)]['Support Votes']
    
    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)

        if (len(entities) == 0):
            return (False, "Sorry, I could not find an answer")
        
        try: 
            entity = self.label_to_uri[entities[0]]
            tag_mlp, uri_mlp = self.intent_classifier.classify_query(query)

            relation = "wdt:" + uri_mlp.split("/")[-1]
            entity =  "wd:" + entity.split("/")[-1] ## check for ddis here!
            print(f"crowd entity: {entity}, relation: {relation}")

            if ((entity, relation) in self.answer_dict):
                answer = self.answer_dict[(entity, relation)]['Answer']
                parts = answer.split(":")
                if (len(parts) == 2 and parts[0] == "wd"):
                    uri = f"http://www.wikidata.org/entity/{parts[1]}"
                    if (uri not in self.uri_to_label):
                        return (False, "Sorry, coulnd't find the answer.")
                    label = self.uri_to_label[uri]
                    return  (True, f"The answer is {label}.\n [Crowd, inter-rater agreement {self.answer_dict[(entity, relation)]['Kappa']}, The answer distribution for this specific task was {self.answer_dict[(entity, relation)]['Support Votes']} support vote(s), {self.answer_dict[(entity, relation)]['Reject Votes']} reject vote(s)]")
                else:
                    return (True, f"The answer is {self.answer_dict[(entity, relation)]['Answer']}.\n [Crowd, inter-rater agreement {self.answer_dict[(entity, relation)]['Kappa']}, The answer distribution for this specific task was {self.answer_dict[(entity, relation)]['Support Votes']} support vote(s), {self.answer_dict[(entity, relation)]['Reject Votes']} reject vote(s)]")
            else:
                return (False, "Sorry, I could not find an answer")
        except Exception as e:
            return (False, "Sorry, I could not find an answer")