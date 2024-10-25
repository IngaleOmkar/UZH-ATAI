# Parent class for all responder types. Currently, only factual and embeddings are supported.
class Responder:
    def __init__(self, data_repository, entity_extractor, intent_classifier):
        self.data_repository = data_repository
        self.entity_extractor = entity_extractor
        self.intent_classifier = intent_classifier
    
    def answer_query(self, query):
        pass



