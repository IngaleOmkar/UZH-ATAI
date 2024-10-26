import spacy
from transformers import pipeline
from data_repository import DataRepository
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Extractor:
    def __init__(self, data_repository: DataRepository):

        self.data_repository = data_repository
        self.ner_embeddings = data_repository.get_ner_embeddings()
        self.ner_entities_list = data_repository.get_ner_entities_list()
        self.torch_ner_model = data_repository.get_torch_ner_model()

        # Load the NER model from Hugging Face
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        
        # Load SpaCy's dependency parser
        self.nlp = spacy.load("en_core_web_sm")

    def get_embedding_entities(self, entities):
        embedding_entities = []
        for entity in entities:
            entity_embedding = self.torch_ner_model.encode(entity)
            similarities = cosine_similarity([entity_embedding], self.ner_embeddings)[0]
            top_indices = np.argsort(similarities)[-1:][::-1]

            matches = [(list(self.ner_entities_list)[idx], similarities[idx]) for idx in top_indices]

            for identified_entity, score in matches:
                embedding_entities.append(identified_entity)
        return embedding_entities
    
    def get_guaranteed_entities(self, query):
        guaranteed_entities = []
        entities = self.extract_ner(query)
        for entity in entities:
            if entity in self.ner_entities_list:
                guaranteed_entities.append(entity)
            else:
                guaranteed_entities.append(self.get_embedding_entities([entity])[0])
        return guaranteed_entities


    def extract_ner(self, sentence):
        # Perform Named Entity Recognition
        ner_results = self.ner_pipeline(sentence)

        # Extract entity names
        entities = [entity['word'] for entity in ner_results if entity['entity_group'] in ['PER', 'LOC', 'ORG', 'MISC']]
        return entities

    def extract_predicates(self, sentence):
        # Parse the sentence with SpaCy
        doc = self.nlp(sentence)

        # Extract predicates (verbs)
        predicates = [token.lemma_ for token in doc if token.dep_ == "ROOT" or token.pos_ == "VERB"]
        return predicates

    def extract_all(self, sentence):
        # Extract entities and predicates
        entities = self.extract_ner(sentence)
        predicates = self.extract_predicates(sentence)

        return entities, predicates
