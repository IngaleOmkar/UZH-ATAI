import spacy
from transformers import pipeline

class Extractor:
    def __init__(self):
        # Load the NER model from Hugging Face
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        
        # Load SpaCy's dependency parser
        self.nlp = spacy.load("en_core_web_sm")

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
