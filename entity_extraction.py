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
    
    def merge_adjacent_entities(self, entities, max_gap=1, force_merge_names=True):
        if not entities:
            return []
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Handle WordPiece tokens (##)
            is_wordpiece = next_entity['word'].startswith('##')
            
            # Calculate gap between entities
            gap = next_entity['start'] - current['end']
            
            # Check if either entity group is PER or MISC
            is_name_related = (
                current['entity_group'] in ['PER', 'MISC'] and 
                next_entity['entity_group'] in ['PER', 'MISC']
            )
            
            # Conditions for merging:
            # 1. WordPiece continuation
            # 2. Close enough entities (within max_gap) AND either:
            #    a. Same entity group OR
            #    b. One entity contains the other's text OR
            #    c. Both entities are name-related (PER or MISC) if force_merge_names is True
            should_merge = (
                is_wordpiece or
                (gap <= max_gap and (
                    current['entity_group'] == next_entity['entity_group'] or
                    current['word'].lower() in next_entity['word'].lower() or
                    next_entity['word'].lower() in current['word'].lower() or
                    (force_merge_names and is_name_related)
                ))
            )
            
            if should_merge:
                # For WordPiece tokens, remove the ## prefix
                next_word = next_entity['word'].replace('##', '')
                
                # Merge entities
                current = {
                    'entity_group': 'PER' if is_name_related else current['entity_group'],  # Prefer PER for name entities
                    'word': (current['word'] + (' ' if not is_wordpiece else '') + next_word),
                    'start': current['start'],
                    'end': next_entity['end'],
                    'score': (current['score'] + next_entity['score']) / 2
                }
            else:
                merged.append(current)
                current = next_entity
                
        merged.append(current)
        return merged
    
    def get_guaranteed_entities(self, query, max_gap=2):
        guaranteed_entities = []
        entities = self.extract_ner(query)
        entities = self.merge_adjacent_entities(entities, max_gap)
        entity_names = []
        for entity in entities:
            entity_names.append(query[entity['start']:entity['end']])
        entities = entity_names
        for entity in entities:
            if entity in self.ner_entities_list:
                guaranteed_entities.append(entity)
            else:
                guaranteed_entities.append(self.get_embedding_entities([entity])[0])
        return guaranteed_entities


    def extract_ner(self, sentence):
        # Perform Named Entity Recognition
        ner_results = self.ner_pipeline(sentence)

        return ner_results

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
