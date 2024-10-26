import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class IntentClassifier:
    def __init__(self, data_repository):
        self.data_repository = data_repository

    def classify_query(self, query):
        pass


class MLPBasedIntentClassifier(IntentClassifier):
    def __init__(self, data_repository):
        super().__init__(data_repository)
        self.model = BertForSequenceClassification.from_pretrained("data/movie_tag_model")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.data_repository = data_repository

        wikidata_attributes = {
            "revenue": "box office",
            "director": "director",
            "actor": "cast member",
            "release_date": "publication date",
            "genre": "genre",
            "rating": "IMDb ID",
            #"budget": "budget",
            "producer": "producer",
            "screenwriter": "screenwriter",
            #"birth_date": "date of birth",
            "birth_place": "place of birth"
        }

        self.tag2uri = {
            tag:self.data_repository.get_label_to_uri()[label] for tag, label in wikidata_attributes.items()
        }

        self.id2label = self.data_repository.get_id2label()

    def classify_query(self, query):
        encoded_dataset = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")

        # Perform the classification
        self.model.eval()
        with torch.no_grad():
            output = self.model(**encoded_dataset)
            predicted_label = torch.argmax(output.logits, dim=1).item()

        # Map the predicted label to the corresponding tag
        tag = self.id2label[str(predicted_label)]

        return tag, self.tag2uri[tag]

class EmbeddingBasedIntentClassifier(IntentClassifier):
    def __init__(self, data_repository):
        """
        Initialize the classifier with categories and load the embedding model.
        
        Args:
            categories: List of category labels
            model_name: Name of the sentence-transformers model to use
        """
        super().__init__(data_repository)
        self.categories = list(data_repository.get_rel2lbl().values())
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Pre-compute embeddings for all categories
        self.category_embeddings = self.model.encode(self.categories)

        self.label2uri = data_repository.get_label_to_uri()
        
    def classify_query(self, query: str):
        """
        Classify the input query and return top_k most similar categories with scores.
        
        Args:
            query: Input text to classify
            top_k: Number of top matches to return
            
        Returns:
            List of tuples containing (category, similarity_score)
        """
        # Get embedding for the query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity with all category embeddings
        similarities = np.dot(self.category_embeddings, query_embedding) / (
            np.linalg.norm(self.category_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top_k matches
        top_indices = np.argsort(similarities)[::-1][:1]
        
        possibilities = [(self.categories[idx], similarities[idx]) for idx in top_indices]
        return possibilities[0][0], self.label2uri[possibilities[0][0]]
