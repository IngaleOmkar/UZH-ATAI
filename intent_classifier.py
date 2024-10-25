import torch
import json
from transformers import BertForSequenceClassification, BertTokenizer

class IntentClassifier:
    def __init__(self, data_repository):
        self.model = BertForSequenceClassification.from_pretrained("factual/movie_tag_model")
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
