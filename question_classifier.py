import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class QuestionClassifier():

    def __init__(self):
        self.model = None
        self.model = BertForSequenceClassification.from_pretrained("data/final_classification_model")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.id2tag = {0: 'qna', 1: 'recommendation', 2: 'image'}
    
    def classify(self, query)->str:
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model(**inputs)
        predicted_label = torch.argmax(outputs.logits[0]).item()
        return self.id2tag[predicted_label]