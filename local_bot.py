# FOR LOCAL TESTING ONLY. 

from typing import List
import time
from embeddings import EmbeddingsResponder
from entity_extraction import Extractor
from factual import FactualResponder
from data_repository import DataRepository
from intent_classifier import IntentClassifier, MLPBasedIntentClassifier, EmbeddingBasedIntentClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
from embeddings import EmbeddingsResponder
from recommender import RecommendationResponder
from question_classifier import QuestionClassifier
from image import ImageResponder
from crowd import CrowdsourceResoponder

class Agent:
    def __init__(self):

        self.data_repository = DataRepository()
        
        self.mlp_intent_classifier = MLPBasedIntentClassifier(self.data_repository)
        self.emb_intent_classifier = EmbeddingBasedIntentClassifier(self.data_repository)

        self.extractor = Extractor(self.data_repository)
        self.embeddings = EmbeddingsResponder(self.data_repository, self.extractor, self.mlp_intent_classifier, self.emb_intent_classifier)
        self.factual = FactualResponder(self.data_repository, self.extractor, mlp_intent_classifier = self.mlp_intent_classifier, emb_intent_classifier = self.emb_intent_classifier)
        self.recommender = RecommendationResponder(self.data_repository, self.extractor, mlp_intent_classifier = self.mlp_intent_classifier)
        self.image = ImageResponder(self.data_repository, self.extractor, self.emb_intent_classifier)
        self.question_classifier = QuestionClassifier()
        self.crowd = CrowdsourceResoponder(self.data_repository, self.extractor, self.emb_intent_classifier)

    def answer(self, query):
        question_type = self.question_classifier.classify(query)
        print("Question type: ", question_type)

        if(question_type == "qna"):
            results = {}

            with ThreadPoolExecutor(max_workers=3) as executor:

                futures = {
                    executor.submit(self.answer_factual, query): "factual",
                    executor.submit(self.answer_embedding, query): "embedding",
                    executor.submit(self.answer_crowd, query): "crowd"
                }

                for future in as_completed(futures):
                    answer_type =  futures[future]
                    try:
                        result = future.result()
                        results[answer_type] = result
                    except:
                        results[answer_type] = "None"
                
                answer_crowd = results["crowd"]
                answer_factual = results["factual"]
                answer_embedding = results["embedding"]

            print(results)
            
            if "sorry" not in answer_crowd:
                return answer_crowd
            if "sorry" not in answer_factual:
                return answer_factual
            if "sorry" not in answer_embedding:
                return answer_embedding
            return "I am very sorry, but no answer was found."
            
        elif(question_type == "recommendation"):
            return self.answer_recommendation(query)
        elif(question_type == "image"):
            return self.answer_image(query)
        else:
            return "I am very sorry, but no answer was found."


    def answer_recommendation(self, query):
        answer_string = ""

        try:
            results = self.recommender.answer_query(query)
            answer_string += "I think you might like "
            answer_string += self.array_to_sentence(results)
            answer_string += "."
        except Exception as e:
            print(e)
            answer_string = "I am sorry, I cannot answer your question."

        print(answer_string)
        
        return answer_string
    
    def array_to_sentence(self, arr):
        if not arr:
            return ""
        elif len(arr) == 1:
            return arr[0]
        elif len(arr) == 2:
            return " and ".join(arr)
        else:
            return ", ".join(arr[:-1]) + ", and " + arr[-1]
        
    def answer_factual(self, query):
        try:
            results = self.factual.answer_query(query)
            return results
        except Exception as e:
            return "I am very sorry, but no answer was found."

    def answer_embedding(self, query):
        try:
            results = self.embeddings.answer_query(query)
            answer_string = ""
            for result in results:
                answer_string += result + " \n"
            return results
        except Exception as e:
            return "I am very sorry, but no answer was found."
    
    def answer_image(self, query):
        try:
            results = self.image.answer_query(query)
            imgs = ["image:" + x for x in results]
            return imgs
        except Exception as e:
            return "I am very sorry, but no answer was found."
        
    def answer_crowd(self, query):
        try:
            results = self.crowd.answer_query(query)
            return results
        except Exception as e:
            return "I am very sorry, but no answer was found."

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent()
    while True:
        query = input("Ask me a question: ")
        print(demo_bot.answer(query))
        print("\n")
