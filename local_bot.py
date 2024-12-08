# FOR LOCAL TESTING ONLY. 

from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from embeddings import EmbeddingsResponder
from entity_extraction import Extractor
from factual import FactualResponder
from data_repository import DataRepository
from intent_classifier import IntentClassifier, MLPBasedIntentClassifier, EmbeddingBasedIntentClassifier
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, TimeoutError, as_completed
from embeddings import EmbeddingsResponder
from recommender import RecommendationResponder
from question_classifier import QuestionClassifier
from image import ImageResponder
from crowd import CrowdsourceResoponder
from formatter import FormatHelper
from answer_wrapper import AnswerWrapper

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
        self.answer_wrapper = AnswerWrapper()

    def answer(self, query):
        question_type = self.question_classifier.classify(query)
        print("Question type: ", question_type)

        if question_type == "qna":
            results = {}

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.answer_factual, query): "factual",
                    executor.submit(self.answer_embedding, query): "embedding",
                    executor.submit(self.answer_crowd, query): "crowd"
                }

                # Wait for all futures to complete
                wait(futures.keys(), return_when=ALL_COMPLETED)

                # Process results after all threads have finished
                for future in futures:
                    answer_type = futures[future]
                    try:
                        result = future.result()
                        results[answer_type] = result
                    except Exception as e:
                        print(f"Error occurred while processing {answer_type}: {e}")
                        results[answer_type] = "None"

            # Print results for debugging
            print(results)

            answer_technique = None

            # Return results based on priority order
            try:
                for method in ["factual", "embedding"]:
                    if results.get(method) is not None and type(results[method]) is not str and results[method][0] == True:
                        print(f"results: {results[method]}")
                        llm = self.answer_wrapper.wrap_answer(query, results[method][1])
                        print(f"llm res: {llm}")
                        if llm[0]:
                            ans = llm[1].content
                        else:
                            ans = results[method][1]

                        print(f"ans: {ans}")

                        if results.get("crowd") is not None and type(results["crowd"]) is not str and results["crowd"][0] == True:
                            answer_technique = "crowd"
                            ans += " " + results["crowd"][1]

                        if answer_technique is not None:
                            ans += "\n\nThis answer is of type: " + answer_technique

                        print(ans)
                        return ans
            except Exception as e:
                return "No suitable answer was found."
                
            # If no suitable answer is found, return None or a default response
            return "No suitable answer was found."

        elif question_type == "recommendation":
            return self.answer_recommendation(query)
        elif question_type == "image":
            return self.answer_image(query)
        else:
            return "I am very sorry, but no answer was found."


    def answer_recommendation(self, query):
        answer_string = ""

        try:
            results, justification = self.recommender.answer_query(query)
            answer_string += "I think you might like "
            answer_string += FormatHelper.array_to_sentence(results)
            answer_string += ", " + justification + "."
        except Exception as e:
            print(e)
            answer_string = "I am sorry, I cannot answer your question."
        
        return answer_string
        
    def answer_factual(self, query):
        try:
            results = self.factual.answer_query(query)
            return results
        except Exception as e:
            return (False, "I am very sorry, but no answer was found.")

    def answer_embedding(self, query):
        try:
            results = self.embeddings.answer_query(query)
            return results
        except Exception as e:
            return (False, "I am very sorry, but no answer was found.")
    
    def answer_image(self, query):
        try:
            results_arr, answer_string = self.image.answer_query(query)
            # imgs = ["image:" + x for x in results]
            # print("returned image: ", imgs)
            # answer_string = ""
            # for img in imgs:
            #     answer_string += img + " \n"
            # print("returned image string: ", answer_string)
            return answer_string
        except Exception as e:
            return "I am very sorry, but no answer was found."
        
    def answer_crowd(self, query):
        try:
            results = self.crowd.answer_query(query)
            return results
        except Exception as e:
            return (False, "I am very sorry, but no answer was found.")

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent()
    while True:
        query = input("Ask me a question: ")
        print(demo_bot.answer(query))
        print("\n")
