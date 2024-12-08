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

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password):

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

        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # Send a message to the corresponding chat room using the post_messages method of the room object.
                    try:
                        answer = self.answer(message.message)
                        answer = answer.encode('latin-1', errors='replace').decode('latin-1')
                        print(answer)
                        room.post_messages(answer)

                    except Exception as e:
                        room.post_messages(f"An error occurred: {e}")
                        print(e)

                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

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

            # Return results based on priority order
            try:
                for method in ["factual", "embedding"]:
                    if results.get(method) is not None and type(results[method]) is not str and results[method][0] == True:
                        print(f"results: {results[method]}")
                        if type(results[method][1]) is list or type(results[method][1]) is tuple:
                            list_ans = results[method][1]
                            intermidiate_ans = ""
                            for i in range(len(list_ans)):
                                intermidiate_ans += list_ans[i] + ", "
                            llm = self.answer_wrapper.wrap_answer(query, intermidiate_ans)
                        else: 
                            llm = self.answer_wrapper.wrap_answer(query, results[method][1])
                        print(f"llm res: {llm}")
                        if llm[0]:
                            ans = llm[1].content
                        else:
                            ans = results[method][1]

                        print(f"ans: {ans}")

                        if results.get("crowd") is not None and type(results["crowd"]) is not str and results["crowd"][0] == True:
                            ans += " " + results["crowd"][1]

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
            if results[0]:
                answer_string = results[1] 
            print("returned fatcual: ", answer_string)
            return (True, answer_string)
        except Exception as e:
            return (False, "I am very sorry, but no answer was found.")

    def answer_embedding(self, query):
        try:
            results = self.embeddings.answer_query(query)
            answer_string = results[1]
            print("returned embedding: ", answer_string)
            return (True, answer_string)
        except Exception as e:
            return (False, "I am very sorry, but no answer was found.")
    
    def answer_image(self, query):
        try:
            results = self.image.answer_query(query)
            imgs = ["image:" + x for x in results]
            print("returned image: ", imgs)
            answer_string = ""
            for img in imgs:
                answer_string += img + " \n"
            print("returned image string: ", answer_string)
            return answer_string
        except Exception as e:
            return "I am very sorry, but no answer was found."
        
    def answer_crowd(self, query):
        try:
            results = self.crowd.answer_query(query)
            if(results[0]):
                answer_string = results[1] 
                print("returned crowd: ", answer_string)
                return (True, answer_string)
        except Exception as e:
            return (False, "I am very sorry, but no answer was found.")

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("sharp-cloud", "Q1jeQ8I2")
    demo_bot.listen()
