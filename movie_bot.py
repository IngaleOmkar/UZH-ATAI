from speakeasypy import Speakeasy, Chatroom
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
        images = self.image.answer_query(query)
        if images:
            return "image:" + images[0]
        else:
            return "NF"
            crowd_result = self.crowd.answer_query(query)
            if ("sorry" not in crowd_result.lower()):
                return crowd_result
            
            print("continue with factual and embedding.")

            with ThreadPoolExecutor(max_workers=2) as executor:

                futures = {
                    executor.submit(self.answer_factual, query): "factual",
                    executor.submit(self.answer_embedding, query): "embedding"
                }

                for future in as_completed(futures):
                    answer_type =  futures[future]
                    try:
                        result = future.result()
                        results[answer_type] = result
                    except:
                        results[answer_type] = "None"
            
            answer_factual = results["factual"]
            answer_embedding = results["embedding"]

            print(f"answer factual: {answer_factual}")
            print(f"answer embedding: {answer_embedding}")

            answer_string = ""

            if "very sorry" not in answer_factual:
                # encode the answer in utf-8 to avoid encoding issues
                if(type(answer_factual) == list):
                    answer_factual = " ".join(answer_factual)
                # answer_factual = answer_factual.encode('utf-8')
                answer_string += f"I think the answer is {answer_factual} (factual)"
            elif "very sorry" not in answer_embedding:
                answer_string += f"I think the answer is {answer_embedding} (embedding)"
            else:
                answer_string = "No answer was found."
            
        #     return answer_string

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
            return results
        except Exception as e:
            return "I am very sorry, but no answer was found."

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("sharp-cloud", "Q1jeQ8I2")
    demo_bot.listen()
