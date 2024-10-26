from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from embeddings import EmbeddingsResponder
from entity_extraction import Extractor
from factual import FactualResponder
from data_repository import DataRepository
from intent_classifier import IntentClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password):

        self.data_repository = DataRepository()
        self.intent_classifier = IntentClassifier(self.data_repository)
        self.extractor = Extractor(self.data_repository)
        self.embeddings = EmbeddingsResponder(self.data_repository, self.extractor, self.intent_classifier)
        self.factual = FactualResponder(self.data_repository, self.extractor, self.intent_classifier)

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
        results = {}
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

        answer_string = ""

        if answer_factual != "None":
            answer_string += f"I think the answer is {answer_factual} (factual)"
        if answer_embedding != "None":
            answer_string += f"I think the answer is {answer_embedding} (embedding)"
        
        return answer_string

        
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
