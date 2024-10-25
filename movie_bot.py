from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from embeddings import Embeddings
from entity_extraction import Extractor
from factual import Factual

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password):

        self.extractor = Extractor()
        self.embeddings = Embeddings()
        self.factual = Factual()

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

    def answer(self, nl_query):
        entities, predicates = self.extractor.extract_all(nl_query)
        print(entities, predicates)

        try:
            result = self.answer_factual(nl_query, entities, predicates)
            return f"The answer is: {result}"
        except Exception as e:
            try:
                results = self.answer_embedding(entities, predicates)
                return f"The 3 most likely answers inferred by the embedding: {results}"
            except Exception as e:
                return "I am very sorry, but no answer was found."

        
    def answer_factual(self, query, entities, predicates):
        status, results = self.factual.answer_query(query, entities, predicates)
        return status, results

    def answer_embedding(self, entities, predicates):
        results = self.embeddings.find(entities[0], predicates[0])
        return results
        

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("sharp-cloud", "Q1jeQ8I2")
    demo_bot.listen()
