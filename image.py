import pandas as pd
import re
from responder import Responder

class ImageResponder(Responder):

    def __init__(self, data_repository, entity_extractor, intent_classifier):
        super().__init__(data_repository, entity_extractor, intent_classifier)
        self.image_df = data_repository.get_image_df()
        self.master_df = data_repository.get_master_df()
        self.movies = data_repository.get_movies_df()
        self.regex = r'nm\d{7}' # regex for actor id
        self.poster_df = self.image_df[self.image_df['type'] == 'poster']
        # convert 'movie' column to ensure it is a list 
        self.poster_df['movie'] = self.poster_df['movie'].apply(lambda x: eval(x))

    def is_actor(self, list_of_ids):
        for id in list_of_ids:
            if re.match(self.regex, id):
                return True, id
        return False, None

    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)
        images = []
        if len(entities) == 0:
            raise Exception("I'm sorry, I couldn't understand the query.")
        else:
            for entity in entities:
                print("current entity: ", entity)
                # Check if entity is a movie
                possible_ids = self.master_df[self.master_df['label'].str.contains(entity, na=False)]['id'].to_list()
                possible_ids = [str(x) for x in possible_ids]
                print("possible_ids: ", possible_ids)
                if len(possible_ids) == 0:
                    continue
                # Check if entity is an actor
                is_actor, id = self.is_actor(possible_ids)
                print("is_actor: ", is_actor)

                if is_actor: # is an actor
                    # good_matches = self.image_df[self.image_df['cast'].apply(lambda x: len(x) == 1 and x[0] in possible_ids)]
                    good_matches = self.image_df[self.image_df['cast'].apply(lambda x: len(x) == 1 and x[0] == id)]
                    if not good_matches.empty:
                        images.append(good_matches.iloc[0]['img'].split(".jpg")[0])
                        continue
                    else:
                        possible_matches = self.image_df[self.image_df['cast'].apply(lambda x: any(item in possible_ids for item in x))]
                        if not possible_matches.empty:
                            images.append(possible_matches.iloc[0]['img'].split(".jpg")[0])
                            continue
                        
                good_movie_matches = self.poster_df[self.poster_df['movie'].apply(lambda x: x[0] in possible_ids)]
                if not good_movie_matches.empty:
                    print("good_movie_matches: ", good_movie_matches)
                    images.append(good_movie_matches.iloc[0]['img'].split(".jpg")[0])
                    continue
                else:
                    possible_movie_matches = self.poster_df[self.poster_df['movie'].apply(lambda x: any(item in possible_ids for item in x))]
                    if not possible_movie_matches.empty:
                        print("possible_movie_matches: ", possible_movie_matches)
                        images.append(possible_movie_matches.iloc[0]['img'].split(".jpg")[0])
                        continue
            if(len(images) == 0):
                raise Exception("I'm sorry, I couldn't find any images for this entity.")
            return images