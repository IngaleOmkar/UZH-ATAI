import pandas as pd
import re
from responder import Responder

class ImageResponder(Responder):

    def __init__(self, data_repository, entity_extractor, intent_classifier):
        super().__init__(data_repository, entity_extractor, intent_classifier)
        self.image_df = data_repository.get_image_df()
        self.master_df = data_repository.get_master_df()
        self.movies = data_repository.get_movies_df()
        self.regex = r'^nm\d{7}$'

    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)
        if len(entities) == 0:
            raise Exception("I'm sorry, I couldn't understand the query.")
        else:
            images=[]
            for entity in entities:
                print("entity: ", entity)
                possible_ids = self.master_df[self.master_df['label'].str.contains(entity, na=False)]['id'].to_list()
                possible_ids = [str(x) for x in possible_ids]
                # cap the length of possible_ids to 5
                possible_ids = possible_ids[:5]
                print("possible_ids: ", possible_ids)
                if len(possible_ids) == 0:
                    continue
                else:
                    possible_images = []
                    if re.match(self.regex, entity): # is an actor 
                        print('identified actor entity: ', entity)
                        # Filter rows where 'cast' has only one element and that element is in possible_ids
                        good_matches = self.image_df[self.image_df['cast'].apply(lambda x: len(x) == 1 and x[0] in possible_ids)]

                        # If there are good matches, insert the first one at the beginning of possible_images
                        if not good_matches.empty:
                            possible_images.insert(0, good_matches.iloc[0]['img'])
                        else:
                            # Filter rows where any element in 'cast' is in possible_ids
                            possible_matches = self.image_df[self.image_df['cast'].apply(lambda x: any(item in possible_ids for item in x))]
                            
                            # Append images to possible_images
                            possible_images.extend(possible_matches['img'].tolist())

                    else: # is a movie
                        print('identified movie entity: ', entity)

                        # Filter rows where 'movie' has only one element and that element is in possible_ids
                        good_movie_matches = self.image_df[self.image_df['movie'].apply(lambda x: len(x) == 1 and x[0] in possible_ids)]

                        # If there are good matches, insert the first one at the beginning of possible_images
                        if not good_movie_matches.empty:
                            possible_images.insert(0, good_movie_matches.iloc[0]['img'])
                        else:
                            print("No good matches found")
                            # Filter rows where any element in 'movie' is in possible_ids
                            possible_movie_matches = self.image_df[self.image_df['movie'].apply(lambda x: any(item in possible_ids for item in x))]
                            
                            # Append images to possible_images
                            possible_images.extend(possible_movie_matches['img'].tolist())
                        
                    print("possible_images: ", possible_images)
                    if len(possible_images) == 0:
                        continue 
                    else:
                        images.append(possible_images[0].split(".jpg")[0])
            if(len(images) == 0):
                raise Exception("I'm sorry, I couldn't find any images for this entity.")
            return images