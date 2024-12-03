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
                print("possible_ids: ", possible_ids)
                if len(possible_ids) == 0:
                    raise Exception("I'm sorry, I couldn't find any images for this entity.")
                else:
                    possible_images = []
                    if re.match(self.regex, entity): # is an actor 
                        for index, row in self.image_df.iterrows():
                            if(len(row['cast']) == 1 and row['cast'][0] in possible_ids):
                                possible_images.insert(0, row['img']) # good match 
                                break
                            elif any(item in possible_ids for item in row['cast']):
                                possible_images.append(row['img'])
                                if len(possible_ids) > 5:
                                    break # no point in searching for more 
                    else: # is a movie
                        for index, row in self.image_df.iterrows():
                            if(len(row['movie']) == 1 and row['movie'][0] in possible_ids):
                                possible_images.insert(0, row['img'])
                            elif any(item in possible_ids for item in row['movie']):
                                possible_images.append(row['img'])
                                if len(possible_ids) > 5:
                                    break
                    print("possible_images: ", possible_images)
                    if len(possible_images) == 0:
                        continue 
                    else:
                        images.append(possible_images[0].split(".jpg")[0])
            if(len(images) == 0):
                raise Exception("I'm sorry, I couldn't find any images for this entity.")
            return images

            
            