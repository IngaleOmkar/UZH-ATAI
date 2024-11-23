import pandas as pd
from responder import Responder

class ImageResponder(Responder):

    def __init__(self, data_repository, entity_extractor, intent_classifier):
        super().__init__(data_repository, entity_extractor, intent_classifier)
        self.image_df = data_repository.get_image_df()
        self.master_df = data_repository.get_master_df()
        self.movies = data_repository.get_movies_df()

    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)
        if len(entities) == 0:
            raise Exception("I'm sorry, I couldn't understand the query.")
        else:
            actors = []
            movies = []
            for entity in entities:
                # some values may be N/A. Skip them in the if statement 
                if len(self.movies[self.movies['movie_name'].str.contains(entity, na=False)]) == 0:
                    # not a movie
                    actors.append(entity)
                else:
                    movies.append(entity)
            print("actors: ", actors)
            print("movies: ", movies)
            if len(actors) == 0:
                raise Exception("I'm sorry, I couldn't understand the query.")
            if len(actors) == 1:
                # only single actor 
                actor = actors[0]
                print("actor: ", actor)
                possible_ids = self.master_df[self.master_df['label'].str.contains(actor, na=False)]['id'].to_list()
                possible_ids = [str(x) for x in possible_ids if 'nm' in str(x)]
                print("possible_ids: ", possible_ids, possible_ids[0])
                if len(possible_ids) == 0:
                    raise Exception("I'm sorry, I couldn't find any images for this actor.")
                else:
                    # possible_images = self.image_df[self.image_df['cast'].apply(lambda x: any(item in possible_ids for item in x))]['img'].to_list()
                    possible_images = []
                    for index, row in self.image_df.iterrows():
                        if(len(row['cast']) == 1 and row['cast'][0] in possible_ids):
                            possible_images.insert(0, row['img'])
                        elif any(item in possible_ids for item in row['cast']):
                            possible_images.append(row['img'])
                    print("possible_images: ", possible_images)
                    if len(possible_images) == 0:
                        raise Exception("I'm sorry, I couldn't find any images for this actor.")
                    else:
                        # return max 3 images. If less than 3 images, return all
                        img_arr = []
                        for i in range(min(3, len(possible_images))):
                            img_arr.append(possible_images[i].split(".jpg")[0])
                        return img_arr

            
            