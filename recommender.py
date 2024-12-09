from sklearn.metrics import pairwise_distances
from responder import Responder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import process
from formatter import FormatHelper

class RecommendationResponder(Responder):
        
    def __init__(self, data_repository, entity_extractor, mlp_intent_classifier):
        super().__init__(data_repository, entity_extractor, mlp_intent_classifier)
        self.data_repository = data_repository
        self.entity_extractor = entity_extractor
        self.graph = self.data_repository.get_graph()
        self.entity_emb = self.data_repository.get_entity_emb()
        self.relation_emb = self.data_repository.get_relation_emb()
        self.ent2id = self.data_repository.get_ent2id()
        self.id2ent = self.data_repository.get_id2ent()
        self.rel2id = self.data_repository.get_rel2id()
        self.id2rel = self.data_repository.get_id2rel()
        self.ent2lbl = self.data_repository.get_ent2lbl()
        self.lbl2ent = self.data_repository.get_lbl2ent()
        self.rel2lbl = self.data_repository.get_rel2lbl()
        self.lbl2rel = self.data_repository.get_lbl2rel()
        self.movies = self.data_repository.get_movies_df()
        self.list_of_all_genres = self.data_repository.get_list_of_all_genres()
        self.list_of_all_genres_graph = self.data_repository.get_list_of_all_genres_graph()
        self.list_of_all_actors = self.data_repository.get_list_of_all_actors()
        self.uri_to_label = self.data_repository.get_uri_to_label()
        self.label_to_uri = self.data_repository.get_label_to_uri()

    def actor_query(self, actors):
        actors_values = " ".join([f"wd:{actor}" for actor in actors])

        q = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX ddis: <http://ddis.ch/atai/>

            SELECT ?movie ?boxOffice ?rating
            WHERE {{
                ?movie wdt:P161 ?actor ;
                    wdt:P2142 ?boxOffice .

                OPTIONAL {{ ?movie ddis:rating ?rating . }}

                VALUES ?actor {{ {actors_values} }}

                FILTER(!BOUND(?rating) || ?rating > 7.0)
            }}
            ORDER BY DESC(?boxOffice)
            LIMIT 3
        """

        results = self.graph.query(q)
        movies = [str(row.movie) for row in results]
        print(f"nr movies: {len(movies)}")
        return movies[:3]
   
    def genre_query(self, genres):
        genres_values = " ".join([f"wd:{genre}" for genre in genres])

        q = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX ddis: <http://ddis.ch/atai/>

            SELECT ?movie ?boxOffice ?rating
            WHERE {{
                ?movie wdt:P136 ?movieGenre ;
                    wdt:P2142 ?boxOffice .

                OPTIONAL {{ ?movie ddis:rating ?rating . }}

                VALUES ?movieGenre {{ {genres_values} }}
                FILTER(!BOUND(?rating) || ?rating > 7.0)
            }}
            ORDER BY DESC(?boxOffice)
            LIMIT 3
        """
        
        results = self.graph.query(q)
        movies = [str(row.movie) for row in results]
        return movies
    
    def movie_query(self):

        q = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX ddis: <http://ddis.ch/atai/>

            SELECT ?movie ?boxOffice
            WHERE {{
                ?movie wdt:P2142 ?boxOffice .
                OPTIONAL {{ ?movie ddis:rating ?rating . }}

                FILTER(!BOUND(?rating) || ?rating > 7.5)
            }}
            ORDER BY DESC(?boxOffice)
            LIMIT 3
        """

        results = self.graph.query(q)
        movies = [str(row.movie) for row in results]
        print(f"nr movies: {len(movies)}")
        return movies[:3]

    def get_ent_vec(self, ent_label):
        if (ent_label not in self.lbl2ent):
            raise Exception("Label not found.")
        
        ent = self.lbl2ent[ent_label]

        if (ent not in self.ent2id):
            raise Exception("Entity not found.")
        
        ent_id = self.ent2id[ent]

        ent_vec = self.entity_emb[ent_id]
        return ent_vec
    
    def get_subgenres(self, genre):
        return [s for s in self.list_of_all_genres_graph if genre in s]
    
    def get_external_movies(self, query):
        genre = process.extractOne(query, self.list_of_all_genres)[0]
        print(f"ext genre: {genre}")
        genre = genre.upper()
        list_of_relevant_movies = self.movies[self.movies['genre'].apply(lambda x: genre in x)]
        return list_of_relevant_movies['movie_name'].tolist()[:3]
    
    def get_popular_movies(self):
        results = self.movie_query()
        lbls = [self.uri_to_label[uri] for uri in results]
        return lbls

    def answer_query(self, query):
        entities = self.entity_extractor.get_guaranteed_entities(query)
        print(f"Identified entities: {entities}")

        title_entities = []
        actor_entities = []

        for entity in entities:
            if (entity in self.list_of_all_actors):
                actor_entities.append(entity)
            elif len(self.movies[self.movies['movie_name'].str.contains(entity)]) > 0:
                title_entities.append(entity)

        print(f"Identified title entities: {title_entities}")
        print(f"Identified actor entities: {actor_entities}")

        if(len(title_entities) == 0 and len(actor_entities) == 0): # GENRE
            match = process.extractOne(query, self.list_of_all_genres_graph)
            genre = match[0]
            genres = self.get_subgenres(genre)
            uris = [self.label_to_uri[g.lower()] for g in genres if g.lower() in self.label_to_uri]
            if (len(uris) == 0):
                return self.get_popular_movies(), "because these are very popular movies"
                # return self.get_external_movies(query), f"because, according to our external data, these are popular movies of the genre {genre}"
            ents = [uri.split("/")[-1] for uri in uris]
            results = self.genre_query(ents)
            print(f"genre ents: {ents}")
            lbls = [self.uri_to_label[ent] for ent in results]
            if (len(lbls) == 0):
                return self.get_popular_movies(), "because these are very popular movies"
                # return self.get_external_movies(query), f"because, according to our external data, these are popular movies of the genre {genre}"
            else:
                return lbls, f"because these are popular movies of the genre {genre}"
        elif (len(actor_entities) > 0 and actor_entities[0] in self.lbl2ent): # ACTOR
            ents = [self.lbl2ent[actor] for actor in actor_entities if actor in self.lbl2ent]
            ents = [ent.split("/")[-1] for ent in ents]
            results = self.actor_query(ents)
            lbls = [self.uri_to_label[ent] for ent in results]
            return lbls, f"because these are popular movies starring {FormatHelper.array_to_sentence(actor_entities)}"
        else: # SIMILARITY
            entity_vecs = []
            for ent in title_entities:
                try: 
                    entity_vecs.append(self.get_ent_vec(ent))
                except Exception as e:
                    print(f"entity {ent} could not be found in embedding: {e}")
                    
            entity_vec = np.mean(entity_vecs, axis=0)

            try:
                distances = pairwise_distances(entity_vec.reshape(1, -1), self.entity_emb)
                closest_indices = np.argsort(distances[0])[:30]
                ents = [self.ent2lbl[self.id2ent[id]] for id in closest_indices]
                print(ents)

                final_ents = []

                for ent in ents:
                    if ent not in [entity for entity in entities] and len(self.movies[self.movies['movie_name'].str.contains(ent)]) > 0:
                        final_ents.append(ent)

                print(final_ents)

                if len(final_ents) >= 3:
                    return final_ents[:3], f"because these movies are similar to the movies you named"
                else:
                    return final_ents, f"because these movies are similar to the movies you named"
            except:
                raise Exception("I'm sorry, I couldn't find the answer to your question.")
