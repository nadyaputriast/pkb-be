from flask import Flask, request, jsonify
from pymongo import MongoClient, TEXT
from bson import ObjectId
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import re
import traceback

from generator import converse_with_llm
# from apiKey import MONGO_CONNECTION_STRING
import os
from dotenv import load_dotenv
load_dotenv()

from flask_cors import CORS
import spacy
import logging

logging.basicConfig(level=logging.INFO)

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
try:
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client["RecommendationMovie"]
    movies_collection = db["movies"]
    history_collection = db["search_history"]
    logging.info("MongoDB connection successful")
except Exception as e:
    logging.error(f"MongoDB connection failed: {e}")
    raise

try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("Spacy model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load spacy model: {e}")
    raise

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("SentenceTransformer model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}")
    raise

app = Flask(__name__)
CORS(app)

GENRE_SYNONYMS = {
    "romance": ["romance", "romantic", "love", "rom-com", "romantic comedy"],
    "action": ["action", "adventure", "fight", "combat", "superhero", "martial arts"],
    "comedy": ["comedy", "funny", "humor", "satire", "comedic", "hilarious"],
    "horror": ["horror", "scary", "thriller", "fear", "frightening", "spooky"],
    "sci-fi": ["sci-fi", "science fiction", "space", "alien", "futuristic", "technology"],
    "family": ["family", "kids", "children", "child-friendly", "child", "all ages"],
    "animation": ["animation", "animated", "cartoon", "pixar", "disney", "anime"],
    "drama": ["drama", "emotional", "tearjerker", "melodrama", "dramatic"],
    "fantasy": ["fantasy", "magic", "myth", "supernatural", "wizard", "fairy tale", "magical"],
    "mystery": ["mystery", "detective", "crime", "suspense", "whodunit", "investigation"],
    "documentary": ["documentary", "docu", "real-life", "non-fiction", "biopic", "true story"],
    "musical": ["musical", "music", "singing", "dance", "broadway"],
    "war": ["war", "battle", "military", "army", "soldier", "combat"],
    "history": ["history", "historical", "period drama", "biographical", "period"],
    "sports": ["sports", "athletic", "competition", "soccer", "basketball", "football"],
    "crime": ["crime", "gangster", "mafia", "noir", "heist", "criminal"]
}

def parse_advanced_filter(query):
    filters = {}
    query_lower = query.lower()
    
    if "top" in query_lower or "high-rated" in query_lower:
        filters["vote_average"] = {"$gte": 8.5}
    if "popular" in query_lower:
        filters["vote_count"] = {"$gte": 500}
    if "recent" in query_lower:
        filters["release_date"] = {"$gte": "2020-01-01"}
    if "old" in query_lower:
        filters["release_date"] = {"$lt": "2000-01-01"}
        
    return filters

def clean_document(doc):
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

def retrieve_similar_movies(query, n=6):
    try:
        logging.info(f"Retrieving similar movies for query: {query}")
        query_embedding = model.encode(query).tolist()
        filters = parse_advanced_filter(query)
        
        # Vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "movie_index", 
                    "queryVector": query_embedding,
                    "path": "movie_embedding",
                    "limit": n,
                    "numCandidates": 1000,
                }
            }
        ]
        
        # Add filters if any
        if filters:
            pipeline.append({"$match": filters})
            
        pipeline.append({
            "$project": {
                "title": 1,
                "overview": 1,
                "poster_path": 1,
                "vote_average": 1,
                "vote_count": 1,
                "release_date": 1,
                "genre_name": 1,
                "score": {"$meta": "searchScore"}
            }
        })
        
        similar_movies_search = movies_collection.aggregate(pipeline)
        similar_movies = [clean_document(movie) for movie in similar_movies_search]
        logging.info(f"Found {len(similar_movies)} similar movies")
        return similar_movies
        
    except Exception as e:
        logging.error(f"Error in retrieve_similar_movies: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Fallback to simple search
        try:
            logging.info("Attempting fallback search")
            fallback_movies = list(movies_collection.find(
                {},
                {
                    "title": 1,
                    "overview": 1,
                    "poster_path": 1,
                    "vote_average": 1,
                    "vote_count": 1,
                    "release_date": 1,
                    "genre_name": 1
                }
            ).sort("vote_average", -1).limit(n))
            
            return [clean_document(movie) for movie in fallback_movies]
            
        except Exception as fallback_error:
            logging.error(f"Fallback search also failed: {fallback_error}")
            return []

def retrieve_similar_movies_by_genre(genre, n=6, query=""):
    try:
        logging.info(f"Retrieving movies by genre: {genre}")
        filters = parse_advanced_filter(query)
        filters["genre_name"] = {"$regex": genre, "$options": "i"}
        
        matching_movies = list(movies_collection.find(
            filters, 
            {
                "title": 1,
                "overview": 1,
                "poster_path": 1,
                "vote_average": 1,
                "vote_count": 1,
                "release_date": 1,
                "genre_name": 1
            },
        ).sort("vote_average", -1).limit(n))
        
        result = [clean_document(movie) for movie in matching_movies]
        logging.info(f"Found {len(result)} movies for genre: {genre}")
        return result
        
    except Exception as e:
        logging.error(f"Error in retrieve_similar_movies_by_genre: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return []

def process_query(query):
    try:
        doc = nlp(query)
        keywords = [chunk.text.lower() for chunk in doc.noun_chunks] + [
            token.text.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop
        ]
        return list(set(keywords))
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return [query.lower()]

def match_genre(keywords):
    try:
        for keyword in keywords:
            for genre, synonyms in GENRE_SYNONYMS.items():
                if keyword.lower() in synonyms:
                    return genre
        return None
    except Exception as e:
        logging.error(f"Error matching genre: {e}")
        return None

@app.route("/api/query", methods=["POST"])
def handle_query():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        query = data.get("query", "").strip()
        skip_cache = data.get("skipCache", False)

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        logging.info(f"Processing query: {query}, Skip cache: {skip_cache}")

        # Check existing entry - SKIP jika skipCache = True
        if not skip_cache:
            try:
                existing_entry = history_collection.find_one({"query": query})
                if existing_entry and "result" in existing_entry:
                    logging.info("Returning cached result")
                    return jsonify(existing_entry["result"])
            except Exception as e:
                logging.warning(f"Error checking history: {e}")

        # Process query
        input_prompt = process_query(query)
        genre_match = match_genre(input_prompt)

        logging.info(f"Keywords: {input_prompt}, Genre match: {genre_match}")

        # Get similar movies
        if genre_match:
            similar_movies = retrieve_similar_movies_by_genre(genre_match, n=6, query=query)
        else:
            cleaned_query = " ".join(input_prompt) if input_prompt else query
            similar_movies = retrieve_similar_movies(cleaned_query, n=6)

        if not similar_movies:
            return jsonify({
                "similar_movies": [],
                "recommendation": "Sorry, no movies found for your query. Please try different search terms."
            })

        # === PROMPT LLM DENGAN DETAIL ===
        try:
            logging.info("=== STARTING LLM RECOMMENDATION ===")

            # Ambil info detail 3 film teratas
            movie_infos = []
            for movie in similar_movies[:3]:
                info = f"Title: {movie.get('title', 'Unknown')}\n"
                info += f"Overview: {movie.get('overview', '')}\n"
                info += f"Genres: {movie.get('genre_name', '')}\n"
                info += f"Rating: {movie.get('vote_average', '')}/10\n"
                info += f"Vote Count: {movie.get('vote_count', '')}\n"
                movie_infos.append(info)

            detailed_prompt = f"""
You are a helpful movie recommendation assistant.

User is looking for: {query}

Here are some movies you found:
{chr(10).join(movie_infos)}

For each movie, explain in 1-2 sentences why it is a good recommendation for the user, mentioning the title and its unique qualities. Use a friendly and informative tone.
"""

            logging.info(f"Calling LLM with prompt: {detailed_prompt}")

            recommendation = converse_with_llm(detailed_prompt)
            logging.info(f"LLM OUTPUT: {recommendation}")

        except Exception as e:
            logging.error(f"LLM failed: {e}")
            # FALLBACK: Manual recommendation
            if similar_movies:
                titles = [movie.get('title', 'Unknown') for movie in similar_movies[:5]]
                recommendation = f"Perfect! I found these great movies for '{query}': {', '.join(titles)}. Each one matches what you're looking for and comes highly recommended!"
            else:
                recommendation = f"Sorry, no movies found for '{query}'. Try different keywords!"

        result = {"similar_movies": similar_movies, "recommendation": recommendation}

        try:
            if similar_movies and not skip_cache:
                history_collection.insert_one({
                    "query": query,
                    "result": result,
                    "timestamp": time.time()
                })
                logging.info("Query and result stored in history")
        except Exception as e:
            logging.warning(f"Error storing in history: {e}")

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in handle_query: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        history = history_collection.find({}, {"_id": 0, "query": 1}).sort("timestamp", -1).limit(10)
        return jsonify([entry["query"] for entry in history if "query" in entry])
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return jsonify([])

@app.route("/api/movies", methods=["GET"])
def get_movies():
    try:
        year = request.args.get("year")
        month = request.args.get("month")
        query = {}

        if year and month:
            regex = f"^{year}-{month}"
            query["release_date"] = {"$regex": regex}

        movies = list(movies_collection.find(
            query,
            {
                "_id": 0,
                "title": 1,
                "overview": 1,
                "poster_path": 1,
                "vote_average": 1,
                "vote_count": 1,
                "release_date": 1,
                "genre_name": 1,
                "runtime": 1,
                "popularity": 1
            }
        ).sort("popularity", -1))

        return jsonify(movies)
        
    except Exception as e:
        logging.error(f"Error fetching movies: {e}")
        return jsonify([])

# @app.route("/health", methods=["GET"])
# def health_check():
#     return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == "__main__":
    app.run(debug=True, port=5001)