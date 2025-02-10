# Import required libraries.
import sqlite3               # For interacting with the SQLite database.
import numpy as np           # For numerical operations and array handling.
import faiss                 # For efficient similarity search (install via pip install faiss-cpu).
from dataclasses import dataclass  # For creating data classes.
import heapq                 # For heap queue algorithms, used to rank recommendations.
from collections import defaultdict  # For dictionaries with default values.
import random                # For random selections.
import os                    # For file system operations.

# Define a data class to represent a song and its attributes.
@dataclass
class Song:
    track_id: str       # Unique identifier for the song.
    title: str          # Song title.
    artist: str         # Artist name.
    duration: float     # Duration of the song in seconds.
    hotttnesss: float   # A measure of the artist's popularity.
    year: int           # Year of release.
    familiarity: float  # Familiarity metric for the artist.
    popularity: int = 0 # Popularity counter (e.g., number of plays or interactions).
    features: np.ndarray = None  # Feature vector used for similarity matching.

# Class for handling music recommendations using multiple strategies.
class MusicRecommender:
    def __init__(self, db_path="million_songs.db", feature_file="features.dat", index_file="music_index.faiss"):
        # Set file paths for the database, feature storage, and FAISS index.
        self.db_path = db_path
        self.feature_file = feature_file
        self.index_file = index_file
        
        # Initialize data structures:
        self.songs = {}  # Mapping from song track_id to Song objects.
        self.users = defaultdict(dict)  # User data, including liked songs and interaction history.
        self.feature_matrix = None  # Memory-mapped numpy array to store song feature vectors.
        self.faiss_index = None  # FAISS index for fast similarity search.
        self.song_id_to_index = {}  # Map from song track_id to its index in the feature matrix.
        self.index_to_song_id = {}  # Reverse mapping from index to song track_id.
        
        # Prepare the memory-mapped file for feature storage.
        self._init_memmap()

    def _init_memmap(self):
        """
        Initialize memory-mapped storage for features.
        
        This function removes any existing feature file to ensure there are no shape conflicts
        when a new feature matrix is created.
        """
        if os.path.exists(self.feature_file):
            os.remove(self.feature_file)
        # The actual memory-mapped array will be created in load_data() once the total number of songs is known.
        self.feature_matrix = None

    def load_data(self, batch_size=50000):
        """
        Load songs from the SQLite database in batches and build a memory-mapped feature matrix.
        
        Parameters:
            batch_size (int): Number of songs to load per batch to handle large datasets efficiently.
        """
        # Connect to the SQLite database.
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Count the total number of songs in the database.
            cursor.execute("SELECT COUNT(*) FROM songs")
            total_songs = cursor.fetchone()[0]
            
            # Pre-allocate a memory-mapped file to store feature vectors for all songs.
            # Each song is represented by a 4-dimensional vector.
            self.feature_matrix = np.memmap(self.feature_file, dtype='float32', mode='w+', shape=(total_songs, 4))
            offset = 0  # Offset to track the position in the feature matrix.
            
            # Process the songs in batches.
            while offset < total_songs:
                cursor.execute(f"""
                    SELECT track_id, title, artist_name, duration, artist_hotttnesss, year, artist_familiarity 
                    FROM songs
                    LIMIT {batch_size} OFFSET {offset}
                """)
                batch = cursor.fetchall()
                if not batch:
                    break  # Stop if no more songs are returned.
                batch_size_actual = len(batch)
                # Temporary array to hold feature vectors for the current batch.
                features_batch = np.empty((batch_size_actual, 4), dtype='float32')
                for i, row in enumerate(batch):
                    track_id = row[0]
                    # Create the feature vector:
                    # [duration, artist_hotttnesss, normalized year (year/100), artist_familiarity].
                    features = np.array([row[3], row[4], row[5] / 100.0, row[6]], dtype='float32')
                    # Normalize the feature vector so that the inner product approximates cosine similarity.
                    norm = np.linalg.norm(features)
                    if norm > 0:
                        features = features / norm
                    # Create a Song instance with the fetched attributes and computed feature vector.
                    self.songs[track_id] = Song(
                        track_id=row[0],
                        title=row[1],
                        artist=row[2],
                        duration=row[3],
                        hotttnesss=row[4],
                        year=row[5],
                        familiarity=row[6],
                        popularity=0,
                        features=features
                    )
                    # Map the song's track_id to its position in the feature matrix.
                    self.song_id_to_index[track_id] = offset + i
                    self.index_to_song_id[offset + i] = track_id
                    # Store the feature vector in the temporary batch array.
                    features_batch[i] = features
                # Write the batch of feature vectors into the memory-mapped file.
                self.feature_matrix[offset: offset + batch_size_actual] = features_batch
                offset += batch_size_actual
                print(f"Loaded {offset}/{total_songs} songs...")
        # After loading all songs, build the FAISS index for similarity search.
        self._build_faiss_index()

    def _build_faiss_index(self):
        """
        Build a FAISS index from the feature matrix for fast similarity queries.
        
        We use FAISS's IndexFlatIP which performs inner product searches.
        Since the feature vectors are normalized, inner product similarity is equivalent to cosine similarity.
        """
        print("Building FAISS index...")
        dim = 4  # Dimensionality of each feature vector.
        # Convert the memory-mapped feature matrix into a contiguous NumPy array.
        features_np = np.array(self.feature_matrix)
        self.faiss_index = faiss.IndexFlatIP(dim)
        # Add the feature vectors to the FAISS index.
        self.faiss_index.add(features_np)
        # Optionally save the FAISS index to disk.
        faiss.write_index(self.faiss_index, self.index_file)
        print("FAISS index built and saved.")

    def get_similar_songs(self, song_id, k=10):
        """
        Retrieve the k most similar songs (excluding the query song itself) using the FAISS index.
        
        Parameters:
            song_id (str): The track ID of the reference song.
            k (int): Number of similar songs to retrieve.
            
        Returns:
            List of tuples (similar_song_id, similarity_score).
        """
        if song_id not in self.song_id_to_index:
            return []  # Return empty list if the song is not found.
        # Retrieve the index corresponding to the song.
        idx = self.song_id_to_index[song_id]
        # Reshape the feature vector for querying (FAISS requires a 2D array).
        query = self.feature_matrix[idx].reshape(1, 4)
        # Search for the nearest neighbors in the FAISS index. Request k+1 to account for the song itself.
        distances, indices = self.faiss_index.search(query, k + 1)
        if indices.shape[1] <= 1:
            return []
        similar = []
        # Skip the first result (the song itself) and collect the next k similar songs.
        for i, score in zip(indices[0][1:], distances[0][1:]):
            similar.append((self.index_to_song_id[i], score))
        return similar

    def recommend(self, user_id, top_n=10):
        """
        Generate personalized song recommendations using a hybrid strategy.
        
        The recommendation strategy consists of:
          1. Content-based filtering: Boosting songs similar to those the user has liked.
          2. Popularity-based ranking: Incorporating overall song popularity.
          3. Diversity: Adding a random component to introduce variety.
          
        Parameters:
            user_id (str): The user's identifier.
            top_n (int): Number of recommendations to generate.
            
        Returns:
            List of Song objects recommended for the user.
        """
        if user_id not in self.users:
            return []
        user = self.users[user_id]
        scores = defaultdict(float)  # Dictionary to accumulate scores for each song.
        
        # --- Content-based component ---
        # For each song the user has liked, get similar songs and add a weighted score.
        for liked_id in user.get('liked', []):
            for similar_id, score in self.get_similar_songs(liked_id, 50):
                scores[similar_id] += score * 0.6  # Weight the similarity score.
        
        # --- Popularity-based component ---
        # Get the top 100 most popular songs.
        popular = heapq.nlargest(100, self.songs.values(), key=lambda x: x.popularity)
        for song in popular:
            # Add a score based on the song's popularity.
            scores[song.track_id] += 0.2 * (song.popularity / 1000.0)
        
        # --- Diversity component ---
        # Add a small random score to a random selection of songs to ensure diversity.
        for _ in range(100):
            random_song = random.choice(list(self.songs.keys()))
            scores[random_song] += 0.2 * random.random()
        
        # Build a heap of scores for ranking, excluding songs the user has already interacted with.
        heap = [(-score, song_id) for song_id, score in scores.items() if song_id not in user.get('history', set())]
        heapq.heapify(heap)
        # Extract the top_n songs based on the highest scores.
        return [self.songs[sid] for _, sid in heapq.nsmallest(top_n, heap)]

    def record_interaction(self, user_id, song_id):
        """
        Record a user interaction with a song and update the song's popularity.
        
        This function updates:
          - The user's history of interacted songs.
          - The song's popularity score.
          - The user's liked songs (if they have liked fewer than 100 songs so far).
        
        Parameters:
            user_id (str): The user's identifier.
            song_id (str): The identifier of the song interacted with.
        """
        # Create a new user entry if one does not exist.
        if user_id not in self.users:
            self.users[user_id] = {'liked': set(), 'history': set()}
        self.users[user_id]['history'].add(song_id)
        if song_id in self.songs:
            # Increase the popularity count for the song.
            self.songs[song_id].popularity += 1
            # Add to the liked list if the user has liked fewer than 100 songs.
            if len(self.users[user_id]['liked']) < 100:
                self.users[user_id]['liked'].add(song_id)

    def display_song_ids(self, num_songs=10):
        """
        Print the first num_songs song IDs from the loaded songs.
        
        Parameters:
            num_songs (int): Number of song IDs to display.
        """
        for i, song_id in enumerate(self.songs.keys()):
            if i >= num_songs:
                break
            print(song_id)

def main():
    """
    Main function that provides an interactive command-line interface (CLI).
    
    Workflow:
      1. Initialize the music recommender and load the song data.
      2. Display a sample of songs to the user.
      3. Allow the user to select songs they like.
      4. Record the user interactions.
      5. Generate and display personalized song recommendations.
    """
    # Initialize the recommender system with the SQLite database.
    recommender = MusicRecommender("million_songs.db")
    print("Loading data...")
    # Load song data; here, a smaller batch size is used for the demo.
    recommender.load_data(batch_size=1000)
    
    # Check that songs have been loaded.
    if not recommender.songs:
        print("No songs loaded. Exiting.")
        return
    
    # Display a random sample of songs for the user to review.
    sample_songs = random.sample(list(recommender.songs.values()), min(5, len(recommender.songs)))
    print("\nHere are some sample songs:")
    for idx, song in enumerate(sample_songs, 1):
        print(f"{idx}. {song.title} - {song.artist} ({song.year})")
    
    # Prompt the user to select songs they like (entering the corresponding numbers).
    liked_input = input("\nEnter the numbers of the songs you like (comma-separated): ")
    liked_numbers = [s.strip() for s in liked_input.split(",") if s.strip().isdigit()]
    liked_numbers = [int(s) for s in liked_numbers]
    
    liked_song_ids = []
    for num in liked_numbers:
        if 1 <= num <= len(sample_songs):
            liked_song_ids.append(sample_songs[num - 1].track_id)
    
    if not liked_song_ids:
        print("No valid song selections made. Exiting.")
        return
    
    # Ask the user for their unique user ID.
    user_id = input("\nEnter your user ID: ").strip()
    if not user_id:
        print("Invalid user ID. Exiting.")
        return
    
    # Initialize or reset the user's entry in the system.
    recommender.users[user_id] = {'liked': set(), 'history': set()}
    # Record each liked song as a user interaction.
    for track_id in liked_song_ids:
        # NOTE: The correct method name is record_interaction.
        recommender.record_interaction(user_id, track_id)
    
    # Generate personalized song recommendations.
    recommendations = recommender.recommend(user_id, top_n=10)
    print("\nYour recommended songs:")
    for idx, song in enumerate(recommendations, 1):
        print(f"{idx}. {song.title} - {song.artist} ({song.year})")
    
# Run the CLI if the script is executed directly.
if __name__ == "__main__":
    main()
