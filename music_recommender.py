# Import required libraries
import pandas as pd  # Data manipulation
import networkx as nx  # Graph data structure
import heapq  # Priority queue implementation
from collections import defaultdict  # Dictionary with default values
from sklearn.metrics.pairwise import cosine_similarity  # Similarity calculation
import random  # Random sampling

class MusicRecommender:
    def __init__(self):
        """Initialize recommender system with empty data structures"""
        self.graph = nx.Graph()  # Main graph storing users-songs relationships
        self.songs = {}  # Dictionary of song_id -> song metadata
        self.users = {}  # Dictionary of user_id -> user preferences/history
        self.genre_map = defaultdict(list)  # genre -> list of song_ids mapping
        self.similarity_cache = defaultdict(list)  # song_id -> [(similar_song, score)]

    def load_dataset(self, csv_path):
        """Load songs from CSV file with basic error handling"""
        try:
            df = pd.read_csv(csv_path)  # Read CSV into DataFrame
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return

        # Verify required columns are present
        expected_columns = {'track_id', 'title', 'artist', 'genre', 'danceability', 'energy', 'valence', 'year', 'popularity'}
        if not expected_columns.issubset(df.columns):
            print("CSV file is missing one or more required columns.")
            return
        
        # Process each row in the DataFrame
        for _, row in df.iterrows():
            try:
                track_id = row['track_id']
                # Store song metadata
                self.songs[track_id] = {
                    'track_id': track_id,
                    'title': row['title'],
                    'artist': row['artist'],
                    'genre': row['genre'],
                    'features': [  # Audio features for similarity calculation
                        row['danceability'],
                        row['energy'],
                        row['valence']
                    ],
                    'year': row['year'],
                    'popularity': row['popularity']
                }
                # Add to genre index and graph
                self.genre_map[row['genre'].lower()].append(track_id)
                self.graph.add_node(track_id, type='song')  # Add song node to graph
            except Exception as e:
                print(f"Error processing row: {e}")

    def add_user(self, user_id, fav_genres=None):
        """Register new user with validation"""
        if not user_id or user_id.strip() == "":
            print("User ID cannot be empty.")
            return False
        # Initialize user data structure
        self.users[user_id] = {
            'fav_genres': [g.strip().lower() for g in fav_genres] if fav_genres else [],
            'liked_songs': [],  # Songs explicitly liked by user
            'history': []  # All interacted songs
        }
        self.graph.add_node(user_id, type='user')  # Add user node to graph
        return True

    def record_interaction(self, user_id, song_id):
        """Track user-song interactions and update graph"""
        if user_id in self.users and song_id in self.songs:
            # Create edge between user and song with weight
            self.graph.add_edge(user_id, song_id, weight=1)
            # Update user preferences and song popularity
            self.users[user_id]['liked_songs'].append(song_id)
            self.users[user_id]['history'].append(song_id)
            self.songs[song_id]['popularity'] += 1  # Increment popularity count
        else:
            print("Invalid user or song ID for interaction.")

    def calculate_similarity(self):
        """Precompute song similarities using cosine similarity on audio features"""
        song_ids = list(self.songs.keys())
        if not song_ids:
            print("No songs available to calculate similarity.")
            return
        
        # Extract feature vectors and compute similarity matrix
        features = [self.songs[sid]['features'] for sid in song_ids]
        similarity_matrix = cosine_similarity(features)
        
        # Build similarity graph edges and cache
        for i, sid1 in enumerate(song_ids):
            for j, sid2 in enumerate(song_ids[i+1:], start=i+1):
                if similarity_matrix[i][j] > 0.7:  # Threshold for similarity
                    # Add edge between similar songs
                    self.graph.add_edge(sid1, sid2, weight=similarity_matrix[i][j])
                    # Cache similarity results for quick access
                    self.similarity_cache[sid1].append((sid2, similarity_matrix[i][j]))
                    self.similarity_cache[sid2].append((sid1, similarity_matrix[i][j]))

    def recommend(self, user_id, top_n=6):
        """Generate personalized recommendations using multiple strategies"""
        if user_id not in self.users:
            print("User not found.")
            return []

        heap = []  # Use max-heap pattern with negative scores
        user = self.users[user_id]
        
        # Strategy 1: Genre-based recommendations with popularity bias
        for genre in user['fav_genres']:
            for song_id in self.genre_map.get(genre, []):
                song = self.songs[song_id]
                # Calculate composite score (60% popularity, 40% recency)
                score = (
                    song['popularity'] * 0.6 +
                    (song['year'] / 2024) * 0.4  # Normalize year to favor newer songs
                )
                heapq.heappush(heap, (-score, song_id))  # Negative for max-heap
        
        # Strategy 2: Content-based filtering using precomputed similarities
        for song_id in user['liked_songs']:
            for neighbor, similarity in self.similarity_cache.get(song_id, []):
                heapq.heappush(heap, (-similarity, neighbor))  # Higher similarity first
        
        # Strategy 3: Collaborative filtering using graph neighbors
        for neighbor in self.graph.neighbors(user_id):
            if self.graph.nodes[neighbor].get('type') == 'song':
                # Use edge weight as recommendation score
                weight = self.graph[user_id][neighbor]['weight']
                heapq.heappush(heap, (-weight, neighbor))
        
        # Aggregate and deduplicate results
        seen = set()
        recommendations = []
        while heap and len(recommendations) < top_n:
            score, song_id = heapq.heappop(heap)
            if song_id not in seen:
                seen.add(song_id)
                recommendations.append(self.songs[song_id])
        
        return recommendations[:top_n]  # Return requested number of recommendations

def main():
    """Main CLI interface for the recommender system"""
    recommender = MusicRecommender()
    recommender.load_dataset('songs.csv')  # Load song data
    recommender.calculate_similarity()  # Precompute similarities
    
    # CLI interface
    print("🎵 Modern Music Recommender 2024 🎵")
    user_id = input("Enter your user ID: ").strip()
    if not user_id:
        print("User ID is required. Exiting.")
        return
    
    # Get user preferences
    genres_input = input(
        "\nEnter favorite genres (comma-separated):\nOptions: pop, r&b, hip-hop, country, rock, electronic\n"
    )
    genres = [g for g in genres_input.split(',') if g.strip()]
    
    # Display sample tracks for selection
    if recommender.songs:
        print("\nRecent popular tracks:")
        # Randomly sample 5 tracks for demonstration
        sample_tracks = random.sample(list(recommender.songs.values()), min(5, len(recommender.songs)))
        for idx, track in enumerate(sample_tracks, 1):
            print(f"{idx}. {track['title']} - {track['artist']} ({track['year']})")
    else:
        print("No tracks available.")
        return
    
    # Handle track selection with error checking
    choice = input("\nChoose a track you like (1-5): ").strip()
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(sample_tracks):
            liked_song_id = sample_tracks[choice_index]['track_id']
        else:
            print("Invalid choice. No song will be recorded as liked.")
            liked_song_id = None
    except ValueError:
        print("Non-integer input. No song will be recorded as liked.")
        liked_song_id = None
    
    # Register user and record interaction
    if not recommender.add_user(user_id, genres):
        print("Failed to add user. Exiting.")
        return
    
    if liked_song_id:
        recommender.record_interaction(user_id, liked_song_id)
    
    # Generate and display recommendations
    print("\n🎧 Generating your personalized playlist...")
    playlist = recommender.recommend(user_id)
    
    if playlist:
        print("\n🔥 Your Generated Playlist:")
        for idx, track in enumerate(playlist, 1):
            print(f"{idx}. {track['title']}")
            print(f"   Artist: {track['artist']}")
            print(f"   Genre: {track['genre'].upper()} | Year: {track['year']}")
            # Display audio features with formatting
            print(f"   Dance: {track['features'][0]:.2f} | Energy: {track['features'][1]:.2f}\n")
    else:
        print("No recommendations available at this time.")

if __name__ == "__main__":
    main()