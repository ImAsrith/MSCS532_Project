import time
import psutil
import matplotlib.pyplot as plt

# Import the two versions using aliases.
from music_recommender import MusicRecommender as MusicRecommenderPOC
from music_recommender_v1 import MusicRecommender as MusicRecommenderV1

TARGET_SONG_COUNT = 1_000_000  # Target dataset size for extrapolation

def profile_load_data(recommender_class, dataset, batch_size=None):
    """
    Profiles the data loading phase.
    For the POC version, 'dataset' is assumed to be a CSV file (e.g., "songs.csv").
    For the final version, 'dataset' is a SQLite DB (e.g., "million_songs.db"),
    and batch_size can be specified.
    Returns the recommender instance, load time (seconds), and memory used (MB).
    """
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_time = time.time()

    # Instantiate the recommender without passing the dataset.
    rec = recommender_class()

    # For the POC version, load_data expects the CSV file path.
    if dataset.endswith('.csv'):
        rec.load_dataset(dataset)
    else:
        # For the final version, use the batch_size parameter if provided.
        if batch_size:
            rec.load_data(batch_size=batch_size)
        else:
            rec.load_data()

    load_time = time.time() - start_time
    mem_after = process.memory_info().rss
    mem_used_mb = (mem_after - mem_before) / (1024 * 1024)
    return rec, load_time, mem_used_mb

def profile_recommendation_time(rec, iterations=100, top_n=5):
    """
    Profiles the average recommendation generation time.
    Ensures that a test user with at least one liked song exists and then repeatedly
    calls the recommend() method, returning the average time per recommendation (in seconds).
    """
    test_user = "test_user"
    # For the POC version, the user record is expected to include 'fav_genres', 'liked_songs', and 'history'.
    # For the final version, it uses 'liked' and 'history'.
    from music_recommender import MusicRecommender as MusicRecommenderPOC
    if isinstance(rec, MusicRecommenderPOC):
        rec.users[test_user] = {'fav_genres': [], 'liked_songs': [], 'history': []}
    else:
        rec.users[test_user] = {'liked': set(), 'history': set()}
    
    # Simulate an interaction with the first song in the dataset.
    if rec.songs:
        first_song = list(rec.songs.keys())[0]
        rec.record_interaction(test_user, first_song)
    
    times = []
    for _ in range(iterations):
        start = time.time()
        rec.recommend(test_user, top_n=top_n)
        times.append(time.time() - start)
    avg_recommend_time = sum(times) / iterations
    return avg_recommend_time

def main():
    # --- Profile the POC implementation (CSV, ~200 songs) ---
    print("Profiling POC version (CSV, ~200 songs)...")
    poc_dataset = "songs.csv"  # CSV file for the POC version
    poc_rec, poc_load_time, poc_mem_used = profile_load_data(MusicRecommenderPOC, poc_dataset)
    poc_recommend_time = profile_recommendation_time(poc_rec)
    actual_poc_count = len(poc_rec.songs)
    scaling_factor = TARGET_SONG_COUNT / actual_poc_count
    # Extrapolate the POC metrics to the target dataset size.
    extrapolated_poc_load_time = poc_load_time * scaling_factor
    extrapolated_poc_mem_used = poc_mem_used * scaling_factor
    extrapolated_poc_recommend_time = poc_recommend_time * scaling_factor

    print(f"POC - Actual song count: {actual_poc_count}")
    print(f"POC - Load time: {poc_load_time:.3f} s, Memory used: {poc_mem_used:.2f} MB, "
          f"Recommendation time: {poc_recommend_time*1000:.2f} ms")
    print(f"Extrapolated POC (for {TARGET_SONG_COUNT} songs) - "
          f"Load time: {extrapolated_poc_load_time:.3f} s, Memory used: {extrapolated_poc_mem_used:.2f} MB, "
          f"Recommendation time: {extrapolated_poc_recommend_time*1000:.2f} ms")
    
    # --- Profile the Final implementation (SQLite, 1M songs) ---
    print("\nProfiling Final version (SQLite, 1M songs)...")
    final_dataset = "million_songs.db"  # SQLite DB for the final version
    final_rec, final_load_time, final_mem_used = profile_load_data(MusicRecommenderV1, final_dataset, batch_size=50000)
    final_recommend_time = profile_recommendation_time(final_rec)
    print(f"Final - Load time: {final_load_time:.3f} s, Memory used: {final_mem_used:.2f} MB, "
          f"Recommendation time: {final_recommend_time*1000:.2f} ms")
    
    # --- Plotting the results ---
    versions = ['Extrapolated POC', 'Final']
    load_times = [extrapolated_poc_load_time, final_load_time]
    mem_usages = [extrapolated_poc_mem_used, final_mem_used]
    recommend_times = [extrapolated_poc_recommend_time*1000, final_recommend_time*1000]  # in milliseconds
    
    # Plot Data Load Time
    plt.figure(figsize=(8, 6))
    plt.bar(versions, load_times, color=['skyblue', 'lightgreen'])
    plt.ylabel("Load Time (seconds)")
    plt.title("Data Load Time Comparison")
    plt.grid(True, axis='y')
    plt.show()
    
    # Plot Memory Usage
    plt.figure(figsize=(8, 6))
    plt.bar(versions, mem_usages, color=['skyblue', 'lightgreen'])
    plt.ylabel("Memory Used (MB)")
    plt.title("Memory Usage Comparison")
    plt.grid(True, axis='y')
    plt.show()
    
    # Plot Recommendation Generation Time
    plt.figure(figsize=(8, 6))
    plt.bar(versions, recommend_times, color=['skyblue', 'lightgreen'])
    plt.ylabel("Recommendation Time (ms)")
    plt.title("Recommendation Generation Time Comparison")
    plt.grid(True, axis='y')
    plt.show()

if __name__ == "__main__":
    main()
