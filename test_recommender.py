import pytest
import time
import psutil
from music_recommender_v1 import MusicRecommender

class TestMusicRecommender:
    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Set up a MusicRecommender instance using a smaller batch size for testing.
        This fixture automatically runs before each test method.
        """
        # Instantiate the recommender with your database (assumed to be 'million_songs.db')
        self.rec = MusicRecommender("million_songs.db")
        # Use a smaller batch size for faster testing.
        self.rec.load_data(batch_size=1000)
        yield

    @pytest.mark.slow
    def test_load_data(self):
        """Test that songs are loaded from the database."""
        assert len(self.rec.songs) > 0, "Songs should be loaded"


    @pytest.mark.slow
    def test_recommend(self):
        """Test that recommendations are generated for a user."""
        # Simulate a user interaction.
        song_id = list(self.rec.songs.keys())[0]
        self.rec.record_interaction("user1", song_id)
        recommendations = self.rec.recommend("user1", top_n=5)
        assert len(recommendations) > 0, "Recommendations should be generated"

    @pytest.mark.slow
    def test_memory_usage(self):
        """Test that memory usage does not exceed a threshold during data load."""
        process = psutil.Process()
        memory_before = process.memory_info().rss
        rec2 = MusicRecommender("million_songs.db")
        rec2.load_data(batch_size=1000)  # Using a small batch for testing purposes.
        memory_after = process.memory_info().rss
        # Assert that memory increase is under 500 MB.
        assert (memory_after - memory_before) < 500 * 1024 * 1024, "Memory usage is too high"

    @pytest.mark.slow
    def test_stress(self):
        """Stress test: ensure that loading a large batch completes in under 60 seconds."""
        rec3 = MusicRecommender("million_songs.db")
        start_time = time.time()
        rec3.load_data(batch_size=50000)
        duration = time.time() - start_time
        assert duration < 60, f"Loading took too long: {duration:.2f} seconds"
