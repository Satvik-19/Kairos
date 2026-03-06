"""Tests for EntropyPool."""
import threading
import time

from kairos.entropy.pool import EntropyPool


class TestEntropyPool:
    def test_feed_and_read(self):
        pool = EntropyPool(size=1024)
        data = bytes(range(32))
        pool.feed(data)
        result = pool.read(32)
        assert len(result) == 32

    def test_read_returns_correct_length(self):
        pool = EntropyPool(size=1024)
        pool.feed(b"\xAB" * 64)
        result = pool.read(64)
        assert len(result) == 64

    def test_ring_buffer_wraps(self):
        pool = EntropyPool(size=64)
        # Feed more than pool size
        for i in range(10):
            pool.feed(bytes([i] * 16))
        result = pool.read(32)
        assert len(result) == 32

    def test_read_all_returns_full_buffer(self):
        pool = EntropyPool(size=128)
        pool.feed(b"\xFF" * 128)
        result = pool.read_all()
        assert len(result) == 128

    def test_fill_percent_increases(self):
        pool = EntropyPool(size=1024)
        assert pool.fill_percent() == 0.0
        pool.feed(b"\x00" * 512)
        assert pool.fill_percent() == 50.0
        pool.feed(b"\x00" * 512)
        assert pool.fill_percent() == 100.0

    def test_fill_percent_caps_at_100(self):
        pool = EntropyPool(size=64)
        pool.feed(b"\x00" * 200)
        assert pool.fill_percent() == 100.0

    def test_thread_safe_concurrent_feed_read(self):
        pool = EntropyPool(size=1024)
        errors = []

        def feeder():
            for _ in range(100):
                try:
                    pool.feed(b"\xAB" * 32)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(100):
                try:
                    _ = pool.read(32)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=feeder) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
