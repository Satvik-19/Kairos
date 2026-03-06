"""Thread-safe ring-buffer entropy pool."""
import threading


class EntropyPool:
    """
    Thread-safe ring buffer for entropy bytes.
    Size: 1024 bytes by default.
    """

    def __init__(self, size: int = 1024):
        self._size = size
        self._buffer = bytearray(size)
        self._write_ptr = 0
        self._bytes_fed = 0
        self._lock = threading.Lock()

    def feed(self, data: bytes):
        """Write bytes into the ring buffer at the write pointer, wrapping around."""
        with self._lock:
            for byte in data:
                self._buffer[self._write_ptr] = byte
                self._write_ptr = (self._write_ptr + 1) % self._size
            self._bytes_fed += len(data)

    def read(self, n: int) -> bytes:
        """Read n bytes from the buffer (from current write position, going back)."""
        with self._lock:
            if n >= self._size:
                return bytes(self._buffer)
            # Read the most recently written n bytes
            end = self._write_ptr
            start = (end - n) % self._size
            if start < end:
                return bytes(self._buffer[start:end])
            else:
                # Wrap-around read
                return bytes(self._buffer[start:]) + bytes(self._buffer[:end])

    def read_all(self) -> bytes:
        """Return a snapshot of the entire buffer."""
        with self._lock:
            return bytes(self._buffer)

    def fill_percent(self) -> float:
        """
        Return the current write-head position as a percentage of pool size.
        Cycles 0→100% each time the ring buffer completes a full rotation,
        giving a live activity indicator rather than a static 'always full' value.
        """
        with self._lock:
            return (self._write_ptr / self._size) * 100.0
