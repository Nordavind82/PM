"""Tests for I/O optimization module."""

import numpy as np
import pytest
import time
import threading

from pstm.utils.io_optimization import (
    BufferSlot,
    DoubleBuffer,
    Prefetcher,
    IOStats,
)


class TestBufferSlot:
    def test_create_slot(self):
        slot = BufferSlot()
        assert slot.data is None
        assert not slot.in_use

    def test_clear_slot(self):
        slot = BufferSlot()
        slot.data = "test"
        slot.in_use = True
        slot.metadata["key"] = "value"
        slot.ready.set()

        slot.clear()

        assert slot.data is None
        assert not slot.in_use
        assert len(slot.metadata) == 0
        assert not slot.ready.is_set()


class TestDoubleBuffer:
    def test_create_buffer(self):
        buffer = DoubleBuffer(n_slots=2)
        assert buffer.n_slots == 2
        assert len(buffer.slots) == 2

    def test_get_fill_slot(self):
        buffer = DoubleBuffer(n_slots=2)
        slot_id = buffer.get_fill_slot()
        assert slot_id is not None
        assert 0 <= slot_id < 2

    def test_fill_and_ready(self):
        buffer = DoubleBuffer(n_slots=2)

        slot_id = buffer.get_fill_slot()
        buffer.begin_fill(slot_id)
        buffer.slots[slot_id].data = "test_data"
        buffer.end_fill(slot_id)

        ready_id = buffer.wait_for_ready(timeout=1.0)
        assert ready_id == slot_id
        assert buffer.slots[ready_id].data == "test_data"

    def test_release(self):
        buffer = DoubleBuffer(n_slots=2)

        slot_id = buffer.get_fill_slot()
        buffer.begin_fill(slot_id)
        buffer.end_fill(slot_id)

        ready_id = buffer.wait_for_ready(timeout=1.0)
        buffer.release(ready_id)

        assert not buffer.slots[ready_id].in_use
        assert buffer.slots[ready_id].data is None

    def test_is_full_empty(self):
        buffer = DoubleBuffer(n_slots=2)

        assert buffer.is_empty()
        assert not buffer.is_full()

        # Fill all slots
        for _ in range(2):
            slot_id = buffer.get_fill_slot()
            if slot_id is not None:
                buffer.begin_fill(slot_id)
                buffer.end_fill(slot_id)

        # Now should be full
        assert buffer.is_full()


class TestPrefetcher:
    def test_create_prefetcher(self):
        def load(idx):
            return idx * 2

        prefetcher = Prefetcher(load, n_prefetch=2)
        assert prefetcher.n_prefetch == 2

    def test_get_item(self):
        def load(idx):
            return f"item_{idx}"

        prefetcher = Prefetcher(load, n_prefetch=2)
        result = prefetcher.get(5)
        assert result == "item_5"

    def test_prefetch(self):
        load_count = {"count": 0}

        def load(idx):
            load_count["count"] += 1
            return idx * 2

        prefetcher = Prefetcher(load, n_prefetch=3)
        prefetcher.prefetch([1, 2, 3])

        time.sleep(0.2)  # Allow prefetch

        # Items should be cached
        result1 = prefetcher.get(1)
        result2 = prefetcher.get(2)

        assert result1 == 2
        assert result2 == 4

        prefetcher.shutdown()


class TestIOStats:
    def test_create_stats(self):
        stats = IOStats()
        assert stats.reads == 0
        assert stats.writes == 0

    def test_record_read(self):
        stats = IOStats()
        stats.record_read(n_bytes=1024, elapsed_s=0.1)

        assert stats.reads == 1
        assert stats.read_bytes == 1024
        assert stats.read_time_s == 0.1

    def test_record_write(self):
        stats = IOStats()
        stats.record_write(n_bytes=2048, elapsed_s=0.2)

        assert stats.writes == 1
        assert stats.write_bytes == 2048
        assert stats.write_time_s == 0.2

    def test_throughput(self):
        stats = IOStats()
        stats.record_read(n_bytes=10 * 1024 * 1024, elapsed_s=1.0)  # 10 MB in 1 second

        assert abs(stats.read_throughput_mbps - 10.0) < 0.1

    def test_get_summary(self):
        stats = IOStats()
        stats.record_read(n_bytes=1024, elapsed_s=0.1)
        stats.record_write(n_bytes=2048, elapsed_s=0.2)

        summary = stats.get_summary()
        assert "reads" in summary
        assert "writes" in summary
        assert "read_throughput_mbps" in summary


class TestThreadSafety:
    def test_concurrent_io_stats(self):
        stats = IOStats()

        def record_reads():
            for _ in range(100):
                stats.record_read(1024, 0.01)

        def record_writes():
            for _ in range(100):
                stats.record_write(2048, 0.01)

        t1 = threading.Thread(target=record_reads)
        t2 = threading.Thread(target=record_writes)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert stats.reads == 100
        assert stats.writes == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
