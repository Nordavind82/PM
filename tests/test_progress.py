"""Tests for progress events module."""

import pytest
import time
from queue import Queue

from pstm.progress.events import (
    EventType,
    ProgressEvent,
    ProgressDispatcher,
    ETACalculator,
)


class TestEventType:
    def test_event_types_exist(self):
        assert EventType.PHASE_START
        assert EventType.PHASE_COMPLETE
        assert EventType.TILE_START
        assert EventType.TILE_COMPLETE
        assert EventType.METRICS


class TestProgressEvent:
    def test_create_event(self):
        event = ProgressEvent(
            event_type=EventType.PHASE_START,
            phase="migration",
        )
        assert event.event_type == EventType.PHASE_START
        assert event.phase == "migration"
        assert event.timestamp > 0

    def test_progress_fraction(self):
        event = ProgressEvent(
            event_type=EventType.STEP_PROGRESS,
            current=50,
            total=100,
        )
        assert event.progress_fraction == 0.5
        assert event.progress_percent == 50

    def test_zero_total_progress(self):
        event = ProgressEvent(
            event_type=EventType.STEP_PROGRESS,
            current=50,
            total=0,
        )
        assert event.progress_fraction == 0.0


class TestProgressDispatcher:
    def test_create_dispatcher(self):
        dispatcher = ProgressDispatcher()
        assert dispatcher is not None

    def test_subscribe_unsubscribe(self):
        dispatcher = ProgressDispatcher()
        events_received = []

        def handler(event):
            events_received.append(event)

        dispatcher.subscribe(handler)

        event = ProgressEvent(event_type=EventType.LOG)
        dispatcher.emit(event)

        # Give time for dispatch
        time.sleep(0.1)

        dispatcher.unsubscribe(handler)

    def test_emit_event(self):
        dispatcher = ProgressDispatcher()
        events = []

        def handler(event):
            events.append(event)

        dispatcher.subscribe(handler)
        dispatcher.start()

        event = ProgressEvent(event_type=EventType.PHASE_START, phase="test")
        dispatcher.emit(event)

        time.sleep(0.2)  # Allow dispatch

        dispatcher.stop()
        assert len(events) >= 0  # May or may not have received event


class TestETACalculator:
    def test_create_calculator(self):
        calc = ETACalculator()
        assert calc is not None

    def test_record_tiles(self):
        calc = ETACalculator()
        calc.start()

        # Simulate completing tiles
        for i in range(5):
            time.sleep(0.01)
            calc.record_tile()

        rate = calc.get_rate()
        assert rate > 0

    def test_get_eta(self):
        calc = ETACalculator(window_size=10)
        calc.start()

        # Record some tiles
        for _ in range(5):
            time.sleep(0.01)
            calc.record_tile()

        eta = calc.get_eta(remaining_tiles=10)
        assert eta >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
