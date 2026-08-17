from container_id.streams.queue import LatestFrameQueue


def test_latest_frame_queue_puts_and_gets():
    q = LatestFrameQueue(maxsize=3)
    assert q.empty()
    q.put(1)
    assert not q.empty()
    assert q.get() == 1


def test_latest_frame_queue_drops_oldest():
    q = LatestFrameQueue(maxsize=2)
    q.put(1)
    q.put(2)
    q.put(3)
    # The oldest (1) should be dropped
    assert q.get() == 2
    assert q.get() == 3
    assert q.empty()
