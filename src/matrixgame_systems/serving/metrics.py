from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    REQUESTS = Counter("mgs_requests_total", "Generation requests", ["status"])
    LATENCY = Histogram(
        "mgs_request_latency_seconds",
        "End-to-end request latency",
        buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280),
    )
    QUEUE_WAIT = Histogram(
        "mgs_queue_wait_seconds", "Time waiting before execution", buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 20)
    )
    QUEUE_DEPTH = Gauge("mgs_queue_depth", "Current queued requests")
    ACTIVE = Gauge("mgs_active_requests", "Current active requests")
    BATCH_SIZE = Histogram("mgs_batch_size", "Dispatched compatible batch size", buckets=(1, 2, 4, 8, 16))
    HBM_RESERVED = Gauge("mgs_hbm_reserved_bytes", "Admission-controller HBM reservations")

    def render_metrics() -> bytes:
        return generate_latest()

except ImportError:  # pragma: no cover
    class _Noop:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def dec(self, *args, **kwargs):
            return None

        def set(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

    REQUESTS = LATENCY = QUEUE_WAIT = QUEUE_DEPTH = ACTIVE = BATCH_SIZE = HBM_RESERVED = _Noop()

    def render_metrics() -> bytes:
        return b""
