"""Tests for MetricCollector."""

from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from metrics import MetricCollector, get_metric_collector, set_metric_collector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collector(service_name: str = "test-svc", env: str = "test") -> tuple[MetricCollector, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    collector = MetricCollector._from_reader(reader, service_name=service_name, env=env)
    return collector, reader


def _get_metrics_data(reader: InMemoryMetricReader):
    data = reader.get_metrics_data()
    assert data is not None
    return data


def _collect_names_from_data(data) -> set[str]:
    names: set[str] = set()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.add(metric.name)
    return names


# ---------------------------------------------------------------------------
# Instrument type inference
# ---------------------------------------------------------------------------

def test_counter_inferred_for_count_suffix():
    collector, reader = _make_collector()
    collector.record("requests_count", 5)
    data = _get_metrics_data(reader)
    found = False
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "requests_count":
                    from opentelemetry.sdk.metrics._internal.point import Sum
                    assert isinstance(metric.data, Sum)
                    found = True
    assert found, "requests_count not found in exported metrics"


def test_gauge_inferred_for_rate_suffix():
    collector, reader = _make_collector()
    collector.record("throughput_rate", 42.5)
    data = _get_metrics_data(reader)
    found = False
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "throughput_rate":
                    from opentelemetry.sdk.metrics._internal.point import Gauge
                    assert isinstance(metric.data, Gauge)
                    found = True
    assert found, "throughput_rate not found in exported metrics"


def test_histogram_inferred_for_ms_suffix():
    collector, reader = _make_collector()
    collector.record("inference.predict.duration_ms", 123.4)
    data = _get_metrics_data(reader)
    found = False
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "inference.predict.duration_ms":
                    from opentelemetry.sdk.metrics._internal.point import Histogram
                    assert isinstance(metric.data, Histogram)
                    found = True
    assert found, "inference.predict.duration_ms not found in exported metrics"


def test_histogram_inferred_for_arbitrary_name():
    collector, reader = _make_collector()
    collector.record("something.latency", 99.0)
    data = _get_metrics_data(reader)
    assert _collect_names_from_data(data) == {"something.latency"}


# ---------------------------------------------------------------------------
# Attributes (labels)
# ---------------------------------------------------------------------------

def test_attributes_attached_to_histogram():
    collector, reader = _make_collector()
    collector.record("inference.predict.duration_ms", 50.0, model_name="user_tower")
    data = _get_metrics_data(reader)
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "inference.predict.duration_ms":
                    dp = metric.data.data_points[0]
                    attrs = dp.attributes or {}
                    assert attrs.get("model_name") == "user_tower"


def test_record_histogram_with_model_label():
    reader = InMemoryMetricReader()
    collector = MetricCollector._from_reader(reader, "inference", "test")
    collector.record("inference.predict.duration_ms", 12.5, model_name="user_tower")
    data = reader.get_metrics_data()
    [metric] = [
        m
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    ]
    assert metric.name == "inference.predict.duration_ms"
    [point] = list(metric.data.data_points)
    assert point.attributes == {"model_name": "user_tower"}


# ---------------------------------------------------------------------------
# Lazy instrument reuse
# ---------------------------------------------------------------------------

def test_same_instrument_reused_across_calls():
    collector, reader = _make_collector()
    collector.record("inference.predict.duration_ms", 10.0)
    collector.record("inference.predict.duration_ms", 20.0)
    data = _get_metrics_data(reader)
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "inference.predict.duration_ms":
                    from opentelemetry.sdk.metrics._internal.point import Histogram
                    assert isinstance(metric.data, Histogram)
                    # Both values should be in the same histogram
                    dp = metric.data.data_points[0]
                    assert dp.count == 2


# ---------------------------------------------------------------------------
# GCP exporter construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", ["stage", "prod"])
def test_gcp_exporter_uses_unique_identifier(env):
    """Cloud Run scales greenearth-inference to multiple concurrent
    instances, each running its own exporter. Without a unique identifier per
    exporter, two instances exporting the same metric+label combination in
    the same interval collide on GCP's cumulative point ordering and the
    whole batch write is rejected (see issue #263)."""
    with patch(
        "opentelemetry.exporter.cloud_monitoring.CloudMonitoringMetricsExporter"
    ) as mock_exporter_cls:
        MetricCollector(
            service_name="test-svc",
            env=env,
            export_interval_sec=60,
        )
    _, kwargs = mock_exporter_cls.call_args
    assert kwargs.get("add_unique_identifier") is True


# ---------------------------------------------------------------------------
# Local/dev (non-deployed environment)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_env_records_without_exporting(capsys):
    """Local/dev should record metrics without printing a resource_metrics blob."""
    collector = MetricCollector(
        service_name="test-svc",
        env="local",
        export_interval_sec=60,
    )
    collector.record("some.metric_ms", 1.0)
    # Force a flush; nothing should be written to stdout/stderr in dev.
    await collector.shutdown()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "resource_metrics" not in captured.out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_set_and_get_metric_collector():
    collector, _ = _make_collector()
    set_metric_collector(collector)
    assert get_metric_collector() is collector
    set_metric_collector(None)
    assert get_metric_collector() is None


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shutdown_does_not_raise():
    collector, _ = _make_collector()
    await collector.shutdown()


# ---------------------------------------------------------------------------
# Histogram bucket boundaries
#
# Ported from the api's metrics module: the OTel defaults leave four buckets
# above 1s, so a predict-latency p95 in the serving range is an interpolation
# across a multi-second bucket.
# ---------------------------------------------------------------------------

from metrics import (  # noqa: E402
    LATENCY_MS_BOUNDARIES,
    histogram_boundaries,
)


def test_predict_duration_uses_latency_boundaries():
    assert histogram_boundaries("inference.predict.duration_ms") == LATENCY_MS_BOUNDARIES


def test_unknown_metric_falls_back_to_sdk_default():
    assert histogram_boundaries("something.unrecognised") is None


def test_predict_histogram_exports_custom_bounds():
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    collector = MetricCollector._from_reader(reader, "inference", "test")
    collector.record("inference.predict.duration_ms", 2425.0, model_name="user-tower")

    [metric] = [
        m
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    ]
    [point] = list(metric.data.data_points)
    assert tuple(point.explicit_bounds) == tuple(LATENCY_MS_BOUNDARIES)


def test_multi_second_values_resolve_to_sub_500ms_buckets():
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    collector = MetricCollector._from_reader(reader, "inference", "test")
    for value in (2425.0, 4881.0):
        collector.record("inference.predict.duration_ms", value)

    [metric] = [
        m
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    ]
    [point] = list(metric.data.data_points)
    bounds = list(point.explicit_bounds)
    occupied = [i for i, count in enumerate(point.bucket_counts) if count]
    assert len(occupied) == 2
    for index in occupied:
        lower = bounds[index - 1] if index else 0
        upper = bounds[index] if index < len(bounds) else float("inf")
        assert upper - lower <= 500
