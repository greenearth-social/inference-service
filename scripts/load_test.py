#!/usr/bin/env python3
"""Benchmark one deployed inference-service revision with controlled concurrency.

This is a closed-loop load generator: each client worker sends another request
as soon as its previous request completes. For per-instance concurrency tuning,
deploy Cloud Run with min-instances=max-instances=1, set the candidate Cloud Run
concurrency, and run this script with matching ``--client-concurrency``.

The script targets stage by default, reads the API key from
``GE_INFERENCE_API_KEY`` or Secret Manager, and refuses to benchmark a service
that can autoscale unless ``--allow-autoscaling`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Literal
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter


ModelType = Literal["user-tower", "post-tower", "ranker"]

DEFAULT_PROJECT_ID = "greenearth-471522"
DEFAULT_REGION = "us-east1"
DEFAULT_STAGE_URL = "https://inference-stage.greenearth.social"
DEFAULT_PROD_URL = "https://inference.greenearth.social"
DEFAULT_CANDIDATE_COUNT = 30

_thread_local = threading.local()


@dataclass(frozen=True)
class PayloadSpec:
    model: ModelType
    embed_dim: int
    history_length: int
    candidate_count: int
    expected_output_count: int
    body: str


@dataclass(frozen=True)
class RequestSample:
    request_id: str
    started_at: str
    latency_ms: float
    status_code: int | None
    error: str | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def percentile(values: list[float], percent: float) -> float | None:
    """Return a linearly interpolated percentile, or None for no samples."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def validate_target(base_url: str, allow_non_stage: bool) -> None:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError(f"Invalid base URL: {base_url}")

    is_local = hostname in {"127.0.0.1", "localhost", "::1"}
    is_stage = "stage" in hostname
    if not is_local and not is_stage and not allow_non_stage:
        raise ValueError(
            f"Refusing to load test non-stage target '{hostname}'. "
            "Pass --allow-non-stage only when that is intentional."
        )


def resolve_api_key(environment: str, project_id: str) -> tuple[str, str]:
    api_key = os.environ.get("GE_INFERENCE_API_KEY", "").strip()
    if api_key:
        return api_key, "GE_INFERENCE_API_KEY"

    if shutil.which("gcloud") is None:
        raise RuntimeError(
            "GE_INFERENCE_API_KEY is not set and gcloud is unavailable for Secret Manager lookup"
        )

    secret_name = f"inference-api-key-{environment}"
    result = subprocess.run(
        [
            "gcloud",
            "secrets",
            "versions",
            "access",
            "latest",
            f"--secret={secret_name}",
            f"--project={project_id}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown gcloud error"
        raise RuntimeError(f"Could not read Secret Manager secret '{secret_name}': {detail}")
    api_key = result.stdout.strip()
    if not api_key:
        raise RuntimeError(f"Secret Manager secret '{secret_name}' was empty")
    return api_key, f"Secret Manager ({secret_name})"


def describe_deployment(
    environment: str,
    project_id: str,
    region: str,
) -> dict[str, Any]:
    if shutil.which("gcloud") is None:
        raise RuntimeError("gcloud is unavailable")

    service_name = f"engagement-prediction-inference-{environment}"
    result = subprocess.run(
        [
            "gcloud",
            "run",
            "services",
            "describe",
            service_name,
            f"--project={project_id}",
            f"--region={region}",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown gcloud error"
        raise RuntimeError(f"Could not describe Cloud Run service '{service_name}': {detail}")

    raw = json.loads(result.stdout)
    template = raw.get("spec", {}).get("template", {})
    annotations = template.get("metadata", {}).get("annotations", {})
    spec = template.get("spec", {})
    return {
        "service": raw.get("metadata", {}).get("name", service_name),
        "revision": raw.get("status", {}).get("latestReadyRevisionName"),
        "concurrency": spec.get("containerConcurrency"),
        "min_instances": annotations.get("autoscaling.knative.dev/minScale"),
        "max_instances": annotations.get("autoscaling.knative.dev/maxScale"),
    }


def fetch_ready(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    response = requests.get(
        f"{base_url.rstrip('/')}/ready",
        headers={"X-API-Key": api_key},
        timeout=timeout,
    )
    if response.status_code != 200:
        body = response.text.strip()[:500]
        raise RuntimeError(f"Readiness check failed with HTTP {response.status_code}: {body}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Readiness response was not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ready") is not True:
        raise RuntimeError(f"Inference service is not ready: {payload}")
    return payload


def _model_summary(ready: dict[str, Any], model: ModelType) -> dict[str, Any]:
    models = ready.get("models")
    if not isinstance(models, list):
        raise ValueError("Readiness response is missing its models list")
    for entry in models:
        if isinstance(entry, dict) and entry.get("type") == model:
            if entry.get("ready") is not True:
                raise ValueError(f"Configured model '{model}' is not ready: {entry}")
            return entry
    raise ValueError(f"Model '{model}' is not configured on the target service")


def _vector(rng: random.Random, embed_dim: int) -> list[float]:
    # Six decimal places keeps the request representative while avoiding very
    # large JSON produced by full-precision Python float representations.
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(embed_dim)]


def build_payload(
    ready: dict[str, Any],
    model: ModelType,
    history_length_override: int | None,
    candidate_count: int,
    seed: int,
) -> PayloadSpec:
    embed_dim = ready.get("embed_dim")
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError(f"Readiness response has invalid embed_dim: {embed_dim}")

    model_summary = _model_summary(ready, model)
    max_history_length = model_summary.get("max_history_len")
    if model in {"user-tower", "ranker"}:
        if history_length_override is not None:
            history_length = history_length_override
        elif isinstance(max_history_length, int) and max_history_length > 0:
            history_length = max_history_length
        else:
            raise ValueError(
                f"Model '{model}' did not report a usable max_history_len; "
                "pass --history-length explicitly"
            )
    else:
        history_length = 0

    rng = random.Random(seed)
    if model == "user-tower":
        body: dict[str, Any] = {
            "history_embeddings": [_vector(rng, embed_dim) for _ in range(history_length)],
            "history_author_dids": [
                f"did:plc:benchmark-history-{index}" for index in range(history_length)
            ],
        }
        expected_output_count = 1
        effective_candidate_count = 0
    elif model == "post-tower":
        body = {
            "post_embeddings": [_vector(rng, embed_dim) for _ in range(candidate_count)],
            "target_author_dids": [
                f"did:plc:benchmark-candidate-{index}" for index in range(candidate_count)
            ],
        }
        expected_output_count = candidate_count
        effective_candidate_count = candidate_count
    else:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        body = {
            "history_embeddings": [_vector(rng, embed_dim) for _ in range(history_length)],
            "history_author_dids": [
                f"did:plc:benchmark-history-{index}" for index in range(history_length)
            ],
            "history_liked_at_times": [
                (now - timedelta(hours=index + 1)).isoformat()
                for index in range(history_length)
            ],
            "history_prior_cumulative_likes": [index for index in range(history_length)],
            "candidate_post_embeddings": [
                _vector(rng, embed_dim) for _ in range(candidate_count)
            ],
            "candidate_author_dids": [
                f"did:plc:benchmark-candidate-{index}" for index in range(candidate_count)
            ],
            "candidate_prior_cumulative_likes": [index for index in range(candidate_count)],
        }
        expected_output_count = candidate_count
        effective_candidate_count = candidate_count

    return PayloadSpec(
        model=model,
        embed_dim=embed_dim,
        history_length=history_length,
        candidate_count=effective_candidate_count,
        expected_output_count=expected_output_count,
        body=json.dumps(body, separators=(",", ":")),
    )


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        # A worker issues one request at a time. A one-connection pool per
        # worker preserves keep-alive without hiding client-side queueing.
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


def send_prediction(
    *,
    url: str,
    api_key: str,
    payload: PayloadSpec,
    timeout: float,
    request_id: str,
) -> RequestSample:
    started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    started = time.perf_counter()
    status_code: int | None = None
    error: str | None = None
    try:
        response = _session().post(
            url,
            data=payload.body,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "X-Request-ID": request_id,
            },
            timeout=timeout,
        )
        status_code = response.status_code
        if status_code != 200:
            error = f"HTTP {status_code}: {response.text.strip()[:300]}"
        else:
            try:
                response_payload = response.json()
            except ValueError:
                error = "HTTP 200 response was not valid JSON"
            else:
                outputs = response_payload.get("outputs") if isinstance(response_payload, dict) else None
                if not isinstance(outputs, list):
                    error = "HTTP 200 response did not contain an outputs list"
                elif len(outputs) != payload.expected_output_count:
                    error = (
                        f"HTTP 200 returned {len(outputs)} outputs; "
                        f"expected {payload.expected_output_count}"
                    )
    except requests.RequestException as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000.0
    return RequestSample(
        request_id=request_id,
        started_at=started_at,
        latency_ms=latency_ms,
        status_code=status_code,
        error=error,
    )


def summarize(samples: list[RequestSample], elapsed_seconds: float) -> dict[str, Any]:
    successes = [sample for sample in samples if sample.status_code == 200 and sample.error is None]
    success_latencies = [sample.latency_ms for sample in successes]
    status_counts = Counter(
        str(sample.status_code) if sample.status_code is not None else "connection-error"
        for sample in samples
    )
    error_counts = Counter(sample.error for sample in samples if sample.error is not None)
    return {
        "requests": len(samples),
        "successes": len(successes),
        "errors": len(samples) - len(successes),
        "elapsed_seconds": elapsed_seconds,
        "requests_per_second": len(samples) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "successful_requests_per_second": (
            len(successes) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "latency_ms": {
            "min": min(success_latencies) if success_latencies else None,
            "p50": percentile(success_latencies, 50),
            "p95": percentile(success_latencies, 95),
            "p99": percentile(success_latencies, 99),
            "max": max(success_latencies) if success_latencies else None,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "error_counts": dict(error_counts.most_common(10)),
    }


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def print_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency_ms"]
    print("\nResults")
    print(f"  Requests:       {summary['requests']}")
    print(f"  Successes:      {summary['successes']}")
    print(f"  Errors:         {summary['errors']}")
    print(f"  Elapsed:        {summary['elapsed_seconds']:.2f} s")
    print(f"  Throughput:     {summary['requests_per_second']:.2f} req/s")
    print(f"  Latency p50:    {_format_metric(latency['p50'])} ms")
    print(f"  Latency p95:    {_format_metric(latency['p95'])} ms")
    print(f"  Latency p99:    {_format_metric(latency['p99'])} ms")
    print(f"  Latency min/max:{_format_metric(latency['min'])}/{_format_metric(latency['max'])} ms")
    print(f"  Statuses:       {summary['status_counts']}")
    if summary["error_counts"]:
        print(f"  Top errors:     {summary['error_counts']}")


def run_load(
    *,
    url: str,
    api_key: str,
    payload: PayloadSpec,
    timeout: float,
    duration: float,
    client_concurrency: int,
    run_id: str,
) -> tuple[list[RequestSample], float]:
    request_numbers = itertools.count()
    started = time.perf_counter()
    deadline = started + duration

    def worker(worker_number: int) -> list[RequestSample]:
        worker_samples: list[RequestSample] = []
        while time.perf_counter() < deadline:
            request_number = next(request_numbers)
            worker_samples.append(
                send_prediction(
                    url=url,
                    api_key=api_key,
                    payload=payload,
                    timeout=timeout,
                    request_id=f"inference-benchmark-{run_id}-{worker_number}-{request_number}",
                )
            )
        return worker_samples

    with ThreadPoolExecutor(max_workers=client_concurrency) as executor:
        futures = [executor.submit(worker, worker_number) for worker_number in range(client_concurrency)]
        samples = [sample for future in futures for sample in future.result()]

    return samples, time.perf_counter() - started


def write_results(
    path: Path,
    *,
    run_id: str,
    args: argparse.Namespace,
    deployment: dict[str, Any] | None,
    payload: PayloadSpec,
    summary: dict[str, Any],
    samples: list[RequestSample],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": run_id,
        "target": {
            "base_url": args.base_url,
            "environment": args.environment,
            "deployment": deployment,
        },
        "load": {
            "client_concurrency": args.client_concurrency,
            "requested_duration_seconds": args.duration,
            "warmup_requests": args.warmup_requests,
            "timeout_seconds": args.timeout,
        },
        "payload": {
            "model": payload.model,
            "embed_dim": payload.embed_dim,
            "history_length": payload.history_length,
            "candidate_count": payload.candidate_count,
            "body_bytes": len(payload.body.encode("utf-8")),
            "seed": args.seed,
        },
        "summary": summary,
        "samples": [asdict(sample) for sample in samples],
    }
    path.write_text(json.dumps(result, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a deployed inference service at controlled client concurrency",
    )
    parser.add_argument("--environment", choices=["stage", "prod"], default="stage")
    parser.add_argument("--base-url", help="Inference base URL (defaults from --environment)")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--model",
        choices=["user-tower", "post-tower", "ranker"],
        default="ranker",
        help="Model endpoint to exercise (default: ranker)",
    )
    parser.add_argument(
        "--client-concurrency",
        type=positive_int,
        default=1,
        help="Simultaneous requests generated by this client (default: 1)",
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=60.0,
        help="Measured load duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=10,
        help="Sequential requests before measurement (default: 10)",
    )
    parser.add_argument(
        "--history-length",
        type=positive_int,
        help="Synthetic history length (default: selected model max from /ready)",
    )
    parser.add_argument(
        "--candidate-count",
        type=positive_int,
        default=DEFAULT_CANDIDATE_COUNT,
        help="Synthetic candidate/post batch size (default: 30, matching feed configuration)",
    )
    parser.add_argument("--timeout", type=positive_float, default=30.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--expected-cloud-concurrency",
        type=positive_int,
        help="Fail unless the deployed Cloud Run concurrency matches this value",
    )
    parser.add_argument(
        "--allow-autoscaling",
        action="store_true",
        help="Allow max-instances other than 1; use only for the later autoscaling test",
    )
    parser.add_argument(
        "--allow-non-stage",
        action="store_true",
        help="Permit load testing a URL whose hostname does not contain 'stage'",
    )
    parser.add_argument(
        "--skip-deployment-check",
        action="store_true",
        help="Skip the gcloud concurrency and instance-limit safety check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result JSON path (default: benchmark-results/<run-id>.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate deployment, credentials, readiness, and payload without sending predictions",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")

    default_url = DEFAULT_STAGE_URL if args.environment == "stage" else DEFAULT_PROD_URL
    args.base_url = (args.base_url or default_url).rstrip("/")
    try:
        validate_target(args.base_url, args.allow_non_stage)
    except ValueError as exc:
        parser.error(str(exc))

    hostname = (urlparse(args.base_url).hostname or "").lower()
    is_local = hostname in {"127.0.0.1", "localhost", "::1"}
    deployment: dict[str, Any] | None = None
    if not args.skip_deployment_check and not is_local:
        try:
            deployment = describe_deployment(args.environment, args.project_id, args.region)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

        deployed_concurrency = deployment.get("concurrency")
        deployed_max = deployment.get("max_instances")
        if (
            args.expected_cloud_concurrency is not None
            and deployed_concurrency != args.expected_cloud_concurrency
        ):
            parser.error(
                "Deployed Cloud Run concurrency is "
                f"{deployed_concurrency}, expected {args.expected_cloud_concurrency}"
            )
        if str(deployed_max) != "1" and not args.allow_autoscaling:
            parser.error(
                f"Deployed max-instances is {deployed_max!r}, not 1. "
                "Pin the service to one instance for per-instance tuning, or pass "
                "--allow-autoscaling for the later scaling test."
            )

    try:
        api_key, api_key_source = resolve_api_key(args.environment, args.project_id)
        ready = fetch_ready(args.base_url, api_key, args.timeout)
        payload = build_payload(
            ready,
            args.model,
            args.history_length,
            args.candidate_count,
            args.seed,
        )
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        parser.error(str(exc))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prediction_url = f"{args.base_url}/models/{args.model}/predict"
    print(f"Target:              {args.base_url}")
    if deployment is not None:
        print(f"Revision:            {deployment.get('revision')}")
        print(f"Cloud concurrency:   {deployment.get('concurrency')}")
        print(
            "Cloud instances:     "
            f"min={deployment.get('min_instances')} max={deployment.get('max_instances')}"
        )
    print(f"API key source:      {api_key_source}")
    print(f"Model:               {payload.model}")
    print(f"Embedding dimension: {payload.embed_dim}")
    print(f"History length:      {payload.history_length}")
    print(f"Candidate count:     {payload.candidate_count}")
    print(f"Request body:        {len(payload.body.encode('utf-8')):,} bytes")
    print(f"Client concurrency:  {args.client_concurrency}")

    if args.dry_run:
        print("\nDry run complete; no prediction requests were sent.")
        return 0

    print(f"\nSending {args.warmup_requests} sequential warmup requests...")
    for index in range(args.warmup_requests):
        sample = send_prediction(
            url=prediction_url,
            api_key=api_key,
            payload=payload,
            timeout=args.timeout,
            request_id=f"inference-benchmark-{run_id}-warmup-{index}",
        )
        if sample.error is not None:
            print(f"Warmup request {index + 1} failed: {sample.error}", file=sys.stderr)
            return 1

    print(f"Running measured load for {args.duration:.1f} seconds...")
    samples, elapsed = run_load(
        url=prediction_url,
        api_key=api_key,
        payload=payload,
        timeout=args.timeout,
        duration=args.duration,
        client_concurrency=args.client_concurrency,
        run_id=run_id,
    )
    summary = summarize(samples, elapsed)
    print_summary(summary)

    output = args.output or Path("benchmark-results") / f"{run_id}-{args.model}.json"
    write_results(
        output,
        run_id=run_id,
        args=args,
        deployment=deployment,
        payload=payload,
        summary=summary,
        samples=samples,
    )
    print(f"\nWrote {output}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
