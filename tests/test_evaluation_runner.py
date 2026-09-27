"""Fake-provider runs exercise the production pipeline with no network access."""

import asyncio
import json
from decimal import Decimal

import pytest

from app.classification.provider import ErrorKind, ProviderError, ProviderResult, TokenUsage
from app.config import Settings
from evaluation.artifacts import (
    EXPECTED_IDS,
    MESSAGES_PATH,
    REFERENCES_PATH,
    atomic_json,
    export_results,
    load_run,
    sha256_file,
)
from evaluation.cli import main
from evaluation.runner import evaluate, usage_summary
from tests.fakes import RecordingFakeClassifier


def test_live_cli_guard_never_reads_configuration_or_creates_a_run(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Configuration and provider access require explicit live intent.")

    monkeypatch.setattr("evaluation.cli.Settings.from_env", forbidden)
    monkeypatch.setattr("evaluation.cli.RUNS_DIR", tmp_path / "runs")
    assert main(["run", "--prompt-version", "v1", "--run-id", "guarded"]) == 2
    assert not (tmp_path / "runs").exists()


def test_fake_run_uses_shared_pipeline_masking_and_durable_artifacts(tmp_path, valid_model_output):
    async def scenario():
        frozen_before = {
            path: sha256_file(path)
            for path in (
                MESSAGES_PATH,
                REFERENCES_PATH,
                MESSAGES_PATH.parent.parent / "prompts/v1.md",
            )
        }
        fake = RecordingFakeClassifier(valid_model_output)
        settings = Settings(model="test-model", concurrency_limit=3, max_attempts=1)
        run = await evaluate(
            run_id="fake-v1",
            prompt_version="v1",
            settings=settings,
            classifier=fake,
            runs_dir=tmp_path,
        )
        assert [item["id"] for item in run["outcomes"]] == EXPECTED_IDS
        assert all(item["state"] == "succeeded" for item in run["outcomes"])
        assert len(fake.calls) == 12
        assert [version for _, version in fake.calls] == ["v1"] * 12
        assert [message for message, _ in fake.calls] == [
            item["message"] for item in json.loads(MESSAGES_PATH.read_text(encoding="utf-8"))
        ]
        assert all(item["result"]["metadata"]["model"] == "test-model" for item in run["outcomes"])
        run_dir = tmp_path / "fake-v1"
        assert load_run(run_dir) == run
        assert (run_dir / "report.json").exists()
        assert len((run_dir / "model_calls.jsonl").read_text(encoding="utf-8").splitlines()) == 12
        assert not (run_dir / ".run.lock").exists()
        serialized = (run_dir / "run.json").read_text(encoding="utf-8")
        assert "Hola equipo, un vendedor" not in serialized
        assert "You classify internal requests" not in serialized
        assert {path: sha256_file(path) for path in frozen_before} == frozen_before

    asyncio.run(scenario())


def test_checkpoint_is_written_before_provider_call(tmp_path, valid_model_output):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedClassifier(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                entered.set()
                await release.wait()
                return await super().classify(masked_message=masked_message, prompt=prompt)

        task = asyncio.create_task(
            evaluate(
                run_id="gated",
                prompt_version="v1",
                settings=Settings(model="test-model", concurrency_limit=1, max_attempts=1),
                classifier=GatedClassifier(valid_model_output),
                runs_dir=tmp_path,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            checkpoint = load_run(tmp_path / "gated")
            assert checkpoint["outcomes"][0] == {"id": "MSG-01", "state": "running"}
            assert checkpoint["outcomes"][1] == {"id": "MSG-02", "state": "pending"}
        finally:
            release.set()
        assert len((await task)["outcomes"]) == 12

    asyncio.run(scenario())


def test_worker_pool_and_pipeline_limit_concurrent_provider_calls(tmp_path, valid_model_output):
    async def scenario():
        three_entered = asyncio.Event()
        release = asyncio.Event()

        class BoundedClassifier(RecordingFakeClassifier):
            active = 0
            maximum_active = 0

            async def classify(self, *, masked_message, prompt):
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                if self.active == 3:
                    three_entered.set()
                try:
                    await release.wait()
                    return await super().classify(masked_message=masked_message, prompt=prompt)
                finally:
                    self.active -= 1

        fake = BoundedClassifier(valid_model_output)
        task = asyncio.create_task(
            evaluate(
                run_id="bounded",
                prompt_version="v1",
                settings=Settings(model="test-model", concurrency_limit=3, max_attempts=1),
                classifier=fake,
                runs_dir=tmp_path,
            )
        )
        try:
            await asyncio.wait_for(three_entered.wait(), timeout=3)
            assert fake.maximum_active == 3
            assert (
                sum(row["state"] == "running" for row in load_run(tmp_path / "bounded")["outcomes"])
                == 3
            )
        finally:
            release.set()
        assert all(row["state"] == "succeeded" for row in (await task)["outcomes"])
        assert fake.maximum_active == 3

    asyncio.run(scenario())


def test_isolated_provider_failure_is_sanitized_and_later_messages_continue(
    tmp_path, valid_model_output
):
    async def scenario():
        first_message = json.loads(MESSAGES_PATH.read_text(encoding="utf-8"))[0]["message"]

        class OneFailureClassifier(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                if masked_message == first_message:
                    raise ProviderError(ErrorKind.SERVER, retryable=False)
                return await super().classify(masked_message=masked_message, prompt=prompt)

        run = await evaluate(
            run_id="one-failure",
            prompt_version="v1",
            settings=Settings(model="test-model", max_attempts=1),
            classifier=OneFailureClassifier(valid_model_output),
            runs_dir=tmp_path,
        )
        assert run["outcomes"][0] == {
            "id": "MSG-01",
            "state": "failed",
            "failure_category": "server_error",
        }
        assert all(item["state"] == "succeeded" for item in run["outcomes"][1:])
        assert "ProviderError" not in (tmp_path / "one-failure/run.json").read_text(
            encoding="utf-8"
        )
        report = json.loads((tmp_path / "one-failure/report.json").read_text(encoding="utf-8"))
        assert report["scores"]["failure_category_counts"] == {"server_error": 1}
        assert report["scores"]["overall_exact_fields"]["total"] == 60

    asyncio.run(scenario())


def test_resume_runs_only_pending_and_preserves_failed_and_unknown(tmp_path, valid_model_output):
    async def scenario():
        class AuthenticationFailure(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                raise ProviderError(ErrorKind.AUTHENTICATION, retryable=False)

        settings = Settings(model="test-model", concurrency_limit=1, max_attempts=1)
        first = await evaluate(
            run_id="resume",
            prompt_version="v1",
            settings=settings,
            classifier=AuthenticationFailure(valid_model_output),
            runs_dir=tmp_path,
        )
        assert first["outcomes"][0]["state"] == "failed"
        assert all(item["state"] == "pending" for item in first["outcomes"][1:])
        first["outcomes"][1]["state"] = "running"
        atomic_json(tmp_path / "resume/run.json", first)

        fake = RecordingFakeClassifier(valid_model_output)
        resumed = await evaluate(
            run_id="resume",
            prompt_version="v1",
            settings=settings,
            classifier=fake,
            resume=True,
            runs_dir=tmp_path,
        )
        assert resumed["outcomes"][0]["state"] == "failed"
        assert resumed["outcomes"][1] == {"id": "MSG-02", "state": "interrupted_unknown"}
        assert all(item["state"] == "succeeded" for item in resumed["outcomes"][2:])
        assert len(fake.calls) == 10
        with pytest.raises(ValueError, match="provenance"):
            await evaluate(
                run_id="resume",
                prompt_version="v1",
                settings=Settings(model="changed"),
                classifier=fake,
                resume=True,
                runs_dir=tmp_path,
            )
        assert len(fake.calls) == 10

    asyncio.run(scenario())


def test_schema_retry_and_attempt_usage_are_preserved(tmp_path, valid_model_output):
    async def scenario():
        class InvalidThenValid(RecordingFakeClassifier):
            calls_seen = 0

            async def classify(self, *, masked_message, prompt):
                self.calls_seen += 1
                content = (
                    {**valid_model_output, "categoria": "invalid"}
                    if self.calls_seen == 1
                    else valid_model_output
                )
                return ProviderResult(content, self.model, 2.0, TokenUsage(10, 2))

        fake = InvalidThenValid(valid_model_output)
        settings = Settings(
            model="test-model",
            concurrency_limit=1,
            max_attempts=2,
            backoff_base_seconds=0,
            backoff_max_seconds=0,
            input_price_per_million=Decimal("1"),
            output_price_per_million=Decimal("2"),
        )
        run = await evaluate(
            run_id="retry",
            prompt_version="v1",
            settings=settings,
            classifier=fake,
            runs_dir=tmp_path,
        )
        assert fake.calls_seen == 13
        assert run["outcomes"][0]["result"]["metadata"]["attempt"] == 2
        assert run["outcomes"][0]["result"]["metadata"]["input_tokens"] == 10
        events = [
            json.loads(line)
            for line in (tmp_path / "retry/model_calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert events[0]["error_type"] == "invalid_classification"
        assert events[0]["success"] is False
        report = json.loads((tmp_path / "retry/report.json").read_text(encoding="utf-8"))
        assert report["usage"]["observed_model_calls"] == 13
        assert report["usage"]["input_tokens_known_subtotal"] == 130
        assert report["usage"]["usage_complete"] is True
        assert report["usage"]["cost_complete"] is True
        exported = export_results(run, tmp_path / "retry/resultados.json")
        assert len(exported) == 12
        assert "estimated_cost_usd" not in exported[0]
        no_repeat = RecordingFakeClassifier(valid_model_output)
        await evaluate(
            run_id="retry",
            prompt_version="v1",
            settings=settings,
            classifier=no_repeat,
            resume=True,
            runs_dir=tmp_path,
        )
        assert no_repeat.calls == []

    asyncio.run(scenario())


def test_attempt_accounting_detects_missing_duplicate_and_foreign_events(tmp_path):
    event_path = tmp_path / "model_calls.jsonl"
    run = {
        "prompt_version": "v1",
        "outcomes": [
            {
                "id": "MSG-01",
                "state": "succeeded",
                "result": {
                    "metadata": {
                        "attempt": 2,
                        "model": "test-model",
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cached_input_tokens": None,
                        "estimated_cost_usd": "0.01",
                    }
                },
            }
        ],
    }

    def event(attempt, *, message_id="MSG-01", success=False):
        return {
            "event": "model_call",
            "message_id": message_id,
            "attempt": attempt,
            "model": "test-model",
            "prompt_version": "v1",
            "success": success,
            "input_tokens": 10,
            "output_tokens": 2,
            "estimated_cost_usd": 0.01,
        }

    first, second = event(1), event(2, success=True)

    def report(events):
        event_path.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")
        return usage_summary(event_path, run)

    complete = report([first, second])
    assert complete["observed_model_calls"] == 2
    assert complete["usage_complete"] is True
    assert complete["cost_complete"] is True
    assert complete["accounting_incomplete_reasons"] == []

    missing = report([second])
    assert missing["usage_complete"] is False
    assert missing["cost_complete"] is False
    assert missing["input_tokens_known_subtotal"] == 10
    assert missing["accounting_incomplete_reasons"] == [
        {"message_id": "MSG-01", "reason": "incomplete_attempt_sequence"}
    ]

    duplicate = report([first, first, second])
    assert duplicate["usage_complete"] is False
    assert duplicate["cost_complete"] is False
    assert duplicate["input_tokens_known_subtotal"] == 20
    assert {"message_id": "MSG-01", "reason": "duplicate_attempt"} in duplicate[
        "accounting_incomplete_reasons"
    ]

    foreign = report([first, second, event(1, message_id="MSG-99")])
    assert foreign["usage_complete"] is False
    assert foreign["cost_complete"] is False
    assert foreign["input_tokens_known_subtotal"] == 20
    assert {"reason": "foreign_message_event"} in foreign["accounting_incomplete_reasons"]

    run["outcomes"][0]["result"]["metadata"]["attempt"] = 1
    inconsistent = report([first, second])
    assert inconsistent["usage_complete"] is False
    assert inconsistent["cost_complete"] is False
    assert {"message_id": "MSG-01", "reason": "incomplete_attempt_sequence"} in inconsistent[
        "accounting_incomplete_reasons"
    ]

    run["outcomes"][0]["result"]["metadata"]["attempt"] = 2
    run["outcomes"][0]["result"]["metadata"]["input_tokens"] = 11
    mismatched_metadata = report([first, second])
    assert mismatched_metadata["usage_complete"] is False
    assert mismatched_metadata["cost_complete"] is False
    assert {"message_id": "MSG-01", "reason": "inconsistent_success_metadata"} in (
        mismatched_metadata["accounting_incomplete_reasons"]
    )


def test_truncated_last_attempt_preserves_preceding_known_usage(tmp_path):
    event_path = tmp_path / "model_calls.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "event": "model_call",
                "message_id": "MSG-01",
                "attempt": 1,
                "model": "test-model",
                "prompt_version": "v1",
                "success": False,
                "input_tokens": 10,
                "output_tokens": 2,
                "estimated_cost_usd": 0.01,
            }
        )
        + '\n{"event": "model_call", "message_id": "MSG-01",',
        encoding="utf-8",
    )
    run = {
        "prompt_version": "v1",
        "outcomes": [
            {
                "id": "MSG-01",
                "state": "succeeded",
                "result": {
                    "metadata": {
                        "attempt": 2,
                        "model": "test-model",
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cached_input_tokens": None,
                        "estimated_cost_usd": "0.01",
                    }
                },
            }
        ],
    }
    report = usage_summary(event_path, run)
    assert report["observed_model_calls"] == 1
    assert report["input_tokens_known_subtotal"] == 10
    assert report["usage_complete"] is False
    assert report["cost_complete"] is False
    assert {"reason": "truncated_final_event"} in report["accounting_incomplete_reasons"]


@pytest.mark.parametrize("close_fails", [False, True])
def test_checkpoint_failure_drains_active_worker_before_cleanup(
    tmp_path, monkeypatch, valid_model_output, close_fails
):
    async def scenario():
        original_checkpoint = atomic_json
        run_dir = tmp_path / "checkpoint-failure"
        checkpoint_error = OSError("synthetic checkpoint failure")
        observations = {"failed": False, "active": 0, "cancelled": False, "closed": False}

        class GatedClassifier(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                observations["active"] += 1
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    observations["cancelled"] = True
                    assert (run_dir / ".run.lock").exists()
                    raise
                finally:
                    observations["active"] -= 1

            async def aclose(self):
                observations["closed"] = True
                assert observations["active"] == 0
                assert observations["cancelled"] is True
                assert (run_dir / ".run.lock").exists()
                if close_fails:
                    raise RuntimeError("synthetic cleanup failure")

        def checkpoint(path, value):
            if (
                path.name == "run.json"
                and value["outcomes"][1]["state"] == "running"
                and observations["active"] == 1
                and not observations["failed"]
            ):
                observations["failed"] = True
                raise checkpoint_error
            original_checkpoint(path, value)

        monkeypatch.setattr("evaluation.runner.atomic_json", checkpoint)
        with pytest.raises(OSError) as caught:
            await asyncio.wait_for(
                evaluate(
                    run_id="checkpoint-failure",
                    prompt_version="v1",
                    settings=Settings(model="test-model", concurrency_limit=2, max_attempts=1),
                    classifier=GatedClassifier(valid_model_output),
                    runs_dir=tmp_path,
                ),
                timeout=3,
            )
        assert caught.value is checkpoint_error
        assert observations == {"failed": True, "active": 0, "cancelled": True, "closed": True}
        assert not (run_dir / ".run.lock").exists()

    asyncio.run(scenario())


def test_cancelling_evaluation_drains_all_active_workers(tmp_path, valid_model_output):
    async def scenario():
        all_entered = asyncio.Event()
        run_dir = tmp_path / "cancelled-run"
        state = {"active": 0, "cancelled": 0, "closed": False}

        class GatedClassifier(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                state["active"] += 1
                if state["active"] == 3:
                    all_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    state["cancelled"] += 1
                    assert (run_dir / ".run.lock").exists()
                    raise
                finally:
                    state["active"] -= 1

            async def aclose(self):
                state["closed"] = True
                assert state["active"] == 0
                assert state["cancelled"] == 3
                assert (run_dir / ".run.lock").exists()

        task = asyncio.create_task(
            evaluate(
                run_id="cancelled-run",
                prompt_version="v1",
                settings=Settings(model="test-model", concurrency_limit=3, max_attempts=1),
                classifier=GatedClassifier(valid_model_output),
                runs_dir=tmp_path,
            )
        )
        await asyncio.wait_for(all_entered.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state == {"active": 0, "cancelled": 3, "closed": True}
        assert not (run_dir / ".run.lock").exists()
        run = load_run(run_dir)
        assert sum(item["state"] == "interrupted_unknown" for item in run["outcomes"]) == 3

    asyncio.run(scenario())


def test_execution_failure_never_serializes_exception_text(tmp_path, valid_model_output):
    async def scenario():
        class SensitiveFailure(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                raise RuntimeError("private diagnosis ana@example.com")

        run = await evaluate(
            run_id="sanitized",
            prompt_version="v1",
            settings=Settings(model="test-model", concurrency_limit=1, max_attempts=1),
            classifier=SensitiveFailure(valid_model_output),
            runs_dir=tmp_path,
        )
        assert all(item["failure_category"] == "execution_error" for item in run["outcomes"])
        for name in ("run.json", "report.json", "model_calls.jsonl"):
            artifact = (tmp_path / "sanitized" / name).read_text(encoding="utf-8")
            assert "ana@example.com" not in artifact
            assert "private diagnosis" not in artifact

    asyncio.run(scenario())
