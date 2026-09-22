import asyncio

import httpx
import pytest

import app.llm as llm
import app.main as main
from app.config import settings
from app.models import OpenRouterModelsResponse, SourceAgentSpec, SourceResult


@pytest.fixture(autouse=True)
def reset_llm_state(monkeypatch):
    """Isolate module-level client/cache globals between tests."""
    monkeypatch.setattr(llm, "_shared_http_client", None)
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "_openai_client_key", None)
    monkeypatch.setattr(llm, "_models_cache", None)
    monkeypatch.setattr(llm, "MARKDOWN_MODEL_RETRY_BACKOFF_SECONDS", 0)
    yield
    if llm._shared_http_client is not None:
        asyncio.run(llm._shared_http_client.aclose())
        llm._shared_http_client = None


def _catalog(*, name: str = "openai/a") -> OpenRouterModelsResponse:
    return OpenRouterModelsResponse.model_validate(
        {"data": [{"id": name, "name": name}]}
    )


class TestSharedClient:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm.build_client()

    def test_reuses_instance_until_settings_change(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "k1")
        first = llm.build_client()
        assert llm.build_client() is first

        monkeypatch.setattr(settings, "openai_api_key", "k2")
        second = llm.build_client()
        assert second is not first


class TestFetchModelsCache:
    def _client(self, calls: list[httpx.Request], status: int = 200) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if status != 200:
                return httpx.Response(status)
            return httpx.Response(200, json={"data": [{"id": "openai/a", "name": "a"}]})

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_second_call_served_from_cache(self, monkeypatch):
        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls))

        async def run():
            first = await llm.fetch_openrouter_models()
            second = await llm.fetch_openrouter_models()
            return first, second

        first, second = asyncio.run(run())
        assert len(calls) == 1
        assert second is first

    def test_refresh_failure_serves_stale_cache(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_base_url", "https://openrouter.test/v1")
        stale = _catalog(name="openai/stale")
        monkeypatch.setattr(
            llm,
            "_models_cache",
            (llm.time.monotonic() - llm._MODELS_CACHE_TTL_SECONDS - 1, stale),
        )

        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls, status=500))

        result = asyncio.run(llm.fetch_openrouter_models())
        assert len(calls) == 1
        assert result is stale

    def test_error_without_cache_propagates(self, monkeypatch):
        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls, status=500))

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(llm.fetch_openrouter_models())


class TestTransientRetry:
    def test_retries_retryable_error_then_succeeds(self):
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise llm.CompletionFailure("no choices")
            return "done"

        result = asyncio.run(
            llm._with_transient_retry(op, context="t", model="m", max_retries=1)
        )
        assert result == "done"
        assert calls == 2

    def test_non_retryable_error_propagates_immediately(self):
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError):
            asyncio.run(
                llm._with_transient_retry(op, context="t", model="m", max_retries=3)
            )
        assert calls == 1


class TestRunSingleModel:
    def test_retryable_exception_retried(self, monkeypatch):
        attempts = 0

        async def fake_completion(**kwargs) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise llm.CompletionFailure("no choices")
            return "final answer"

        monkeypatch.setattr(llm, "_run_chat_completion_with_tool_loop", fake_completion)

        result = asyncio.run(
            llm.run_single_model(
                client=None,
                model="openai/a",
                prompt="p",
                temperature=0.2,
                web_search_enabled=False,
                reasoning_effort="medium",
                reasoning_exclude=False,
                attachments=[],
            )
        )
        assert result.status == "ok"
        assert result.content == "final answer"
        assert attempts == 2

    def test_non_retryable_exception_errors_without_retry(self, monkeypatch):
        attempts = 0

        async def fake_completion(**kwargs) -> str:
            nonlocal attempts
            attempts += 1
            raise ValueError("permanent")

        monkeypatch.setattr(llm, "_run_chat_completion_with_tool_loop", fake_completion)

        result = asyncio.run(
            llm.run_single_model(
                client=None,
                model="openai/a",
                prompt="p",
                temperature=0.2,
                web_search_enabled=False,
                reasoning_effort="medium",
                reasoning_exclude=False,
                attachments=[],
            )
        )
        assert result.status == "error"
        assert "permanent" in result.error
        assert attempts == 1


class TestRunSourceModels:
    def test_pending_tasks_cancelled_when_consumer_closes(self, monkeypatch):
        cancelled = asyncio.Event()

        async def fake_run_single_model(*, client, model, **kwargs) -> SourceResult:
            if model == "openai/fast":
                return SourceResult(
                    model=model, agent_id=model, content="x", status="ok"
                )
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return SourceResult(model=model, agent_id=model, content="y", status="ok")

        monkeypatch.setattr(llm, "run_single_model", fake_run_single_model)

        async def run() -> None:
            agents = [
                SourceAgentSpec(id="openai/fast", model="openai/fast"),
                SourceAgentSpec(id="openai/slow", model="openai/slow"),
            ]
            stream = llm.run_source_models(
                client=None,
                agents=agents,
                prompt="p",
                temperature=0.2,
                web_search_enabled=False,
                reasoning_effort="medium",
                reasoning_exclude=False,
                attachments=[],
            )
            first = await stream.__anext__()
            assert first.model == "openai/fast"
            await stream.aclose()

        asyncio.run(run())
        assert cancelled.is_set()


class TestDebateJobConcurrency:
    def test_jobs_bounded_by_parallel_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "openchat_max_parallel_sources", 2)
        active = 0
        max_active = 0

        async def fake_run_model(**kwargs) -> str:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.01)
            finally:
                active -= 1
            return "review"

        jobs = [
            main.DebateJob(
                pair_index=i,
                reviewer_id=f"r{i}",
                reviewer_model=f"rm{i}",
                target_id="t",
                target_model="tm",
                prompt="p",
            )
            for i in range(5)
        ]

        async def run() -> int:
            count = 0
            async for outcome in main.run_debate_jobs(
                jobs=jobs,
                client=None,
                debate_system_prompt="s",
                temperature=0.2,
                reasoning_effort="medium",
                reasoning_exclude=False,
                run_model=fake_run_model,
            ):
                count += 1
            return count

        assert asyncio.run(run()) == 5
        assert max_active == 2


class TestStreamWithHeartbeat:
    def test_events_pass_through_and_heartbeat_fills_gap(self, monkeypatch):
        monkeypatch.setattr(main, "SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)

        async def events():
            yield "event: a\n\n"
            await asyncio.sleep(0.15)
            yield "event: b\n\n"

        async def run() -> list[str]:
            return [item async for item in main._stream_with_heartbeat(events())]

        items = asyncio.run(run())
        assert items[0] == "event: a\n\n"
        assert items[-1] == "event: b\n\n"
        assert items[1:-1] == [": heartbeat\n\n"] * len(items[1:-1])
        assert len(items[1:-1]) >= 1

    def test_inner_exception_propagates(self, monkeypatch):
        async def events():
            yield "event: a\n\n"
            raise RuntimeError("stream broke")

        async def run() -> list[str]:
            return [item async for item in main._stream_with_heartbeat(events())]

        with pytest.raises(RuntimeError, match="stream broke"):
            asyncio.run(run())
