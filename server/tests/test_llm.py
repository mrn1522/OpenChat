import asyncio
import gzip

import httpx
import pytest

import app.llm as llm
import app.main as main
from app.config import settings
from app.models import (
    AttachmentInput,
    DirectChatMessage,
    OpenRouterModelsResponse,
    SourceAgentSpec,
    SourceResult,
)


@pytest.fixture(autouse=True)
def reset_llm_state(monkeypatch):
    """Isolate module-level client/cache globals between tests."""
    monkeypatch.setattr(llm, "_shared_http_client", None)
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "_openai_client_key", None)
    monkeypatch.setattr(llm, "_models_cache", None)
    monkeypatch.setattr(llm, "_service_tiers_cache", {})
    monkeypatch.setattr(llm, "_service_tiers_inflight", {})
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
            (
                "https://openrouter.test/v1",
                llm.time.monotonic() - llm._MODELS_CACHE_TTL_SECONDS - 1,
                stale,
            ),
        )

        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls, status=500))

        result = asyncio.run(llm.fetch_openrouter_models())
        assert len(calls) == 1
        assert result is stale

    def test_malformed_response_serves_stale_cache(self, monkeypatch):
        """A 200 with unparseable JSON triggers the same stale fallback as a 5xx."""
        base_url = "https://openrouter.test/v1"
        monkeypatch.setattr(settings, "openai_base_url", base_url)
        stale = _catalog(name="openai/stale")
        monkeypatch.setattr(
            llm,
            "_models_cache",
            (base_url, llm.time.monotonic() - llm._MODELS_CACHE_TTL_SECONDS - 1, stale),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"{not json")

        monkeypatch.setattr(
            llm, "_shared_http", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

        assert asyncio.run(llm.fetch_openrouter_models()) is stale

    @pytest.mark.parametrize("content", [b"{}", b'{"data": {}}', b'{"data": []}', b'{"data": [7, null]}'])
    def test_empty_or_shapeless_catalog_serves_stale(self, monkeypatch, content):
        """A 200 without usable models is malformed — not a new (empty) cache."""
        base_url = "https://openrouter.test/v1"
        monkeypatch.setattr(settings, "openai_base_url", base_url)
        stale = _catalog(name="openai/stale")
        monkeypatch.setattr(
            llm,
            "_models_cache",
            (base_url, llm.time.monotonic() - llm._MODELS_CACHE_TTL_SECONDS - 1, stale),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=content)

        monkeypatch.setattr(
            llm, "_shared_http", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

        assert asyncio.run(llm.fetch_openrouter_models()) is stale

    def test_base_url_change_ignores_other_provider_catalog(self, monkeypatch):
        """A cache entry from a different base_url is neither fresh nor stale-valid."""
        monkeypatch.setattr(settings, "openai_base_url", "https://new-provider.test/v1")
        monkeypatch.setattr(
            llm,
            "_models_cache",
            (
                "https://old-provider.test/v1",
                llm.time.monotonic(),  # unexpired for the old provider
                _catalog(name="openai/old"),
            ),
        )

        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls, status=500))

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(llm.fetch_openrouter_models())
        assert len(calls) == 1

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


class TestBuildDirectChatMessages:
    _messages = [
        DirectChatMessage(role="user", content="first"),
        DirectChatMessage(role="assistant", content="reply"),
        DirectChatMessage(role="user", content="describe this"),
    ]

    def _image(self) -> AttachmentInput:
        return AttachmentInput(
            name="shot.png",
            size=10,
            content_type="image/png",
            content="QUJD",
        )

    def test_no_attachments_keeps_string_content(self):
        built = llm._build_direct_chat_messages(
            messages=self._messages,
            attachments=[],
            system_prompt="sys",
        )
        assert built[0] == {"role": "system", "content": "sys"}
        assert built[-1] == {"role": "user", "content": "describe this"}

    def test_text_attachments_merge_into_last_user_message(self):
        built = llm._build_direct_chat_messages(
            messages=self._messages,
            attachments=[
                AttachmentInput(
                    name="notes.txt",
                    size=5,
                    content_type="text/plain",
                    content="hello",
                )
            ],
            system_prompt="sys",
        )
        assert built[1]["content"] == "first"  # earlier turns untouched
        last = built[-1]["content"]
        assert isinstance(last, str)
        assert "describe this" in last
        assert "notes.txt" in last
        assert "hello" in last

    def test_image_attachments_build_multipart_content(self):
        built = llm._build_direct_chat_messages(
            messages=self._messages,
            attachments=[self._image()],
            system_prompt="sys",
        )
        last = built[-1]["content"]
        assert isinstance(last, list)
        assert last[0] == {"type": "text", "text": "describe this"}
        assert last[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,QUJD"},
        }

    def test_mixed_attachments_merge_text_and_images(self):
        built = llm._build_direct_chat_messages(
            messages=self._messages,
            attachments=[
                AttachmentInput(
                    name="notes.txt",
                    size=5,
                    content_type="text/plain",
                    content="ctx",
                ),
                self._image(),
            ],
            system_prompt="sys",
        )
        last = built[-1]["content"]
        assert isinstance(last, list)
        assert last[0]["type"] == "text"
        assert "ctx" in last[0]["text"]
        assert last[1]["type"] == "image_url"

    def test_prompt_with_attachments_skips_images(self):
        prompt = llm._build_prompt_with_attachments("hello", [self._image()])
        assert prompt == "hello"


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


class TestGZipUnlessSSE:
    def _drive(self, app, path: str = "/") -> list[dict]:
        sent: list[dict] = []

        async def send(message):
            sent.append(dict(message))

        middleware = main.GZipUnlessSSEMiddleware(app)
        scope = {
            "type": "http",
            "headers": [(b"accept-encoding", b"gzip")],
            "path": path,
        }
        asyncio.run(middleware(scope, None, send))
        return sent

    def test_event_stream_passes_through_uncompressed(self):
        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"event: a\n\n",
                    "more_body": True,
                }
            )
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        sent = self._drive(app)
        bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
        assert b"event: a\n\n" in bodies
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert b"content-encoding" not in dict(start["headers"])

    def test_json_response_is_compressed(self):
        payload = b'{"data": "' + b"x" * 5000 + b'"}'

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode()),
                    ],
                }
            )
            await send(
                {"type": "http.response.body", "body": payload, "more_body": False}
            )

        sent = self._drive(app)
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert dict(start["headers"])[b"content-encoding"] == b"gzip"
        body = b"".join(m["body"] for m in sent if m["type"] == "http.response.body")
        assert gzip.decompress(body) == payload


class TestExtractServiceTiers:
    def test_collects_tier_suffixes(self):
        payload = {
            "data": {
                "endpoints": [
                    {"tag": "openai/flex"},
                    {"tag": "openai/fast"},  # "fast" normalizes to "priority"
                    {"tag": "google-vertex/global/priority"},
                ]
            }
        }
        assert llm._extract_service_tiers(payload) == ["flex", "priority"]

    def test_ignores_non_tier_suffixes(self):
        payload = {
            "data": {
                "endpoints": [
                    {"tag": "openai/us-east5"},
                    {"tag": "openai/fp8"},
                    {"tag": "openai"},
                    {"tag": 42},
                    {"other": "x"},
                    "not-a-dict",
                ]
            }
        }
        assert llm._extract_service_tiers(payload) == []

    @pytest.mark.parametrize(
        "payload",
        [{}, {"data": {}}, {"data": {"endpoints": "x"}}, [], "x", None],
    )
    def test_malformed_payload_returns_empty(self, payload):
        assert llm._extract_service_tiers(payload) == []


class TestFetchModelServiceTiers:
    def _client(
        self, calls: list[httpx.Request], *, status: int = 200, payload: dict | None = None
    ) -> httpx.AsyncClient:
        if payload is None:
            payload = {"data": {"endpoints": [{"tag": "openai/flex"}]}}

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if status != 200:
                return httpx.Response(status)
            return httpx.Response(200, json=payload)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_fetches_endpoints_url_and_caches(self, monkeypatch):
        monkeypatch.setattr(settings, "openai_base_url", "https://openrouter.test/v1")
        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls))

        async def run():
            first = await llm.fetch_model_service_tiers("openai/a")
            second = await llm.fetch_model_service_tiers("openai/a")
            return first, second

        first, second = asyncio.run(run())
        assert first == ["flex"]
        assert second == ["flex"]
        assert len(calls) == 1
        assert str(calls[0].url) == "https://openrouter.test/v1/models/openai/a/endpoints"

    def test_refresh_failure_serves_stale_cache(self, monkeypatch):
        base_url = "https://openrouter.test/v1"
        monkeypatch.setattr(settings, "openai_base_url", base_url)
        monkeypatch.setattr(
            llm,
            "_service_tiers_cache",
            {
                (base_url, "openai/a"): (
                    llm.time.monotonic() - llm._SERVICE_TIERS_CACHE_TTL_SECONDS - 1,
                    ["flex"],
                )
            },
        )

        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls, status=500))

        assert asyncio.run(llm.fetch_model_service_tiers("openai/a")) == ["flex"]
        assert len(calls) == 1

    def test_malformed_response_serves_stale_cache(self, monkeypatch):
        """A 200 without a usable endpoint list is malformed, not 'no tiers'."""
        base_url = "https://openrouter.test/v1"
        monkeypatch.setattr(settings, "openai_base_url", base_url)
        monkeypatch.setattr(
            llm,
            "_service_tiers_cache",
            {
                (base_url, "openai/a"): (
                    llm.time.monotonic() - llm._SERVICE_TIERS_CACHE_TTL_SECONDS - 1,
                    ["flex"],
                )
            },
        )

        calls: list[httpx.Request] = []
        monkeypatch.setattr(
            llm, "_shared_http", lambda: self._client(calls, payload={"data": {}})
        )

        assert asyncio.run(llm.fetch_model_service_tiers("openai/a")) == ["flex"]
        assert len(calls) == 1

    def test_empty_endpoints_list_caches_as_no_tiers(self, monkeypatch):
        """An explicit empty roster is a valid 'no tiers' result, not malformed."""
        calls: list[httpx.Request] = []
        monkeypatch.setattr(
            llm,
            "_shared_http",
            lambda: self._client(calls, payload={"data": {"endpoints": []}}),
        )

        assert asyncio.run(llm.fetch_model_service_tiers("openai/a")) == []
        assert len(calls) == 1

    def test_error_without_cache_propagates(self, monkeypatch):
        calls: list[httpx.Request] = []
        monkeypatch.setattr(llm, "_shared_http", lambda: self._client(calls, status=500))

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(llm.fetch_model_service_tiers("openai/a"))


class TestServiceTierPlumbing:
    def test_extra_body_includes_named_tiers_only(self):
        base = dict(
            web_search_enabled=False, reasoning_effort="medium", reasoning_exclude=False
        )
        assert llm._build_openrouter_extra_body(**base, service_tier="flex")[
            "service_tier"
        ] == "flex"
        assert llm._build_openrouter_extra_body(**base, service_tier="priority")[
            "service_tier"
        ] == "priority"
        for tier in (None, "default"):
            body = llm._build_openrouter_extra_body(**base, service_tier=tier)
            assert "service_tier" not in body

    def test_run_source_models_passes_per_model_tier(self, monkeypatch):
        captured: dict[str, str | None] = {}

        async def fake_run_single_model(
            *, client, model, service_tier=None, **kwargs
        ) -> SourceResult:
            captured[model] = service_tier
            return SourceResult(model=model, agent_id=model, content="x", status="ok")

        monkeypatch.setattr(llm, "run_single_model", fake_run_single_model)

        async def run() -> None:
            agents = [
                SourceAgentSpec(id="a1", model="openai/a"),
                SourceAgentSpec(id="a2", model="openai/b"),
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
                service_tier_by_model={"openai/a": "flex"},
            )
            async for _ in stream:
                pass

        asyncio.run(run())
        assert captured == {"openai/a": "flex", "openai/b": None}

    def test_run_markdown_model_forwards_tier_to_completion(self, monkeypatch):
        captured = {}

        async def fake_completion(**kwargs) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(llm, "_run_chat_completion_with_tool_loop", fake_completion)

        asyncio.run(
            llm.run_markdown_model(
                client=None,
                model="openai/a",
                prompt="p",
                system_prompt="s",
                temperature=0.2,
                web_search_enabled=False,
                reasoning_effort="medium",
                reasoning_exclude=False,
                context="t",
                service_tier="priority",
            )
        )
        assert captured["extra_body"]["service_tier"] == "priority"

    def test_run_direct_chat_model_forwards_tier_to_completion(self, monkeypatch):
        captured = {}

        async def fake_completion(**kwargs) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(llm, "_run_chat_completion_with_tool_loop", fake_completion)

        asyncio.run(
            llm.run_direct_chat_model(
                client=None,
                model="openai/a",
                messages=[DirectChatMessage(role="user", content="hi")],
                system_prompt="s",
                temperature=0.2,
                web_search_enabled=False,
                reasoning_effort="medium",
                reasoning_exclude=False,
                attachments=[],
                service_tier="flex",
            )
        )
        assert captured["extra_body"]["service_tier"] == "flex"
