from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.embedding_client as embedding_client_module
from src.embedding_client import _EmbeddingClient


@pytest.fixture(autouse=True)
def reset_embedding_client_singletons() -> None:
    embedding_client_module.EmbeddingClient._instance = None
    embedding_client_module.EmbeddingClient._wrapper_instance = None
    yield
    embedding_client_module.EmbeddingClient._instance = None
    embedding_client_module.EmbeddingClient._wrapper_instance = None


def test_custom_provider_prefers_custom_embedding_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embedding_client_module.settings.LLM, "EMBEDDING_PROVIDER", "custom"
    )
    monkeypatch.setattr(
        embedding_client_module.settings.LLM, "OPENAI_API_KEY", "wrong-openai-key"
    )
    monkeypatch.setattr(
        embedding_client_module.settings.LLM,
        "CUSTOM_EMBEDDING_API_KEY",
        "custom-embedding-key",
    )
    monkeypatch.setattr(
        embedding_client_module.settings.LLM,
        "CUSTOM_EMBEDDING_BASE_URL",
        "http://localhost:11434/v1",
    )

    captured: dict[str, str | None] = {}

    def fake_async_openai(*, api_key: str | None, base_url: str | None = None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return SimpleNamespace(embeddings=SimpleNamespace(create=None))

    monkeypatch.setattr(embedding_client_module, "AsyncOpenAI", fake_async_openai)

    client = embedding_client_module.EmbeddingClient()
    client._get_client()

    assert captured == {
        "api_key": "custom-embedding-key",
        "base_url": "http://localhost:11434/v1",
    }


@pytest.mark.asyncio
async def test_gemini_uses_configured_vector_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embedding_client_module.settings.LLM, "EMBEDDING_PROVIDER", "gemini"
    )
    monkeypatch.setattr(embedding_client_module.settings.LLM, "GEMINI_API_KEY", "test")
    monkeypatch.setattr(
        embedding_client_module.settings.VECTOR_STORE, "DIMENSIONS", 256
    )

    calls: list[dict[str, object]] = []

    class FakeGenAIClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.aio = SimpleNamespace(
                models=SimpleNamespace(embed_content=self.embed_content)
            )

        async def embed_content(
            self,
            *,
            model: str,
            contents: str | list[str],
            config: dict[str, int],
        ) -> SimpleNamespace:
            calls.append(
                {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
            )
            item_count = len(contents) if isinstance(contents, list) else 1
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[float(i)] * config["output_dimensionality"])
                    for i in range(item_count)
                ]
            )

    monkeypatch.setattr(embedding_client_module.genai, "Client", FakeGenAIClient)

    client = _EmbeddingClient(provider="gemini", api_key="test")

    single = await client.embed("hello")
    batch = await client.simple_batch_embed(["alpha", "beta"])
    chunked = await client.batch_embed({"msg-1": ("gamma", [1, 2, 3])})

    assert len(single) == 256
    assert [len(item) for item in batch] == [256, 256]
    assert [len(item) for item in chunked["msg-1"]] == [256]
    assert [call["config"] for call in calls] == [
        {"output_dimensionality": 256},
        {"output_dimensionality": 256},
        {"output_dimensionality": 256},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter"])
async def test_openai_compatible_providers_use_configured_vector_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setattr(
        embedding_client_module.settings.VECTOR_STORE, "DIMENSIONS", 256
    )

    calls: list[dict[str, object]] = []

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        dimensions = kwargs.get(
            "dimensions", embedding_client_module.settings.VECTOR_STORE.DIMENSIONS
        )
        inputs = kwargs["input"]
        item_count = len(inputs) if isinstance(inputs, list) else 1
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(i)] * int(dimensions))
                for i in range(item_count)
            ]
        )

    def fake_async_openai(*, api_key: str | None, base_url: str | None = None):
        return SimpleNamespace(
            embeddings=SimpleNamespace(create=fake_create),
            api_key=api_key,
            base_url=base_url,
        )

    monkeypatch.setattr(embedding_client_module, "AsyncOpenAI", fake_async_openai)

    client = _EmbeddingClient(provider=provider, api_key="test")

    single = await client.embed("hello")
    batch = await client.simple_batch_embed(["alpha", "beta"])
    chunked = await client.batch_embed({"msg-1": ("gamma", [1, 2, 3])})

    assert len(single) == 256
    assert [len(item) for item in batch] == [256, 256]
    assert [len(item) for item in chunked["msg-1"]] == [256]
    assert [call["dimensions"] for call in calls] == [256, 256, 256]
