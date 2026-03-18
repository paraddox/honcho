from __future__ import annotations

import pytest

import src.main as main_module


class _DummyEngine:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def dispose(self) -> None:
        self._calls.append("dispose_engine")


@pytest.mark.asyncio
async def test_lifespan_validates_vector_dimensions_before_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_initialize_telemetry_async() -> None:
        calls.append("init_telemetry")

    async def fake_validate_configured_vector_dimensions() -> None:
        calls.append("validate_dimensions")

    async def fake_init_cache() -> None:
        calls.append("init_cache")

    async def fake_close_cache() -> None:
        calls.append("close_cache")

    async def fake_shutdown_telemetry() -> None:
        calls.append("shutdown_telemetry")

    async def fake_close_external_vector_store() -> None:
        calls.append("close_vector_store")

    monkeypatch.setattr(
        main_module, "initialize_telemetry_async", fake_initialize_telemetry_async
    )
    monkeypatch.setattr(
        main_module,
        "validate_configured_vector_dimensions",
        fake_validate_configured_vector_dimensions,
    )
    monkeypatch.setattr(main_module, "init_cache", fake_init_cache)
    monkeypatch.setattr(main_module, "close_cache", fake_close_cache)
    monkeypatch.setattr(main_module, "shutdown_telemetry", fake_shutdown_telemetry)
    monkeypatch.setattr(main_module, "engine", _DummyEngine(calls))

    import src.vector_store as vector_store_module

    monkeypatch.setattr(
        vector_store_module,
        "close_external_vector_store",
        fake_close_external_vector_store,
    )

    async with main_module.lifespan(main_module.app):
        assert calls == [
            "init_telemetry",
            "validate_dimensions",
            "init_cache",
        ]

    assert calls == [
        "init_telemetry",
        "validate_dimensions",
        "init_cache",
        "close_vector_store",
        "close_cache",
        "dispose_engine",
        "shutdown_telemetry",
    ]


@pytest.mark.asyncio
async def test_lifespan_cleans_up_when_dimension_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_initialize_telemetry_async() -> None:
        calls.append("init_telemetry")

    async def fake_validate_configured_vector_dimensions() -> None:
        calls.append("validate_dimensions")
        raise RuntimeError("dimension drift")

    async def fake_init_cache() -> None:
        calls.append("init_cache")

    async def fake_close_cache() -> None:
        calls.append("close_cache")

    async def fake_shutdown_telemetry() -> None:
        calls.append("shutdown_telemetry")

    async def fake_close_external_vector_store() -> None:
        calls.append("close_vector_store")

    monkeypatch.setattr(
        main_module, "initialize_telemetry_async", fake_initialize_telemetry_async
    )
    monkeypatch.setattr(
        main_module,
        "validate_configured_vector_dimensions",
        fake_validate_configured_vector_dimensions,
    )
    monkeypatch.setattr(main_module, "init_cache", fake_init_cache)
    monkeypatch.setattr(main_module, "close_cache", fake_close_cache)
    monkeypatch.setattr(main_module, "shutdown_telemetry", fake_shutdown_telemetry)
    monkeypatch.setattr(main_module, "engine", _DummyEngine(calls))

    import src.vector_store as vector_store_module

    monkeypatch.setattr(
        vector_store_module,
        "close_external_vector_store",
        fake_close_external_vector_store,
    )

    with pytest.raises(RuntimeError, match="dimension drift"):
        async with main_module.lifespan(main_module.app):
            pass

    assert calls == [
        "init_telemetry",
        "validate_dimensions",
        "close_vector_store",
        "close_cache",
        "dispose_engine",
        "shutdown_telemetry",
    ]
