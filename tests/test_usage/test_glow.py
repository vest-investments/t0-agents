"""``usage_ledger`` through glow — the ledger receives what the gateway minted."""

from __future__ import annotations

from typing import cast

from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.glow import App
from t0_agents.testing import MockGateway
from t0_agents.usage import RunUsageRecord


class _Ledger:
    def __init__(self) -> None:
        self.runs: list[RunUsageRecord] = []

    async def record(self, run: RunUsageRecord) -> None:
        self.runs.append(run)


async def test_a_conversation_turn_reaches_the_ledger_with_the_gateway_ids() -> None:
    ledger = _Ledger()
    gateway = MockGateway.script([[Text(type="text", text="hola")]])
    app = App(t0=cast(AsyncTensorZeroGateway, gateway))

    @app.agent("chat", usage_ledger=ledger)
    async def chat(q: str) -> str:
        """A ledgered agent."""

    conv = chat.conversation(tags={"k": "v"})
    assert await conv.send("hey") == "hola"

    assert len(ledger.runs) == 1
    record = ledger.runs[0]
    assert [i.inference_id for i in record.inferences] == [
        str(gateway.responses[0].inference_id)
    ]
    assert record.episode_id == str(gateway.responses[0].episode_id) == conv.id
    assert record.tags == {"t0_agents.agent": "chat", "k": "v"}
    assert record.outcome == "completed"
    assert record.inferences[0].variant_name == "mock_variant"


async def test_the_app_level_ledger_serves_every_agent() -> None:
    ledger = _Ledger()
    gateway = MockGateway.script([[Text(type="text", text="hola")]])
    app = App(t0=cast(AsyncTensorZeroGateway, gateway), usage_ledger=ledger)

    @app.agent("chat")
    async def chat(q: str) -> str:
        """An agent that inherits the App ledger."""

    assert await chat("hey") == "hola"
    assert [r.function_name for r in ledger.runs] == ["agent"]
