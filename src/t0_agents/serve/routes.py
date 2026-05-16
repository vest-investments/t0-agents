"""FastAPI routes for ``/chat``, ``/feedback``, ``/health``, ``/info``.

Routes are built from a :class:`SessionFactory` and a shared
:class:`AsyncTensorZeroGateway`.  The only cross-cutting responsibility
is exception mapping — framework-level failures become meaningful HTTP
status codes so callers do not have to grep response bodies.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import TensorZeroError

from t0_agents.core.exceptions import (
    ConfigurationError,
    GuardrailBlockedError,
    MaxTurnsExceededError,
    T0AgentsError,
)
from t0_agents.serve.protocol import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    InfoResponse,
)
from t0_agents.serve.session import SessionFactory


def build_router(
    *,
    session_factory: SessionFactory,
    t0: AsyncTensorZeroGateway,
) -> APIRouter:
    """Assemble the non-streaming routes against a :class:`SessionFactory`.

    Exception → HTTP mapping (applied to ``POST /chat`` only — other
    routes raise on truly exceptional errors, which FastAPI renders as
    500):

    * :class:`MaxTurnsExceededError` → 408 Request Timeout.
    * :class:`GuardrailBlockedError` → 409 Conflict.
    * :class:`ConfigurationError` → 400 Bad Request.
    * :class:`TensorZeroError` → 502 Bad Gateway.
    * Any other :class:`T0AgentsError` → 400 Bad Request.

    ``POST /feedback`` additionally passes a 4xx gateway response through
    verbatim (the ``/feedback`` payload maps 1:1 to caller input, so an
    "unknown metric" 404 is honestly a caller error); 5xx and missing
    status codes still map to 502 Bad Gateway.
    """
    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        try:
            async with session_factory.session(extra_tags=req.tags) as agent:
                result = await agent.invoke(
                    req.message, episode_id=req.episode_id,
                ).result()
        except MaxTurnsExceededError as e:
            raise HTTPException(status_code=408, detail=str(e)) from e
        except GuardrailBlockedError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ConfigurationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except TensorZeroError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except T0AgentsError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return ChatResponse(
            response=result.response,
            episode_id=result.episode_id,
            inference_ids=list(result.inference_ids),
            turns=result.turns,
            response_source=result.response_source,
            guardrail_log=[asdict(entry) for entry in result.guardrail_log],
        )

    @router.post("/feedback", response_model=FeedbackResponse)
    async def feedback(req: FeedbackRequest) -> FeedbackResponse:
        kwargs: dict[str, object] = {
            "metric_name": req.metric_name,
            "value": req.value,
        }
        if req.inference_id is not None:
            kwargs["inference_id"] = req.inference_id
        if req.episode_id is not None:
            kwargs["episode_id"] = req.episode_id
        if req.tags is not None:
            kwargs["tags"] = req.tags
        if req.dryrun is not None:
            kwargs["dryrun"] = req.dryrun

        try:
            response = await t0.feedback(**kwargs)  # type: ignore[arg-type]
        except TensorZeroError as e:
            # Forward 4xx from the gateway — unlike /chat, the /feedback
            # payload maps 1:1 to caller input (metric_name, inference_id,
            # value), so "unknown metric" or "inference_id not found" is
            # a caller error, not an upstream failure.  5xx and missing
            # status codes still map to 502 Bad Gateway.
            sc = getattr(e, "status_code", None)
            if isinstance(sc, int) and 400 <= sc < 500:
                raise HTTPException(status_code=sc, detail=str(e)) from e
            raise HTTPException(status_code=502, detail=str(e)) from e

        return FeedbackResponse(feedback_id=str(response.feedback_id))

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            gateway="ok",
            tool_provider=session_factory.probe_tool_provider(),
        )

    @router.get("/info", response_model=InfoResponse)
    async def info() -> InfoResponse:
        cfg = session_factory.base_config
        return InfoResponse(
            name=cfg.name,
            function_name=cfg.function_name,
            namespace=cfg.namespace,
            tags=dict(cfg.tags) if cfg.tags else None,
            has_memory=cfg.memory is not None,
            guardrails=[g.name for g in cfg.guardrails],
        )

    return router
