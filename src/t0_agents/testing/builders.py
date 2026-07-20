"""Ergonomic builders for the content blocks a test scripts into a gateway.

The TensorZero SDK's :class:`~tensorzero.ToolCall` content block carries five
mutually-dependent fields — ``id``, ``raw_name``, ``name``, ``raw_arguments``
(a JSON *string*) and ``arguments`` (the parsed dict) — so a hand-written
literal repeats the tool name twice and keeps a JSON string in sync with a
dict by hand.  :func:`tool_call` derives the raw fields from the tool name and
keyword arguments, turning a five-field literal into ``tool_call("db_lookup",
q="x")``.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from tensorzero import ToolCall

__all__ = ["tool_call"]


def tool_call(
    name: str | None = None,
    /,
    *,
    id: str | None = None,
    server: str | None = None,
    tool: str | None = None,
    **arguments: Any,
) -> ToolCall:
    """Build a :class:`~tensorzero.ToolCall` for a scripted gateway turn.

    Pass the final (possibly server-prefixed) tool name positionally::

        tool_call("db_lookup", q="x")

    or compose it from a mounted server's namespace and bare tool name — the
    ``{server}_{tool}`` form the glow runtime mints for a prefixed mount::

        tool_call(server="db", tool="lookup", q="x")   # name -> "db_lookup"

    Exactly one of ``name`` or ``tool`` is required.  Keyword arguments become
    the tool arguments; ``raw_arguments`` is their JSON encoding and
    ``arguments`` is the re-parse of that string — the invariant every real
    ``ToolCall`` holds (``arguments == json.loads(raw_arguments)``), so a tuple
    surfaces as the list JSON delivers.  ``raw_name``/``name`` are both set to
    the resolved name (TensorZero populates both on a real call).  ``id``
    defaults to a unique value so parallel calls in one turn stay distinct;
    pass ``id=`` to pin it.

    Arguments must be JSON-native, exactly as a model emits them; a non-JSON
    value (a UUID, a Pydantic model) raises ``TypeError`` here rather than
    producing a call the wire could never carry.

    ``id``, ``server`` and ``tool`` are reserved keywords — a tool that takes
    an argument literally so named must use the ``tensorzero.ToolCall``
    constructor directly.

    :raises ValueError: if neither or both of ``name``/``tool`` are given, or
        ``server`` is given without ``tool``.
    """
    if name is not None:
        if tool is not None or server is not None:
            raise ValueError(
                "tool_call takes `name`, or `server`/`tool`, not both"
            )
        final_name = name
    elif tool is not None:
        final_name = f"{server}_{tool}" if server is not None else tool
    else:
        raise ValueError("tool_call requires exactly one of `name` or `tool`")

    raw_arguments = json.dumps(arguments)
    return ToolCall(
        id=id if id is not None else f"call-{uuid4().hex[:8]}",
        raw_name=final_name,
        name=final_name,
        raw_arguments=raw_arguments,
        arguments=json.loads(raw_arguments),
    )
