#!/usr/bin/env python3
"""Trace an episode from ClickHouse.

Given an ``episode_id``, queries the local ClickHouse instance (started by
``docker compose up``) and prints a human-readable conversation trace with
linked feedback.

Usage::

    uv run python scripts/trace_episode.py <episode_id>
    uv run python scripts/trace_episode.py <episode_id> --ch-url http://host:8123

Requires ``docker compose up -d`` (TensorZero gateway + ClickHouse).
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

DEFAULT_CH_URL = "http://localhost:8123"
DEFAULT_CH_USER = "chuser"
DEFAULT_CH_PASSWORD = "chpassword"
DEFAULT_CH_DATABASE = "tensorzero"

# Maps feedback table name to the column that holds the target reference.
# Most tables use ``target_id``; ``DemonstrationFeedback`` uses ``inference_id``.
FEEDBACK_TABLES: dict[str, str] = {
    "BooleanMetricFeedback": "target_id",
    "FloatMetricFeedback": "target_id",
    "CommentFeedback": "target_id",
    "DemonstrationFeedback": "inference_id",
}


def _query_ch(
    client: httpx.Client,
    query: str,
    *,
    ch_url: str,
    ch_user: str,
    ch_password: str,
    ch_database: str,
) -> list[dict[str, object]]:
    """Run a ClickHouse query and return parsed JSON rows."""
    resp = client.get(
        ch_url,
        params={
            "query": query,
            "user": ch_user,
            "password": ch_password,
            "database": ch_database,
        },
    )
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def _truncate(s: str, maxlen: int = 120) -> str:
    """Truncate a string for display."""
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 3] + "..."


def _format_content(content_json: str, label: str) -> str:
    """Parse a JSON content array and format it for display."""
    try:
        blocks = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        return f"  {label}: {_truncate(str(content_json))}"

    if not isinstance(blocks, list):
        blocks = [blocks]

    lines: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            btype = block.get("type", "unknown")
            if btype == "text":
                text = block.get("text", "")
                lines.append(f"  {label} [{btype}]: {_truncate(str(text))}")
            elif btype == "tool_call":
                name = block.get("name") or block.get("raw_name", "?")
                args = block.get("arguments", block.get("raw_arguments", ""))
                lines.append(
                    f"  {label} [ToolCall:{name}]: "
                    f"{_truncate(json.dumps(args) if isinstance(args, dict) else str(args))}"
                )
            elif btype == "tool_result":
                name = block.get("name", "?")
                result = block.get("result", "")
                lines.append(
                    f"  {label} [ToolResult:{name}]: {_truncate(str(result))}"
                )
            elif btype == "thought":
                text = block.get("text", "")
                lines.append(f"  {label} [Thought]: {_truncate(str(text))}")
            else:
                lines.append(
                    f"  {label} [{btype}]: {_truncate(json.dumps(block))}"
                )
        else:
            lines.append(f"  {label}: {_truncate(str(block))}")

    return "\n".join(lines) if lines else f"  {label}: (empty)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace a TensorZero episode from ClickHouse.",
    )
    parser.add_argument("episode_id", help="The episode_id to trace")
    parser.add_argument(
        "--ch-url", default=DEFAULT_CH_URL,
        help=f"ClickHouse HTTP URL (default: {DEFAULT_CH_URL})",
    )
    parser.add_argument(
        "--ch-user", default=DEFAULT_CH_USER,
        help=f"ClickHouse user (default: {DEFAULT_CH_USER})",
    )
    parser.add_argument(
        "--ch-password", default=DEFAULT_CH_PASSWORD,
        help=f"ClickHouse password (default: {DEFAULT_CH_PASSWORD})",
    )
    parser.add_argument(
        "--ch-database", default=DEFAULT_CH_DATABASE,
        help=f"ClickHouse database (default: {DEFAULT_CH_DATABASE})",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Print raw JSON instead of formatted output",
    )
    args = parser.parse_args()

    ch_kwargs = dict(
        ch_url=args.ch_url,
        ch_user=args.ch_user,
        ch_password=args.ch_password,
        ch_database=args.ch_database,
    )

    with httpx.Client(timeout=10.0) as client:
        # --- Inferences ---
        inferences = _query_ch(
            client,
            f"SELECT id, input, output, "
            f"function_name, variant_name, processing_time_ms "
            f"FROM ChatInference "
            f"WHERE episode_id = '{args.episode_id}' "
            f"ORDER BY id ASC "
            f"FORMAT JSONEachRow",
            **ch_kwargs,
        )

        if not inferences:
            print(f"No inferences found for episode_id={args.episode_id}")
            sys.exit(1)

        print(f"Episode: {args.episode_id}")
        print(f"Turns:   {len(inferences)}")
        print("=" * 70)

        inference_ids = []
        for i, row in enumerate(inferences, 1):
            inf_id = row.get("id", "?")
            inference_ids.append(str(inf_id))
            func = row.get("function_name", "?")
            variant = row.get("variant_name", "?")
            proc_ms = row.get("processing_time_ms")

            print(f"\nTurn {i}: inference_id={inf_id}")
            print(f"  function={func}, variant={variant}", end="")
            if proc_ms is not None:
                print(f", processing_time={proc_ms}ms")
            else:
                print()

            if args.raw:
                print(f"  input:  {row.get('input', '')}")
                print(f"  output: {row.get('output', '')}")
            else:
                inp = row.get("input", "")
                out = row.get("output", "")
                if isinstance(inp, str):
                    try:
                        inp_parsed = json.loads(inp)
                        messages = (
                            inp_parsed
                            if isinstance(inp_parsed, list)
                            else inp_parsed.get("messages", [])
                        )
                        for msg in messages:
                            if isinstance(msg, dict):
                                role = msg.get("role", "?")
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    print(f"  input [{role}]: {_truncate(content)}")
                                elif isinstance(content, list):
                                    print(_format_content(json.dumps(content), f"input [{role}]"))
                    except (json.JSONDecodeError, TypeError):
                        print(f"  input: {_truncate(str(inp))}")

                if isinstance(out, str):
                    print(_format_content(out, "output"))
                else:
                    print(f"  output: {_truncate(str(out))}")

        # --- Feedback ---
        print("\n" + "=" * 70)
        print("Feedback:")

        found_feedback = False
        # Search by episode_id and all inference_ids
        target_ids = [args.episode_id, *inference_ids]
        for table, id_col in FEEDBACK_TABLES.items():
            for target_id in target_ids:
                rows = _query_ch(
                    client,
                    f"SELECT * FROM {table} "
                    f"WHERE {id_col} = '{target_id}' "
                    f"FORMAT JSONEachRow",
                    **ch_kwargs,
                )
                for row in rows:
                    found_feedback = True
                    value = row.get("value", "?")
                    metric = row.get("metric_name", table)
                    print(
                        f"  [{table}] metric={metric}, "
                        f"target={target_id[:12]}..., value={value}"
                    )

        if not found_feedback:
            print("  (none)")


if __name__ == "__main__":
    main()
