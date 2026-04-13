"""Adapter between FastMCP tool types and TensorZero tool format."""

from __future__ import annotations


def convert_tools() -> None:
    raise NotImplementedError


def serialize_tool_result() -> None:
    raise NotImplementedError


def build_tool_result() -> None:
    raise NotImplementedError
