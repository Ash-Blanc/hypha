"""Scholarly data sources for Hypha."""

from hypha.sources.base import ScholarSource
from hypha.sources.fixture import FixtureSource
from hypha.sources.openalex import OpenAlexSource

__all__ = ["ScholarSource", "OpenAlexSource", "FixtureSource"]
