"""Neo4j driver construction.

The driver is constructed *here only*, at the composition root, and injected into every
function that touches the graph. Nothing below this module may construct a driver or
reach for a module-level global.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase

from graph_rag.config import Neo4jSettings


@contextmanager
def graph_driver(settings: Neo4jSettings) -> Iterator[Driver]:
    driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
    try:
        yield driver
    finally:
        driver.close()
