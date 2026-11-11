"""VENDORED CODE — not written for this demo.

Parsing, element normalisation and chunking are lifted wholesale from the production
ingestion service in the sibling ``monorepo`` checkout (`apps/ai`), with that system's
settings and dependency-injection framework stripped out. Each module names its origin
and marks every deviation with ``LOCAL CHANGE``.

Why copy rather than rewrite: this path is solved, boring, and not what the demo is
about. Reusing it means chunk boundaries here match production, so demo behaviour
predicts real behaviour. Rewriting it — or "improving" it — silently decouples the two.
"""
