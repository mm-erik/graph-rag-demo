set dotenv-load := true

default:
    @just --list

# Bring the infrastructure up and initialise the graph schema.
up:
    docker compose up -d --wait
    uv sync
    uv run graph-rag init-graph

# Create constraints and indexes (idempotent).
init:
    uv run graph-rag init-graph

# Report connectivity and index readiness.
status:
    uv run graph-rag status

# Check the configured Mistral model identifiers against the live API.
verify-models:
    uv run graph-rag verify-models

# Parse one PDF and print the chunks it would contribute to the graph.
parse PDF *ARGS:
    uv run graph-rag parse '{{PDF}}' {{ARGS}}

ingest DIR *ARGS:
    uv run graph-rag ingest '{{DIR}}' {{ARGS}}

# Print the stored document summaries, to compare near-identical reports.
summaries *ARGS:
    uv run graph-rag summaries {{ARGS}}

# Retrieve chunks for a question: vector + full text, fused. --mode isolates a leg.
search QUESTION *ARGS:
    uv run graph-rag search '{{QUESTION}}' {{ARGS}}

# Walk the entity graph from the names in a question. --hops is the knob to turn live.
traverse QUESTION *ARGS:
    uv run graph-rag traverse '{{QUESTION}}' {{ARGS}}

# Answer a question from both legs. --legs isolates one; --max-chunks caps the context.
query QUESTION *ARGS:
    uv run graph-rag query '{{QUESTION}}' {{ARGS}}

# Ask follow-up questions. "Und 2023?" is resolved against the conversation first.
chat QUESTION="" *ARGS:
    uv run graph-rag chat '{{QUESTION}}' {{ARGS}}

# Serve the HTTP API — and the built UI at /, when `just ui-build` has run.
serve *ARGS:
    uv run graph-rag serve {{ARGS}}

# Install the frontend's dependencies. Once per checkout.
ui-install:
    cd frontend && npm install

# The UI with hot reload on :5173, proxying /api to `just serve` in another terminal.
ui:
    cd frontend && npm run dev

# Build the UI into frontend/dist, which `just serve` then serves at /.
ui-build:
    cd frontend && npm run build

# What to run on stage: one process serving the API and the built UI together.
demo: ui-build
    uv run graph-rag serve

# The fast suite: pure logic only, no infrastructure. The smoke test is excluded.
test:
    uv run pytest

# The end-to-end smoke test, against real Neo4j, Mistral and parsing service. Minutes
# rather than milliseconds, and it expects a cold graph — see RUNBOOK.md.
smoke *ARGS:
    uv run pytest -m smoke -v {{ARGS}}

# Stop the containers, keeping the graph volumes.
down:
    docker compose down

logs:
    docker compose logs -f neo4j

# Generate the synthetic adversarial German corpus.
corpus DIR="corpus" *ARGS:
    uv run graph-rag generate-corpus '{{DIR}}' {{ARGS}}

# Print the extracted entities; --bridges shows the cross-client nodes.
entities *ARGS:
    uv run graph-rag entities {{ARGS}}

# Re-run entity extraction over the chunks already in the graph.
extract *ARGS:
    uv run graph-rag extract {{ARGS}}
