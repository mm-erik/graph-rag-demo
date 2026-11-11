"""The chat-tier model, adapted to the narrow protocol the pipeline asks for.

Constructed at the composition root and injected. Everything above this file sees only
`ChatModel.complete`, so no caller depends on the Mistral SDK's request shape.
"""

from __future__ import annotations

from mistralai.client import Mistral

from graph_rag.config import MistralSettings


class MistralChatModel:
    """`ChatModel` backed by the Mistral chat completions endpoint."""

    def __init__(self, client: Mistral, model: str) -> None:
        self._client = client
        self._model = model

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str:
        """Complete a prompt, optionally constrained to a JSON object.

        The constraint is per call rather than per model: level inference parses its
        reply as JSON, while a document summary is prose and would come back wrapped in
        a stray object if the whole model were pinned to JSON mode.
        """
        response = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Both callers are re-run on every ingest of the same corpus; a wandering
            # answer would relabel the hierarchy, or rewrite every summary, between two
            # runs over identical input.
            temperature=0.0,
            **({"response_format": {"type": "json_object"}} if json_object else {}),
        )
        choices = response.choices or []
        if not choices:
            return ""
        return choices[0].message.content or ""


def make_chat_model(settings: MistralSettings) -> MistralChatModel:
    """Build the chat-tier model named by configuration."""
    return MistralChatModel(Mistral(api_key=settings.api_key), settings.chat_model)


def make_extraction_model(settings: MistralSettings) -> MistralChatModel:
    """Build the extraction-tier model named by configuration.

    The same adapter as the chat tier, pointed at the cheaper model: extraction is one
    call per chunk and is the only high-volume call site in the pipeline, so the tier
    is the whole difference between the two.
    """
    return MistralChatModel(Mistral(api_key=settings.api_key), settings.extraction_model)
