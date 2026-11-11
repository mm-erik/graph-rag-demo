"""Retry a transient provider failure, honouring any advised ``retry_after``.

Provider-agnostic and caller-agnostic: a rate limit is a rate limit whether it
interrupts an ingestion worker, a parse submission or an offline evaluation
run. It lives in :mod:`shared` rather than in :mod:`workers.base`, where it
started, because importing a submodule executes its package's ``__init__``, and
``workers/__init__.py`` eagerly constructs the whole ingestion and parse stack
— a dependency the evaluation harness has no business acquiring just to wait
two seconds before retrying a judge call.

:mod:`workers.base` re-exports both names, so every existing caller keeps
working unchanged.

---

VENDORED — not demo-authored code.

Origin: ``apps/ai/shared/backoff.py`` in the sibling ``monorepo`` checkout (the production ingestion
service). Copied wholesale, with that system's settings/DI framework stripped; every
deviation from the original is marked ``LOCAL CHANGE``.

Do not improve this file. Its value is that behaviour here matches production, so the
demo predicts the real thing; edits silently decouple the two.
"""
from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from graph_rag.vendored.errors import ProviderError, RateLimitError

#: A job-submission result of arbitrary provider-specific shape.
_ResultT = TypeVar("_ResultT")

#: Async sleep function; overridable in tests for determinism.
Sleep = Callable[[float], Awaitable[Any]]


def parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait per a ``Retry-After`` header, or ``None``.

    Lives here because what reads this header is always about to hand the
    answer to :func:`submit_with_backoff` as a ``retry_after``.

    Delta-seconds only. The HTTP-date form is valid per spec but is not what
    the backends this service talks to emit, and deriving a delay from a
    date needs the two clocks to agree -- so it is skipped rather than
    guessed at.

    Anything else unusable yields ``None`` as well, which hands the caller
    back to its own jittered schedule. That is the safe answer and zero is
    not: :func:`submit_with_backoff` reads any number here as an
    instruction, so a negative value would mean "retry at once" -- straight
    back into the rate limit the backoff exists to stay out of -- and a
    non-finite one, which ``float`` happily returns for ``inf``, would mean
    a sleep that never wakes.
    """
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


async def submit_with_backoff(
    submit: Callable[[], Awaitable[_ResultT]],
    *,
    max_retries: int,
    base_backoff: float,
    max_backoff: float,
    retry_on: type[BaseException] | tuple[type[BaseException], ...] = RateLimitError,
    retry_if: Callable[[BaseException], bool] | None = None,
    sleep: Sleep = asyncio.sleep,
) -> _ResultT:
    """Call ``submit``; on a transient failure, back off and retry.

    When ``submit`` raises one of ``retry_on`` (:class:`RateLimitError` by
    default), waits and retries up to ``max_retries`` times. The wait honours
    the exception's ``retry_after`` when present, otherwise uses capped
    exponential backoff (``base_backoff * 2**attempt``, clamped to
    ``max_backoff``) with full jitter to avoid synchronised retries. Once
    retries are exhausted the last failure is reported as a
    :class:`~shared.errors.ProviderError`. Exceptions not in ``retry_on``
    propagate unchanged.

    ``retry_if`` narrows ``retry_on`` further for callers whose retryability
    is not expressible as a type: a predicate returning ``False`` re-raises
    the exception immediately, exactly as if it had not been in ``retry_on``
    at all. This exists because HTTP statuses do not map one-to-one onto
    exception classes -- a transport that reports every unmapped status as the
    same class leaves the status on the exception, and only the caller knows
    which of those statuses are worth re-sending *its* request for.

    Note that an advised ``retry_after`` is used **verbatim** and is not
    clamped to ``max_backoff`` — a provider that asks for two minutes gets two
    minutes. A caller that cannot afford an unbounded stall should clamp inside
    its own ``sleep``, which receives the computed delay.
    """
    attempt = 0
    while True:
        try:
            return await submit()
        except retry_on as exc:
            if retry_if is not None and not retry_if(exc):
                raise
            if attempt >= max_retries:
                raise ProviderError(
                    f"failed after {max_retries} retries: {exc}"
                ) from exc
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = retry_after
            else:
                capped = min(base_backoff * (2**attempt), max_backoff)
                delay = random.uniform(0, capped)
            attempt += 1
            await sleep(delay)
