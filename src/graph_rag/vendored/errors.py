"""VENDORED — not demo-authored code.

Origin: ``apps/ai/shared/errors.py`` in the sibling ``monorepo`` checkout.

Reduced to the exceptions the vendored parsing path raises. The origin's retry/attempt
bookkeeping belongs to that system's worker framework and is dropped here.
"""

class AIError(Exception):
    """Base exception for all AI platform errors."""


class ProviderError(AIError):
    """Raised when a provider operation fails (network, auth, quota, …).

    Carries the provider's HTTP ``status_code`` when ``providers.transport``
    rejected an HTTP response, ``None`` when there was none (connection
    error, unparsable payload) or the raiser doesn't set it. Unlike the
    message, it is safe to log: it tells a rejected key (401/403), an empty
    account (402) and an invalid model (400/404) apart without any provider
    text.
    """

    def __init__(self, message: str = "", *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Raised when a provider signals rate limiting (HTTP 429).

    Carries ``retry_after`` (seconds) when the provider advises one so callers
    can honour it instead of guessing. Subclass of :class:`ProviderError` so
    existing handlers keep working; retry code catches this narrower type.
    """

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ServiceUnavailableError(ProviderError):
    """Raised when a provider is temporarily unavailable (HTTP 503).

    Carries ``retry_after`` (seconds) when advised. Same subclass rationale as
    :class:`RateLimitError`.
    """

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ParseTimeoutError(ProviderError):
    """Raised when parsing a document ran out of time.

    Raised by the self-hosted client when ``unstructured-api`` doesn't answer within
    its timeout.

    LOCAL CHANGE: the origin's ``max_attempts`` budget is dropped with the worker
    framework that read it.
    """


#: HTTP statuses with which a parser refuses the file itself (#1777): the
#: bytes are malformed (400), too large for it (413), a format it can't read
#: (415), or unprocessable (422). Every other 4xx describes the deployment —
#: a rotated key (401/403), a URL briefly wrong behind an ingress (404), a
#: slow client (408), rate limiting (429) — and the next delivery may succeed.
FILE_REJECTED_STATUSES = frozenset({400, 413, 415, 422})


class FileRejectedError(ProviderError):
    """Raised when the parser refused the file itself (#1777).

    Raised by the parser clients, on the one request that carries the file,
    for a status in :data:`FILE_REJECTED_STATUSES`; only the client knows
    which request that was. Never retried: every redelivery would send the
    same bytes and get the same answer.
    """

    retryable = False
