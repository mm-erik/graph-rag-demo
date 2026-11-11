"""HTTP transport for the self-hosted Unstructured API (open-source project).

Unlike the hosted Jobs API (:mod:`providers.parsing.unstructured_http`), the
self-hosted server is synchronous: a single ``POST /general/v0/general``
request partitions the file and returns its elements directly, no job
create/poll/download cycle.

---

VENDORED — not demo-authored code.

Origin: ``apps/ai/providers/parsing/unstructured_self_hosted_http.py`` in the sibling ``monorepo`` checkout (the production ingestion
service). Copied wholesale, with that system's settings/DI framework stripped; every
deviation from the original is marked ``LOCAL CHANGE``.

Do not improve this file. Its value is that behaviour here matches production, so the
demo predicts the real thing; edits silently decouple the two.
"""
import asyncio
import os
import tempfile
from collections.abc import Sequence
from typing import BinaryIO, cast

import httpx

from graph_rag.vendored.models import RawDocument
from graph_rag.vendored.backoff import parse_retry_after, submit_with_backoff
from graph_rag.vendored.errors import (
    FILE_REJECTED_STATUSES,
    FileRejectedError,
    ParseTimeoutError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)

_PARTITION_PATH = "/general/v0/general"


class SelfHostedUnstructuredHTTPClient:
    """:class:`~providers.parsing.unstructured.UnstructuredClient` backed by a
    self-hosted ``Unstructured-IO/unstructured-api`` instance.

    Transient failures where the request never reached (or was rejected by)
    the server -- connection errors, HTTP 429/503 -- are retried up to
    ``max_retries`` times; once retries are exhausted the failure is raised as
    :class:`~shared.errors.ServiceUnavailableError` so callers (e.g.
    callers) can tell a transient/overload condition apart from a genuinely bad file (raised as
    :class:`~shared.errors.FileRejectedError` for a status in
    :data:`~shared.errors.FILE_REJECTED_STATUSES`, never retried; any other
    error status is a plain :class:`~shared.errors.ProviderError` carrying
    it).
    A client-side *timeout* is deliberately NOT retried and surfaces as
    :class:`~shared.errors.ParseTimeoutError`: unlike a connection error, a
    timeout means the request was already accepted and may still be running,
    so retrying would fire a duplicate hi_res job rather than recovering
    anything -- it would only guarantee blowing through the ingestion
    pipeline's overall stage timeout for large/slow documents.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        strategy: str = "hi_res",
        image_block_types: Sequence[str] = ("Image", "Table"),
        languages: Sequence[str] = ("deu", "eng"),
        timeout: float = 300.0,
        max_retries: int = 2,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
        http_client: "httpx.AsyncClient | None" = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._api_key = api_key
        self._strategy = strategy
        self._image_block_types = list(image_block_types)
        self._languages = list(languages)
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._http_client = http_client


    @property
    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self._api_key:
            headers["unstructured-api-key"] = self._api_key
        return headers

    @property
    def _form_data(self) -> dict[str, str | list[str]]:
        """Multipart form fields. httpx renders a list/tuple value as
        repeated fields of the same name, which is how ``languages`` and
        ``extract_image_block_types`` need to be sent."""
        return {
            "strategy": self._strategy,
            "coordinates": "true",
            "languages": list(self._languages),
            "extract_image_block_types": list(self._image_block_types),
        }

    async def partition(self, raw: RawDocument) -> list[dict]:
        client = self._http_client or httpx.AsyncClient(timeout=self._timeout)
        # Streamed from a temp file (rather than the in-memory bytes directly)
        # so a large upload doesn't hold a second full in-memory copy; removed
        # on both the success and failure paths.
        tmp_path = self._write_temp_file(raw.content)
        try:
            try:
                return await submit_with_backoff(
                    lambda: self._request(client, raw, tmp_path),
                    max_retries=self._max_retries,
                    base_backoff=self._base_backoff,
                    max_backoff=self._max_backoff,
                    retry_on=(RateLimitError, ServiceUnavailableError),
                )
            except ProviderError as exc:
                # submit_with_backoff wraps the final, retries-exhausted
                # failure in a plain ProviderError; recover whether it was a
                # transient condition (fallback-worthy) from the cause chain.
                if isinstance(exc.__cause__, (RateLimitError, ServiceUnavailableError)):
                    raise ServiceUnavailableError(str(exc)) from exc
                raise
        finally:
            self._remove_temp_file(tmp_path)

    @staticmethod
    def _write_temp_file(content: bytes) -> str:
        fd, path = tempfile.mkstemp(prefix="unstructured-self-hosted-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
        except BaseException:
            os.unlink(path)
            raise
        return path

    @staticmethod
    def _remove_temp_file(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            # Best-effort cleanup: if the temp file is already gone or cannot
            # be removed, do not mask the primary operation outcome.
            pass

    async def _request(
        self, client: "httpx.AsyncClient", raw: RawDocument, tmp_path: str
    ) -> list[dict]:
        # Stream the file handle straight into the multipart body (httpx reads
        # it in 64KB chunks) rather than loading the whole file into memory --
        # matters for large uploads. Opened/closed off-thread since a plain
        # open()/close() call would otherwise block the event loop.
        # asyncio.to_thread's Callable[..., T] signature can't select open()'s
        # binary-mode overload, so its return type widens to a text/binary
        # union; cast back to what "rb" mode actually returns.
        fh = cast(BinaryIO, await asyncio.to_thread(open, tmp_path, "rb"))
        try:
            response = await client.post(
                f"{self._base_url}{_PARTITION_PATH}",
                headers=self._headers,
                files={"files": (raw.filename, fh, raw.mime_type)},
                data=self._form_data,
                # Explicit per-request timeout: when ``client`` is the app-wide
                # shared httpx.AsyncClient (no timeout set, so httpx's 5s
                # default applies), self._timeout would otherwise be silently
                # ignored since it's only applied when this class constructs
                # its own client.
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            # Not retried (see class docstring): a client-side timeout means
            # the request was already sent and may still be running
            # server-side, so retrying just queues a duplicate job rather
            # than recovering anything.
            raise ParseTimeoutError(
                f"Self-hosted Unstructured timed out after {self._timeout}s "
                f"for {raw.filename!r}: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            raise ServiceUnavailableError(
                f"Self-hosted Unstructured request failed for {raw.filename!r}: {exc}"
            ) from exc
        finally:
            await asyncio.to_thread(fh.close)

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise RateLimitError(
                f"Self-hosted Unstructured rate-limited the request for "
                f"{raw.filename!r}",
                retry_after=parse_retry_after(response.headers.get("Retry-After")),
            )
        if response.status_code == httpx.codes.SERVICE_UNAVAILABLE:
            raise ServiceUnavailableError(
                f"Self-hosted Unstructured is unavailable for {raw.filename!r} "
                f"(HTTP 503, likely low free memory)",
                retry_after=parse_retry_after(response.headers.get("Retry-After")),
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # This request carries the file, so a 400/413/415/422 here is the
            # server refusing the bytes themselves.
            error_type = (
                FileRejectedError
                if response.status_code in FILE_REJECTED_STATUSES
                else ProviderError
            )
            raise error_type(
                f"Self-hosted Unstructured rejected {raw.filename!r} "
                f"(HTTP {response.status_code}): {response.text[:500]}",
                status_code=response.status_code,
            ) from exc

        try:
            elements = response.json()
        except ValueError as exc:
            # A 2xx response with an unparseable body is still a failure --
            # never hand a caller a partial/garbled result.
            raise ProviderError(
                f"Self-hosted Unstructured returned a non-JSON response for "
                f"{raw.filename!r}: {response.text[:500]}"
            ) from exc
        if not isinstance(elements, list):
            raise ProviderError(
                "Self-hosted Unstructured returned an unexpected payload: "
                f"expected a list of elements, got {type(elements).__name__}"
            )
        return elements
