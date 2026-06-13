"""Concurrent axon requests using Bittensor dendrite request preparation."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp


class DendriteTransport:
    def __init__(self, dendrite: Any, *, max_response_bytes: int = 128 * 1024):
        self.dendrite = dendrite
        self.max_response_bytes = int(max_response_bytes)

    async def query(self, *, axons: list[Any], synapse: Any) -> list[Any]:
        if not axons:
            return []
        if not hasattr(self.dendrite, "_get_endpoint_url"):
            return await self._sdk_query(axons=axons, synapse=synapse)

        prepared = [self._prepare(axon, synapse) for axon in axons]
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=256)) as session:
            responses = await asyncio.gather(
                *[
                    self._post(
                        session=session,
                        url=item["url"],
                        headers=item["headers"],
                        body=item["body"],
                        timeout=item["timeout"],
                    )
                    for item in prepared
                ],
                return_exceptions=True,
            )
            return await asyncio.gather(
                *[
                    self._complete(item["synapse"], response)
                    for item, response in zip(prepared, responses)
                ]
            )

    async def _sdk_query(self, *, axons: list[Any], synapse: Any) -> list[Any]:
        timeout = float(getattr(synapse, "timeout", getattr(synapse, "time_limit", 10.0)))
        call = getattr(self.dendrite, "forward", None) or self.dendrite
        result = call(axons=axons, synapse=synapse, timeout=timeout, deserialize=False)
        if hasattr(result, "__await__"):
            result = await result
        return list(result)

    def _prepare(self, axon: Any, synapse: Any) -> dict[str, Any]:
        request_name = synapse.__class__.__name__
        copy_fn = getattr(synapse, "model_copy", None)
        outbound = copy_fn() if copy_fn else synapse.copy()
        timeout = float(getattr(outbound, "timeout", getattr(outbound, "time_limit", 10.0)))
        outbound = self.dendrite.preprocess_synapse_for_request(axon, outbound, timeout)
        return {
            "synapse": outbound,
            "url": self.dendrite._get_endpoint_url(axon, request_name=request_name),
            "headers": outbound.to_headers(),
            "body": outbound.model_dump(),
            "timeout": timeout,
        }

    async def _post(
        self,
        *,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float,
    ) -> aiohttp.ClientResponse:
        return await session.post(url, headers=headers, json=body, timeout=timeout)

    async def _complete(self, synapse: Any, response: Any) -> Any:
        request_name = response.__class__.__name__
        if isinstance(response, Exception):
            return self.dendrite.process_error_message(
                synapse=synapse,
                request_name=request_name,
                exception=response,
            )

        content_length = int(response.headers.get("Content-Length", 0))
        if content_length > self.max_response_bytes:
            return self.dendrite.process_error_message(
                synapse=synapse,
                request_name=request_name,
                exception=ValueError(f"response too large: {content_length} bytes"),
            )

        try:
            payload = await response.json()
            self.dendrite.process_server_response(response, payload, synapse)
            return synapse
        except Exception as exc:
            return self.dendrite.process_error_message(
                synapse=synapse,
                request_name=request_name,
                exception=exc,
            )
