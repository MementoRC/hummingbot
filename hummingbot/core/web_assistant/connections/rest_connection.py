import aiohttp

from hummingbot.core.web_assistant.connections.data_types import RESTRequest, RESTResponse
from hummingbot.core.web_assistant.connections.persistent_client_session import PersistentClientSession


class RESTConnection:
    def __init__(self, aiohttp_client_session: aiohttp.ClientSession):
        self._client_session = aiohttp_client_session

    async def call(self, request: RESTRequest) -> RESTResponse:
        # This polls or starts the client session for a connection
        async with PersistentClientSession() as client_session:
            aiohttp_resp = await client_session.request(
                method=request.method.value,
                url=request.url,
                params=request.params,
                data=request.data,
                headers=request.headers,
            )

            resp = await self._build_resp(aiohttp_resp)
            return resp

    @staticmethod
    async def _build_resp(aiohttp_resp: aiohttp.ClientResponse) -> RESTResponse:
        resp = RESTResponse(aiohttp_resp)
        return resp
