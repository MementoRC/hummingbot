from hummingbot.core.web_assistant.connections.persistent_client_session import PersistentClientSession
from hummingbot.core.web_assistant.connections.rest_connection import RESTConnection
from hummingbot.core.web_assistant.connections.ws_connection import WSConnection


class ConnectionsFactory:
    """This class is a thin wrapper around the underlying REST and WebSocket third-party library.

    The purpose of the class is to isolate the general `web_assistant` infrastructure from the underlying library
    (in this case, `aiohttp`) to enable dependency change with minimal refactoring of the code.

    Note: One future possibility is to enable injection of a specific connection factory implementation in the
    `WebAssistantsFactory` to accommodate cases such as Bittrex that uses a specific WebSocket technology requiring
    a separate third-party library. In that case, a factory can be created that returns `RESTConnection`s using
    `aiohttp` and `WSConnection`s using `signalr_aio`.
    """
    @staticmethod
    async def get_rest_connection() -> RESTConnection:
        session = await PersistentClientSession().get_context_client()
        return RESTConnection(aiohttp_client_session=session)

    @staticmethod
    async def get_ws_connection() -> WSConnection:
        session = await PersistentClientSession().get_context_client()
        return WSConnection(aiohttp_client_session=session)
