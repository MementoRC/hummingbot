from google.protobuf.json_format import MessageToDict
from web_assistant.connections.data_types import WSResponse
from web_assistant.ws_post_processors import WSPostProcessorBase

from hummingbot.connector.exchange.mexc.protobuf import PushDataV3ApiWrapper_pb2


class MexcPostProcessor(WSPostProcessorBase):
    async def post_process(response: WSResponse) -> WSResponse:
        message = response.data
        try:
            if isinstance(message, dict):
                return response
            # Not a dict, continue processing as Protobuf
            # Deserialize the message
            result = PushDataV3ApiWrapper_pb2.PushDataV3ApiWrapper()
            result.ParseFromString(message)
            # Convert message to dict
            response.data = MessageToDict(result)
            return response
        except Exception:
            raise
