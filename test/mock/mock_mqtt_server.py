import logging
from typing import Any

from commlib.serializer import JSONSerializer
import ujson


class FakeMQTTMessage(object):
    def __init__(self, topic, payload):
        self.topic = topic
        fake_payload = {"header": {"reply_to": f"test_reply/{topic}"}, "data": payload}
        self.payload = ujson.dumps(fake_payload)


class FakeMQTTBroker:
    """Fake MQTT broker that aggregates subscriptions across all per-endpoint transports.

    commlib creates one MQTTTransport per endpoint (Publisher, RPCService, PSubscriber, etc.).
    The test patches commlib.transports.mqtt.MQTTTransport so every constructor call goes
    through create_transport().  Each call returns a *new* FakeMQTTTransport that shares the
    broker's single _subscriptions and _received_msgs dicts.  This ensures:
      - Each RPCService's transport starts with is_connected=False, so its run() launches the
        run_forever() thread and calls _transport.subscribe(), registering the RPC command topic.
      - All subscriptions and published messages land in one place regardless of which endpoint
        transport they originate from, so publish_to_subscription() and received_msgs work
        transparently across all endpoints.
    """

    def __init__(self):
        self._subscriptions: dict[str, Any] = {}
        self._received_msgs: dict[str, Any] = {}

    def create_transport(self, *args, **kwargs):
        """Return a new per-endpoint transport that shares the broker's shared state dicts."""
        return FakeMQTTTransport(self._subscriptions, self._received_msgs)

    def publish_to_subscription(self, topic, payload):
        callback = self._subscriptions[topic]
        msg = FakeMQTTMessage(topic=topic, payload=payload)
        callback(client=None, userdata=None, msg=msg)

    @property
    def subscriptions(self):
        return self._subscriptions

    @property
    def received_msgs(self):
        return self._received_msgs

    def is_msg_received(self, topic, content=None, msg_key="msg"):
        msg_found = False
        if topic in self.received_msgs:
            if not content:
                msg_found = True
            else:
                for msg in self.received_msgs[topic]:
                    if str(content) == str(msg[msg_key]):
                        msg_found = True
                        break
        return msg_found

    def clear(self):
        self._subscriptions.clear()
        self._received_msgs.clear()


class FakeMQTTTransport:
    """Per-endpoint transport whose subscription and message dicts are shared with the broker."""

    def __init__(self, subscriptions: dict[str, Any], received_msgs: dict[str, Any]):
        self._subscriptions = subscriptions
        self._received_msgs = received_msgs
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def publish(self, topic: str, payload: dict[str, Any], qos: Any = "", retain: bool = False):
        logging.info(f"\nFakeMQTT publish on\n> {topic}\n     {payload}\n")
        payload = ujson.loads(JSONSerializer.serialize(payload))
        if not self._received_msgs.get(topic):
            self._received_msgs[topic] = []
        self._received_msgs[topic].append(payload)

    def subscribe(self, topic: str, callback: Any, *args, **kwargs):
        self._subscriptions[topic] = callback
        return topic

    def start(self):
        self._connected = True

    def connect(self):
        self._connected = True

    def stop(self):
        self._connected = False

    def loop_forever(self):
        self.start()
