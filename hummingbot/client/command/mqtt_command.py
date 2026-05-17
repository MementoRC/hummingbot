import asyncio
import threading
import time
from typing import TYPE_CHECKING

from remote_iface.hb_compat import create_gateway
from remote_iface.protocols.config import BrokerConfig, GatewayConfig

from hummingbot.core.utils.async_utils import safe_ensure_future

if TYPE_CHECKING:
    from hummingbot.client.hummingbot_application import HummingbotApplication  # noqa: F401


SUBCOMMANDS = ["start", "stop", "restart"]


class MQTTCommand:
    _mqtt_sleep_rate_connection_check: float = 1.0
    _mqtt_sleep_rate_autostart_retry: float = 10.0

    def mqtt_start(
        self,  # type: HummingbotApplication
        timeout: float = 30.0,
    ):
        if threading.current_thread() != threading.main_thread():
            self.ev_loop.call_soon_threadsafe(self.mqtt_start, timeout)
            return
        safe_ensure_future(self.start_mqtt_async(timeout=timeout), loop=self.ev_loop)

    def mqtt_stop(
        self,  # type: HummingbotApplication
    ):
        if threading.current_thread() != threading.main_thread():
            self.ev_loop.call_soon_threadsafe(self.mqtt_stop)
            return
        safe_ensure_future(self.stop_mqtt_async(), loop=self.ev_loop)

    def mqtt_restart(
        self,  # type: HummingbotApplication
        timeout: float = 30.0,
    ):
        if threading.current_thread() != threading.main_thread():
            self.ev_loop.call_soon_threadsafe(self.mqtt_restart, timeout)
            return
        safe_ensure_future(self.restart_mqtt_async(timeout=timeout), loop=self.ev_loop)

    async def start_mqtt_async(
        self,  # type: HummingbotApplication
        timeout: float = 30.0,
    ):
        if self._mqtt is None:
            while True:
                try:
                    start_t = time.time()
                    self.logger().info("Connecting MQTT Bridge...")
                    cm = self.client_config_map.mqtt_bridge
                    broker_cfg = BrokerConfig(
                        host=cm.mqtt_host,
                        port=cm.mqtt_port,
                        username=cm.mqtt_username,
                        password=cm.mqtt_password,
                        ssl=cm.mqtt_ssl,
                    )
                    gateway_cfg = GatewayConfig(
                        namespace=cm.mqtt_namespace,
                        broker=broker_cfg,
                        enable_commands=cm.mqtt_commands,
                        enable_notifier=cm.mqtt_notifier,
                        enable_events=cm.mqtt_events,
                        enable_external_events=cm.mqtt_external_events,
                        enable_log_handler=cm.mqtt_logger,
                    )
                    self._mqtt = create_gateway(self, config=gateway_cfg, broker_config=broker_cfg)
                    self._mqtt.start()
                    while True:
                        if time.time() - start_t > timeout:
                            raise Exception(f"Connection timed out after {timeout} seconds")
                        if self._mqtt.health:
                            self.logger().info("MQTT Bridge connected with success.")
                            break
                        await asyncio.sleep(self._mqtt_sleep_rate_connection_check)
                    break
                except Exception as e:
                    if self.client_config_map.mqtt_bridge.mqtt_autostart:
                        s = self._mqtt_sleep_rate_autostart_retry
                        self.logger().error(f"Failed to connect MQTT Bridge: {str(e)}. Retrying in {s} seconds.")
                        self.notify(f"MQTT Bridge failed to connect to the broker, retrying in {s} seconds.")
                    else:
                        self.logger().error(f"Failed to connect MQTT Bridge: {str(e)}")
                        self.notify("MQTT Bridge failed to connect to the broker.")
                    self._mqtt.stop()
                    self._mqtt = None

                    if self.client_config_map.mqtt_bridge.mqtt_autostart:
                        await asyncio.sleep(self._mqtt_sleep_rate_autostart_retry)
                    else:
                        break

        else:
            self.logger().warning("MQTT Bridge is already running!")
            self.notify("MQTT Bridge is already running!")

    async def stop_mqtt_async(
        self,  # type: HummingbotApplication
    ):
        if self._mqtt is not None:
            try:
                self._mqtt.stop()
                self._mqtt = None
                self.logger().info("MQTT Bridge disconnected")
                self.notify("MQTT Bridge disconnected")
            except Exception as e:
                self.logger().error(f"Failed to stop MQTT Bridge: {str(e)}")
        else:
            self.logger().error("MQTT is already stopped!")
            self.notify("MQTT Bridge is already stopped!")

    async def restart_mqtt_async(
        self,  # type: HummingbotApplication
        timeout: float = 2.0,
    ):
        await self.stop_mqtt_async()
        await self.start_mqtt_async(timeout)
