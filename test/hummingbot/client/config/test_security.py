import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from test.isolated_asyncio_wrapper_test_case import LocalTestEventLoopWrapperTestCase

from hummingbot.client.config import config_crypt, config_helpers, security
from hummingbot.client.config.config_crypt import ETHKeyFileSecretManger, store_password_verification, validate_password
from hummingbot.client.config.config_helpers import (
    ClientConfigAdapter,
    api_keys_from_connector_config_map,
    get_connector_config_yml_path,
    save_to_yml,
)
from hummingbot.client.config.security import Security
from hummingbot.connector.exchange.binance.binance_utils import BinanceConfigMap


class SecurityTest(LocalTestEventLoopWrapperTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The async call scheduler no longer contain a reference to an event loop
        # AsyncCallScheduler.shared_instance().reset_event_loop()

    def setUp(self) -> None:
        super().setUp()
        self.new_conf_dir_path = TemporaryDirectory()
        self.default_pswrd_verification_path = security.PASSWORD_VERIFICATION_PATH
        self.default_connectors_conf_dir_path = config_helpers.CONNECTORS_CONF_DIR_PATH
        mock_conf_dir = Path(self.new_conf_dir_path.name) / "conf"
        mock_conf_dir.mkdir(parents=True, exist_ok=True)
        config_crypt.PASSWORD_VERIFICATION_PATH = mock_conf_dir / ".password_verification"

        security.PASSWORD_VERIFICATION_PATH = config_crypt.PASSWORD_VERIFICATION_PATH
        config_helpers.CONNECTORS_CONF_DIR_PATH = (
            Path(self.new_conf_dir_path.name) / "connectors"
        )
        config_helpers.CONNECTORS_CONF_DIR_PATH.mkdir(parents=True, exist_ok=True)
        self.connector = "binance"
        self.api_key = "someApiKey"
        self.api_secret = "someSecret"

    def tearDown(self) -> None:
        config_crypt.PASSWORD_VERIFICATION_PATH = self.default_pswrd_verification_path
        security.PASSWORD_VERIFICATION_PATH = config_crypt.PASSWORD_VERIFICATION_PATH
        config_helpers.CONNECTORS_CONF_DIR_PATH = self.default_connectors_conf_dir_path
        self.new_conf_dir_path.cleanup()
        self.reset_security()
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        # The async call scheduler no longer contain a reference to an event loop
        # AsyncCallScheduler.shared_instance().reset_event_loop()

    def store_binance_config(self) -> ClientConfigAdapter:
        config_map = ClientConfigAdapter(
            BinanceConfigMap(binance_api_key=self.api_key, binance_api_secret=self.api_secret)
        )
        file_path = get_connector_config_yml_path(self.connector)
        save_to_yml(file_path, config_map)
        return config_map

    @staticmethod
    def reset_decryption_done():
        Security._decryption_done = asyncio.Event()

    @staticmethod
    def reset_security():
        Security.__instance = None
        Security.secrets_manager = None
        Security._secure_configs = {}
        Security._decryption_done = asyncio.Event()

    def test_password_process(self):
        self.assertTrue(Security.new_password_required())

        password = "som-password"
        secrets_manager = ETHKeyFileSecretManger(password)
        store_password_verification(secrets_manager)

        self.assertFalse(Security.new_password_required())
        self.assertTrue(validate_password(secrets_manager))

        another_secrets_manager = ETHKeyFileSecretManger("another-password")

        self.assertFalse(validate_password(another_secrets_manager))

    def test_login(self):
        password = "som-password"
        secrets_manager = ETHKeyFileSecretManger(password)
        store_password_verification(secrets_manager)

        Security.login(secrets_manager)
        self.run_async_with_timeout(Security.wait_til_decryption_done())
        config_map = self.store_binance_config()
        self.local_event_loop.run_until_complete(asyncio.sleep(0.1))
        self.reset_decryption_done()
        Security.decrypt_all()
        self.run_async_with_timeout(Security.wait_til_decryption_done())

        self.assertTrue(Security.is_decryption_done())
        self.assertTrue(Security.any_secure_configs())
        self.assertTrue(Security.connector_config_file_exists(self.connector))

        api_keys = Security.api_keys(self.connector)
        expected_keys = api_keys_from_connector_config_map(config_map)

        self.assertEqual(expected_keys, api_keys)

    def test_update_secure_config(self):
        password = "som-password"
        secrets_manager = ETHKeyFileSecretManger(password)
        store_password_verification(secrets_manager)

        Security.login(secrets_manager)
        self.run_async_with_timeout(Security.wait_til_decryption_done(), 20)
        self.local_event_loop.run_until_complete(asyncio.sleep(0.1))

        binance_config = ClientConfigAdapter(
            BinanceConfigMap(binance_api_key=self.api_key, binance_api_secret=self.api_secret)
        )
        Security.update_secure_config(binance_config)
        self.local_event_loop.run_until_complete(asyncio.sleep(0.1))

        self.reset_decryption_done()
        Security.decrypt_all()
        self.local_event_loop.run_until_complete(asyncio.sleep(0.1))
        self.run_async_with_timeout(Security.wait_til_decryption_done())

        binance_loaded_config = Security.decrypted_value(binance_config.connector)

        self.assertEqual(binance_config, binance_loaded_config)

        binance_config.binance_api_key = "someOtherApiKey"
        Security.update_secure_config(binance_config)

        self.reset_decryption_done()
        Security.decrypt_all()
        self.local_event_loop.run_until_complete(asyncio.sleep(0.1))
        self.run_async_with_timeout(Security.wait_til_decryption_done())

        binance_loaded_config = Security.decrypted_value(binance_config.connector)

        self.assertEqual(binance_config, binance_loaded_config)
