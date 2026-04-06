import pytest

from hummingbot.client.hummingbot_application import HummingbotApplication


class HummingbotApplicationTest:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.app = HummingbotApplication()

    def test_set_strategy_file_name(self):
        strategy_name = "some-strategy"
        file_name = f"{strategy_name}.yml"
        self.app.strategy_file_name = file_name

        assert file_name == self.app.strategy_file_name

    def test_set_strategy_file_name_to_none(self):
        strategy_name = "some-strategy"
        file_name = f"{strategy_name}.yml"

        self.app.strategy_file_name = None

        assert None is self.app.strategy_file_name

        self.app.strategy_file_name = file_name
        self.app.strategy_file_name = None

        assert None is self.app.strategy_file_name
