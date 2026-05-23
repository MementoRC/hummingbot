from unittest.mock import MagicMock, patch

import pytest

from hummingbot.logger.application_warning import ApplicationWarning
from hummingbot.logger.logger import HummingbotLogger


@pytest.fixture()
def hb_logger():
    return HummingbotLogger("test.hb_logger_coverage")


@pytest.fixture(autouse=True)
def reset_logger_callbacks():
    """Reset callback attributes between tests to prevent state leakage."""
    yield
    HummingbotLogger._notify_callback = None
    HummingbotLogger._network_callback = None


def test_network_with_app_warning_msg_not_in_testing_mode(hb_logger):
    """Lines 97, 100: network() calls registered callback with ApplicationWarning
    when app_warning_msg is provided and NOT in testing mode."""
    mock_callback = MagicMock()
    HummingbotLogger.register_network_handler(mock_callback)

    with patch.object(HummingbotLogger, "is_testing_mode", return_value=False):
        hb_logger.network("network log message", app_warning_msg="something is wrong")

    mock_callback.assert_called_once()
    warning_arg = mock_callback.call_args[0][0]
    # The ApplicationWarning should carry the warning message
    assert isinstance(warning_arg, ApplicationWarning)
    assert warning_arg.warning_msg == "something is wrong"


def test_network_no_app_warning_when_testing_mode(hb_logger):
    """network() must NOT call callback when is_testing_mode() returns True."""
    mock_callback = MagicMock()
    HummingbotLogger.register_network_handler(mock_callback)

    with patch.object(HummingbotLogger, "is_testing_mode", return_value=True):
        hb_logger.network("network log message", app_warning_msg="should be ignored")

    mock_callback.assert_not_called()


def test_network_no_app_warning_when_msg_is_none(hb_logger):
    """network() with app_warning_msg=None must not call callback."""
    mock_callback = MagicMock()
    HummingbotLogger.register_network_handler(mock_callback)

    with patch.object(HummingbotLogger, "is_testing_mode", return_value=False):
        hb_logger.network("just a network log")

    mock_callback.assert_not_called()


def test_is_testing_mode_returns_true_during_pytest():
    """is_testing_mode() must detect pytest in sys.argv."""
    assert HummingbotLogger.is_testing_mode() is True


def test_logger_name_for_class():
    class _Dummy:
        pass

    name = HummingbotLogger.logger_name_for_class(_Dummy)
    assert "_Dummy" in name
