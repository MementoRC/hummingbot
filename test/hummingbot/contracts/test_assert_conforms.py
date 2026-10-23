from typing import Protocol

import pytest

from test.hummingbot.contracts import assert_conforms


class _Port(Protocol):
    value: int

    def read(self) -> str: ...

    async def fetch(self) -> str: ...


class _Good:
    value = 1

    def read(self) -> str:
        return "x"

    async def fetch(self) -> str:
        return "y"


class _MissingMember:
    value = 1

    def read(self) -> str:
        return "x"


class _WrongAsyncness:
    value = 1

    def read(self) -> str:
        return "x"

    def fetch(self) -> str:
        return "y"


class _NotCallable:
    value = 1
    read = "nope"

    async def fetch(self) -> str:
        return "y"


def test_conforming_object_passes():
    assert_conforms(_Good(), _Port)


def test_missing_member_fails():
    with pytest.raises(AssertionError, match="missing member 'fetch'"):
        assert_conforms(_MissingMember(), _Port)


def test_sync_async_mismatch_fails():
    with pytest.raises(AssertionError, match="member 'fetch' must be async"):
        assert_conforms(_WrongAsyncness(), _Port)


def test_non_callable_member_fails():
    with pytest.raises(AssertionError, match="member 'read' is not callable"):
        assert_conforms(_NotCallable(), _Port)


def test_missing_annotated_attribute_fails():
    class _NoValue:
        def read(self) -> str:
            return "x"

        async def fetch(self) -> str:
            return "y"

    with pytest.raises(AssertionError, match="missing member 'value'"):
        assert_conforms(_NoValue(), _Port)
