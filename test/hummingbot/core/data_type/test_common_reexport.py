"""Guards the PARENT's re-export contract for ``hummingbot.core.data_type.common``.

This module only asserts that ``hummingbot.core.data_type.common`` re-exports
the exact same objects defined in ``data_type_primitives.common`` and defines
nothing of its own. The enum/behavioral semantics of the underlying types
(``OrderType``, ``TradeType``, etc.) are owned and tested by the
``hb-data-type-primitives`` sub-package (issue #11, PR #12). Nothing here
should re-assert enum member values or predicate behavior.
"""

import inspect
from unittest import TestCase

import data_type_primitives.common as primitives_common

import hummingbot.core.data_type.common as common


class CommonReexportTests(TestCase):
    def test_order_type_is_identical_object(self):
        # common.py's own docstring warns that Enum equality is identity-based:
        # a second, structurally identical OrderType class would compare unequal
        # to the canonical one, silently breaking fee-class selection with no
        # import error. This is exactly the regression that historically
        # corrupted common.py and failed the bleeding-edge quality gate.
        self.assertIs(
            common.OrderType,
            primitives_common.OrderType,
            "hummingbot.core.data_type.common.OrderType is not the same object as "
            "data_type_primitives.common.OrderType. common.py must re-export "
            "OrderType, never redefine it: Enum equality is identity-based, so a "
            "duplicate class definition would silently select the wrong fee class "
            "instead of raising an import error.",
        )

    def test_all_names_are_reexported_identically(self):
        for name in common.__all__:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(common, name),
                    f"{name} is listed in common.__all__ but is not importable from hummingbot.core.data_type.common",
                )
                self.assertTrue(
                    hasattr(primitives_common, name),
                    f"{name} is listed in common.__all__ but does not exist in data_type_primitives.common",
                )
                self.assertIs(
                    getattr(common, name),
                    getattr(primitives_common, name),
                    f"hummingbot.core.data_type.common.{name} is not the same "
                    f"object as data_type_primitives.common.{name}",
                )

    def test_common_defines_nothing_of_its_own(self):
        # common.py must stay a pure re-export: no class or def should be
        # attributed to this module rather than to data_type_primitives.common.
        own_members = [
            member_name
            for member_name, member in inspect.getmembers(common)
            if (inspect.isclass(member) or inspect.isfunction(member))
            and getattr(member, "__module__", None) == common.__name__
        ]
        self.assertEqual(
            [],
            own_members,
            f"hummingbot.core.data_type.common defines its own objects "
            f"{own_members}; it must only re-export from "
            f"data_type_primitives.common",
        )


if __name__ == "__main__":
    TestCase.main()
