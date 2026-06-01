"""Verify that EventListener is now a pure-Python class (C1 conversion)."""

import unittest

import hummingbot.core.event.event_listener as el_module
from hummingbot.core.event.event_listener import EventListener


class EventListenerPyTest(unittest.TestCase):
    def test_module_is_python_file(self):
        """The module must resolve to a .py file, not a compiled .so."""
        assert el_module.__file__ is not None
        assert el_module.__file__.endswith(".py"), f"Expected .py, got: {el_module.__file__}"

    def test_c_call_routes_to_dunder_call(self):
        """c_call must delegate to __call__ so PubSub dispatch works."""
        received: list = []

        class Listener(EventListener):
            def __call__(self, arg):
                received.append(arg)

        listener = Listener()
        listener.c_call("hello")
        self.assertEqual(["hello"], received)

    def test_c_set_event_info_stores_context(self):
        """c_set_event_info must populate the event context properties."""
        listener = EventListener.__new__(EventListener)
        listener.__init__()
        sentinel = object()
        listener.c_set_event_info(42, sentinel)  # type: ignore[arg-type]
        self.assertEqual(42, listener.current_event_tag)
        self.assertIs(sentinel, listener.current_event_caller)

    def test_c_set_event_info_reset(self):
        """PubSub resets event info to (0, None) after dispatch."""
        listener = EventListener.__new__(EventListener)
        listener.__init__()
        listener.c_set_event_info(7, object())
        listener.c_set_event_info(0, None)
        self.assertEqual(0, listener.current_event_tag)
        self.assertIsNone(listener.current_event_caller)

    def test_base_call_raises_not_implemented(self):
        """Unsubclassed EventListener.__call__ must raise NotImplementedError."""
        listener = EventListener.__new__(EventListener)
        listener.__init__()
        with self.assertRaises(NotImplementedError):
            listener("arg")

    def test_weakref_supported(self):
        """PubSub registers listeners as weak references."""
        import weakref

        class Listener(EventListener):
            def __call__(self, arg):
                pass

        listener = Listener()
        ref = weakref.ref(listener)
        self.assertIs(listener, ref())

    def test_initial_state(self):
        """Newly created listener has zeroed event context."""

        class Listener(EventListener):
            def __call__(self, arg):
                pass

        listener = Listener()
        self.assertEqual(0, listener.current_event_tag)
        self.assertIsNone(listener.current_event_caller)


if __name__ == "__main__":
    unittest.main()
