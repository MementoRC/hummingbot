class MonitoredClientSession:
    """
    A monitored version of aiohttp.ClientSession that forwards all the calls to an existing ClientSession object and
    monitors its close method. The close method can be monitored by calling `is_closed()`.

    :param session: an existing and valid ClientSession object to wrap
    :type session: aiohttp.ClientSession
    """

    def __init__(self, session):
        self._session = session
        self._original_close = session.close
        print(f"Creating MonitoredClientSession: {session.close}")
        self._closed = False
        session.close = self._monitored_close

    def is_closed(self):
        """
        Checks whether the `close` method has been called on the wrapped `ClientSession`.

        :return: `True` if the `close` method has been called, `False` otherwise.
        :rtype: bool
        """
        return self._closed

    async def close(self):
        """
        Closes the wrapped `ClientSession` and sets the `_closed` flag to `True`.

        :raises: Any exception raised by the wrapped `ClientSession`'s `close` method.
        """
        if self._session is not None:
            print("Closing session from MonitoredClientSession")
            await self._original_close()
            self._closed = True
            self._session = None
            self._original_close = None

    async def _monitored_close(self):
        """
        Monitors the wrapped `ClientSession`'s `close` method and sets the `_closed` flag to `True`.

        :raises: Any exception raised by the wrapped `ClientSession`'s `close` method.
        """
        print("Closing session from ClientSession triggering MonitoredClientSession")
        if self._session is not None:
            await self._original_close()
            self._closed = True
            self._session = None
            self._original_close = None

    # Forward all other method calls to the underlying session object
    def __getattr__(self, attr):
        """
        Wraps all other method calls to the underlying `aiohttp.ClientSession` instance.

        :param attr: The name of the method to call.

        Example:
        --------
        monitored_session = MonitoredClientSession(session)
        response = await monitored_session.get('https://www.example.com')
        """
        return getattr(self._session, attr)

    def __enter__(self):
        return self._session.__enter__()

    async def __aenter__(self):
        return await self._session.__aenter__()

    async def __aexit__(self, *args):
        return await self._session.__aexit__(*args)
