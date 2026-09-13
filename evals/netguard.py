"""Refuse every network connection that leaves this machine (D15).

Used by the pytest ``no_network`` fixture and by ``ui_flow_eval``. Connections
and DNS lookups to loopback still work, because the model server lives there.
Every refused attempt is recorded: a library that catches the error and carries
on would otherwise hide that it tried.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


class NetworkBlocked(OSError):
    pass


def _host(address: object) -> str:
    return str(address[0]) if isinstance(address, tuple) else str(address)


@contextmanager
def blocked_network(attempts: list[str] | None = None) -> Iterator[list[str]]:
    seen = attempts if attempts is not None else []
    real_connect, real_connect_ex, real_getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def refuse(target: str) -> None:
        seen.append(target)
        raise NetworkBlocked(f"network access blocked: {target}")

    def connect(self: socket.socket, address: object) -> None:
        if _host(address) not in LOOPBACK:
            refuse(repr(address))
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        if _host(address) not in LOOPBACK:
            refuse(repr(address))
        return real_connect_ex(self, address)

    def getaddrinfo(host: object, *args: object, **kwargs: object):
        if host not in LOOPBACK:
            refuse(f"DNS {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo
    try:
        yield seen
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = real_connect, real_connect_ex, real_getaddrinfo
