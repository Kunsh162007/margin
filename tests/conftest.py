"""Shared fixtures.

``no_network`` makes "offline" something a test can fail on: any socket
connection or DNS lookup to a host other than this machine raises.
"""

import pytest

from evals.netguard import NetworkBlocked, blocked_network


@pytest.fixture
def no_network():
    with blocked_network():
        yield NetworkBlocked
