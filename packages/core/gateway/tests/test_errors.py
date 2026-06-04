import pytest

from anvil_gateway.errors import (
    FatalAuthError,
    FatalRequestError,
    GatewayError,
    RetryableError,
    classify_status,
)


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (429, RetryableError),
        (500, RetryableError),
        (503, RetryableError),
        (401, FatalAuthError),
        (402, FatalAuthError),
        (403, FatalAuthError),
        (400, FatalRequestError),
        (404, FatalRequestError),
    ],
)
def test_classify_status(status, exc):
    assert classify_status(status) is exc


def test_hierarchy():
    assert issubclass(RetryableError, GatewayError)
