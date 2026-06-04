"""错误分类法:决定重试/切换/直接抛(见 spec §4.4)。"""

from __future__ import annotations


class GatewayError(Exception):
    """网关错误基类。"""


class RetryableError(GatewayError):
    """429 限流 / 超时 / 5xx — 同 provider 退避重试,仍失败则切 fallback 链。"""


class FatalRequestError(GatewayError):
    """4xx 请求错误(参数错、context 超长)— 不重试不切换,直接抛出。"""


class FatalAuthError(GatewayError):
    """401/402/403 — 标记该 provider 冷却,本次切换下一家。"""


class AllProvidersFailedError(GatewayError):
    """fallback 链全部失败。"""


def classify_status(status: int) -> type[GatewayError]:
    if status == 429 or status >= 500:
        return RetryableError
    if status in (401, 402, 403):
        return FatalAuthError
    return FatalRequestError
