from anvil_gateway.adapters.dashscope import DashScopeAdapter
from anvil_gateway.adapters.deepseek import DeepSeekAdapter


def test_deepseek_cached_field():
    # DeepSeek usage 顶层: prompt_cache_hit_tokens / prompt_cache_miss_tokens
    u = {
        "prompt_tokens": 100,
        "completion_tokens": 9,
        "prompt_cache_hit_tokens": 64,
        "prompt_cache_miss_tokens": 36,
    }
    assert DeepSeekAdapter().parse_cached_tokens(u) == 64


def test_deepseek_cached_absent_defaults_zero():
    assert DeepSeekAdapter().parse_cached_tokens({"prompt_tokens": 10}) == 0


def test_dashscope_cached_field():
    # DashScope OpenAI 兼容: prompt_tokens_details.cached_tokens
    u = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 25}}
    assert DashScopeAdapter().parse_cached_tokens(u) == 25


def test_dashscope_details_absent_defaults_zero():
    assert DashScopeAdapter().parse_cached_tokens({"prompt_tokens": 10}) == 0
    assert DashScopeAdapter().parse_cached_tokens({"prompt_tokens_details": None}) == 0


def test_endpoints_and_env():
    ds, qw = DeepSeekAdapter(), DashScopeAdapter()
    assert ds.base_url == "https://api.deepseek.com/v1" and ds.api_key_env == "DEEPSEEK_API_KEY"
    assert qw.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert qw.api_key_env == "DASHSCOPE_API_KEY"
