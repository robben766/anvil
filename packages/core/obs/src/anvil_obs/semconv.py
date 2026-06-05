"""OTEL GenAI 语义约定字段(对齐 gen-ai spans 草案;自研采集层统一用这些 key)。"""

GEN_AI_SYSTEM = "gen_ai.system"                      # provider: deepseek / dashscope
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
# 以下为 anvil 自有扩展(OTEL 尚无标准 key,前缀区分)
ANVIL_CACHED_TOKENS = "anvil.usage.cached_tokens"
ANVIL_CACHE_HIT_RATE = "anvil.usage.cache_hit_rate"
ANVIL_COST_CNY = "anvil.usage.cost_cny"
ANVIL_TTFT_MS = "anvil.latency.ttft_ms"
ANVIL_SESSION_ID = "anvil.session_id"
