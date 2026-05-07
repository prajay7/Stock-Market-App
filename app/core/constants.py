HEALTH_OK = "ok"
DEFAULT_HORIZON_DAYS = 1

OPENAI_FAST_MODEL = "google/gemini-2.5-flash"
OPENAI_SEARCH_MODEL = "perplexity/sonar"
OPENAI_CHEAP_MODEL = "deepseek/deepseek-chat"

OPENAI_STOCK_MODEL_ALIASES: tuple[str, ...] = (
    "openai_stock_llm_fast",
    "openai_stock_llm",
    "openai_stock_llm_search",
    "openai_stock_llm_cheap",
)
OPENAI_STOCK_MODEL_ALIASES_SET = frozenset(OPENAI_STOCK_MODEL_ALIASES)
