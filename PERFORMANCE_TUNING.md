# Vega2.0 - Performance tuning for limited hardware

# Add these to your .env file

# Reduce memory footprint

CONTEXT_WINDOW_SIZE=5          # Down from 10 - saves memory
CONTEXT_MAX_CHARS=2000         # Down from 4000 - faster responses
MAX_RESPONSE_CHARS=2000        # Limit response length
MAX_PROMPT_CHARS=2000          # Limit input length

# Speed up generation

GEN_TEMPERATURE=0.7            # Good balance
GEN_TOP_K=20                   # Down from 40 - faster sampling
GEN_TOP_P=0.9                  # Keep this for quality

# Cache everything possible

CACHE_TTL_SECONDS=300          # 5 minutes - reuse responses
RETENTION_DAYS=7               # Clean up old conversations

# Timeout faster on slow hardware

LLM_TIMEOUT_SEC=30             # Don't wait forever
LLM_MAX_RETRIES=1              # Fail fast

# Disable expensive features

PII_MASKING=false              # Skip if you don't need privacy
