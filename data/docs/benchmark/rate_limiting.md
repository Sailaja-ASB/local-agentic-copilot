# API Keys and Rate Limiting

An API gateway authenticates every caller, commonly with an API key sent in a header such as X-API-Key, and rejects missing or invalid keys with HTTP 401. Keys are stored as secrets, never in source code or container images.

Rate limiting caps how many requests each key may make per time window, for example 60 requests per minute, and answers with HTTP 429 Too Many Requests when the cap is exceeded. A token bucket or sliding window keeps one noisy client from starving everyone else.
