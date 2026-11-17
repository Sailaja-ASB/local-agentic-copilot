# Streaming and Time to First Token

With streaming, the server sends tokens to the client as they are generated, typically using Server-Sent Events (SSE), instead of waiting for the full answer. Users start reading almost immediately, which makes the application feel much faster even though total generation time is unchanged.

Time to first token (TTFT) measures how long a user waits before any text appears; total latency measures until the last token. Under load, TTFT grows when requests queue behind others at the model server. A cold start, when the model must first be loaded into memory, also inflates TTFT.
