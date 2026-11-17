# Circuit Breakers and Retries

A circuit breaker protects a service from repeatedly calling a dependency that is failing. After a number of consecutive failures the breaker opens and requests fail fast without touching the dependency; after a cool-down it lets a trial request through (half-open) and closes again if that succeeds.

Retries should be limited and should use exponential backoff with jitter. Only retry errors that are likely to be temporary, such as connection failures, and never blindly retry a slow generation, because retrying overloaded model servers multiplies the load.
