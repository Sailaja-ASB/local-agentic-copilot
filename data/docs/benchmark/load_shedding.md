# Load Shedding and Backpressure

Load shedding means rejecting new work quickly when a service is already full, instead of letting every request wait in a growing queue. A gateway can keep a counter of in-flight generations and return HTTP 503 Service Unavailable immediately once a limit such as MAX_INFLIGHT is reached.

The trade-off is explicit: a lower limit causes more rejected requests but keeps latency acceptable for the requests that are accepted; a higher limit rejects less but lets tail latency explode. Clients that receive a 503 should back off and retry later.
