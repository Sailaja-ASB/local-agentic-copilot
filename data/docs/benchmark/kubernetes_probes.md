# Kubernetes Liveness and Readiness Probes

A readiness probe tells Kubernetes whether a pod should receive traffic; when it fails, the pod is removed from the Service endpoints but keeps running. A liveness probe tells Kubernetes whether the container is stuck; when it fails repeatedly, the kubelet restarts the container.

Liveness probes must be lenient. If the node is CPU-starved and the probe times out, Kubernetes may restart a pod that was slow but healthy, cutting every request it was serving. Use generous timeoutSeconds and failureThreshold, keep the liveness endpoint cheap, and never make it depend on downstream services.
