# Horizontal Pod Autoscaler

The Horizontal Pod Autoscaler (HPA) changes the number of replicas of a Deployment based on metrics, most commonly average CPU utilisation reported by metrics-server. When usage is above the target the HPA adds pods; when it stays below, it removes them after a stabilisation window.

For an LLM gateway CPU is often the wrong signal: the gateway mostly waits on the model server, so CPU stays low while users wait. Adding gateway pods does not make a single model server faster. Better signals are queue depth, in-flight requests or time to first token, which tools such as KEDA can scale on.
