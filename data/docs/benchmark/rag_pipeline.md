# The RAG Pipeline

Retrieval-augmented generation answers questions from your own documents instead of the model's memory. The pipeline is: receive the question, retrieve relevant chunks, build a prompt that contains the question plus the retrieved evidence, ask the language model to answer only from that evidence, and return the answer with citations.

Grounding the model in retrieved text reduces hallucination and lets answers stay current without retraining. When retrieval returns nothing relevant, a well-behaved system should say it does not know instead of guessing.
