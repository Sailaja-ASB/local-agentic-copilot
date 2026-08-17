from pathlib import Path  # Lets us save the generated diagram inside the docs folder.

import matplotlib.pyplot as plt  # Creates the architecture diagram.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # Draws component boxes and arrows.


OUTPUT_PATH = Path("docs/architecture.png")  # Final architecture image used in GitHub README.


def add_box(ax, x, y, width, height, text):
    """Draw one rounded architecture component."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        fill=False,
    )  # Creates a clean rounded rectangle without forcing a specific color.

    ax.add_patch(box)  # Adds the component box to the diagram.

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )  # Places the component name in the center of the box.


def add_arrow(ax, start, end):
    """Draw a directional connection between architecture components."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.3,
    )  # Creates a directional flow arrow.

    ax.add_patch(arrow)  # Adds the arrow to the architecture diagram.


fig, ax = plt.subplots(figsize=(12, 16))  # Creates one large vertical diagram canvas.

ax.set_xlim(0, 12)  # Defines horizontal drawing space.
ax.set_ylim(0, 18)  # Defines vertical drawing space.
ax.axis("off")  # Removes normal chart axes because this is an architecture diagram.


# ---------- DATA / RAG PIPELINE ----------

add_box(ax, 3.5, 16.2, 5, 0.8, "User Documents\nTXT • Markdown • PDF")  # Input knowledge sources.

add_box(ax, 3.5, 14.8, 5, 0.8, "Page-Aware Document Ingestion\nChunking + Source/Page Metadata")  # Document processing layer.

add_box(ax, 3.5, 13.4, 5, 0.8, "Sentence-Transformer Embeddings\n+ ChromaDB Vector Index")  # Semantic storage layer.

add_arrow(ax, (6, 16.2), (6, 15.6))  # Documents → ingestion.
add_arrow(ax, (6, 14.8), (6, 14.2))  # Ingestion → vector index.


# ---------- RETRIEVAL ----------

add_box(ax, 0.7, 11.7, 4.4, 0.9, "Dense Semantic Retrieval")  # Meaning-based search.

add_box(ax, 6.9, 11.7, 4.4, 0.9, "BM25 Lexical Retrieval")  # Exact keyword search.

add_arrow(ax, (5.3, 13.4), (3.2, 12.6))  # Vector index → dense search.
add_arrow(ax, (6.7, 13.4), (8.9, 12.6))  # Documents/query → lexical search.

add_box(ax, 3.5, 10.1, 5, 0.9, "Hybrid Candidate Fusion\nReciprocal Rank Fusion")  # Combines dense + BM25 rankings.

add_arrow(ax, (3, 11.7), (5, 11.0))  # Dense → hybrid fusion.
add_arrow(ax, (9, 11.7), (7, 11.0))  # BM25 → hybrid fusion.

add_box(ax, 3.5, 8.6, 5, 0.9, "Cross-Encoder Reranking\n+ Source Metadata")  # Improves final evidence ordering.

add_arrow(ax, (6, 10.1), (6, 9.5))  # Hybrid candidates → reranker.


# ---------- AGENTIC LAYER ----------

add_box(ax, 3.5, 7.1, 5, 0.9, "Researcher Agent\nGrounded Evidence + Requirements")  # Research agent.

add_arrow(ax, (6, 8.6), (6, 8.0))  # Reranked evidence → Researcher.

add_box(ax, 3.5, 5.6, 5, 0.9, "Coder Agent\nTechnical Solution Generation")  # Solution-producing agent.

add_arrow(ax, (6, 7.1), (6, 6.5))  # Researcher → Coder.

add_box(ax, 3.5, 4.1, 5, 0.9, "Deterministic Guardrails\nStack + Citation Validation")  # Rule-based validation.

add_arrow(ax, (6, 5.6), (6, 5.0))  # Coder → guardrails.

add_box(ax, 3.5, 2.6, 5, 0.9, "LLM Reviewer Agent\nAPPROVE / REVISE")  # Semantic reviewer.

add_arrow(ax, (6, 4.1), (6, 3.5))  # Guardrail pass → reviewer.


# ---------- REVISION LOOP ----------

add_arrow(
    ax,
    (3.5, 3.0),
    (2.0, 5.9),
)  # Reviewer revision path back toward Coder.

ax.text(
    1.15,
    4.5,
    "REVISE",
    fontsize=9,
    rotation=90,
    va="center",
)  # Labels the self-correction path.


# ---------- FINAL OUTPUT ----------

add_box(ax, 3.5, 1.0, 5, 0.9, "Grounded Final Output\nAnswer + Source/Page Citations")  # Final approved result.

add_arrow(ax, (6, 2.6), (6, 1.9))  # Reviewer approval → final answer.


ax.text(
    6,
    17.5,
    "Local Agentic Copilot — V1 Architecture",
    ha="center",
    fontsize=18,
    fontweight="bold",
)  # Adds the architecture title.


plt.tight_layout()  # Prevents diagram elements from being clipped.

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)  # Saves a high-resolution PNG suitable for GitHub.

print(f"Architecture diagram saved to: {OUTPUT_PATH}")  # Confirms where the file was generated.