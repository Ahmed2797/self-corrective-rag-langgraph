from typing import List, TypedDict, Literal,Dict
from langchain_core.documents import Document

from pydantic import BaseModel, Field

# ============================================================
# Type Aliases
# ============================================================

SUPPORT_STATUS = Literal[
    "fully_supported",
    "partially_supported",
    "no_support",
]

USEFULNESS_STATUS = Literal[
    "useful",
    "not_useful",
]

# --------------------------
# Evaluation Result
# --------------------------

class EvaluationResult(TypedDict):
    """
    Final evaluation result produced by the evaluation engine.
    """

    hallucination: Dict
    groundedness: Dict
    sycophancy: Dict

    overall_quality: float | None
    is_reliable: bool
    evaluation_status: Literal["success", "failed"]

# ============================================================
# Graph State
# ============================================================
class State(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    """

    # ========================================================
    # User Question
    # ========================================================

    original_question: str
    question: str

    # ========================================================
    # Retrieval Decision
    # ========================================================

    need_retrieval: bool
    retrieval_query: str

    # ========================================================
    # Retrieved Documents
    # ========================================================

    docs: List[Document]
    relevant_docs: List[Document]

    # ========================================================
    # Generated Answer
    # ========================================================

    context: str
    answer: str

    # ========================================================
    # Answer Support Verification — IsSUP
    # ========================================================

    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support",
    ]

    evidence: List[str]
    retries: int

    # ========================================================
    # Answer Usefulness Verification — IsUSE
    # ========================================================

    isuse: Literal[
        "useful",
        "not_useful",
    ]

    use_reason: str

    # ========================================================
    # Query Rewriting
    # ========================================================

    rewrite_tries: int

    # ========================================================
    # Final Answer Evaluation
    # ========================================================

    evaluation: EvaluationResult

    # ========================================================
    # Semantic Cache
    # ========================================================

    cache_hit: bool
    cache_similarity: float



# ============================================================
# 1. Retrieval Decision
# ============================================================

class RetrieveDecision(BaseModel):
    """
    Determines whether document retrieval is required.
    """

    should_retrieve: bool = Field(
        ...,
        description=(
            "True if external/internal documents are required "
            "to answer the question reliably, otherwise False."
        ),
    )


# ============================================================
# 2. Document Relevance Decision
# ============================================================

class RelevanceDecision(BaseModel):
    """
    Determines whether a retrieved document is relevant
    to the user's question.
    """

    is_relevant: bool = Field(
        ...,
        description=(
            "True ONLY if the document contains information "
            "that can directly help answer the question."
        ),
    )


# ============================================================
# 3. Answer Support / IsSUP Decision
# ============================================================

class IsSUPDecision(BaseModel):
    """
    Verifies whether the generated answer is supported
    by the retrieved context.
    """

    issup: SUPPORT_STATUS = Field(
        ...,
        description=(
            "Degree to which the answer is supported by "
            "the provided context."
        ),
    )

    evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Short evidence quotes from the context "
            "supporting the generated answer."
        ),
    )


# ============================================================
# 4. Answer Usefulness / IsUSE Decision
# ============================================================

class IsUSEDecision(BaseModel):
    """
    Determines whether the generated answer is useful
    for the user's original question.
    """

    isuse: USEFULNESS_STATUS = Field(
        ...,
        description="Whether the generated answer is useful to the user.",
    )

    use_reason: str = Field(
        ...,
        description="Short one-line explanation of the usefulness decision.",
    )


# ============================================================
# 5. Query Rewrite Decision
# ============================================================

class RewriteDecision(BaseModel):
    """
    Generates a better retrieval query when the current
    retrieval result is insufficient.
    """

    retrieval_query: str = Field(
        ...,
        description=(
            "Rewritten query optimized for semantic/vector "
            "retrieval against the internal documents."
        ),
    )




# ============================================================
# 7. Semantic Cache Metadata
# ============================================================

class CacheMetadata(TypedDict, total=False):
    """
    Metadata stored with a semantic-cache entry.
    """

    question: str
    answer: str
    evaluation: str
    cache_version: str