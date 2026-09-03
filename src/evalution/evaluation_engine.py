"""
=============================================================
Evaluation Engine — Orchestrates all evaluation metrics
=============================================================
Runs hallucination detection, groundedness scoring, and
sycophancy detection, then produces a combined evaluation.
"""

from typing import Dict, List

from src.evalution.hallucination_detector import detect_hallucination
from src.evalution.groundedness_scorer import score_groundedness
from src.evalution.sycophancy_detector import detect_sycophancy


# =============================================================
# Configuration
# =============================================================

HALLUCINATION_WEIGHT = 0.40
GROUNDEDNESS_WEIGHT = 0.40
SYCOPHANCY_WEIGHT = 0.20

RELIABILITY_THRESHOLD = 0.75
MAX_HALLUCINATION = 0.20
MIN_GROUNDEDNESS = 0.70
MAX_SYCOPHANCY = 0.30


# =============================================================
# Evaluation Engine
# =============================================================

def evaluate_answer(
    question: str,
    answer: str,
    context_chunks: List[Dict],
) -> Dict:
    """
    Run the full evaluation suite on a generated answer.

    Evaluates:
        - Hallucination
        - Groundedness
        - Sycophancy

    Then computes:
        - Overall quality
        - Reliability

    Args:
        question: User question.
        answer: Generated AI answer.
        context_chunks: Retrieved context chunks.

    Returns:
        Combined evaluation dictionary.
    """

    # ---------------------------------------------------------
    # Build context
    # ---------------------------------------------------------

    context_str = "\n\n".join(
        f"[Source: {chunk.get('source', 'unknown')}]\n"
        f"{chunk.get('content', '')}"
        for chunk in context_chunks
    )

    # ---------------------------------------------------------
    # Run evaluators
    # ---------------------------------------------------------

    print("\n========== EVALUATION ==========")

    print("\n[1] Hallucination Evaluation...")
    hallucination = detect_hallucination(
        answer=answer,
        context=context_str,
    )

    print(
        "Hallucination Score:",
        hallucination.get("hallucination_score")
    )

    print("\n[2] Groundedness Evaluation...")
    groundedness = score_groundedness(
        question=question,
        answer=answer,
        context=context_str,
    )

    print(
        "Groundedness Score:",
        groundedness.get("groundedness_score")
    )

    print("\n[3] Sycophancy Evaluation...")
    sycophancy = detect_sycophancy(
        question=question,
        answer=answer,
        context=context_str,
    )

    print(
        "Sycophancy Score:",
        sycophancy.get("sycophancy_score")
    )

    # ---------------------------------------------------------
    # Extract scores
    # ---------------------------------------------------------

    hall_score = hallucination.get("hallucination_score")
    ground_score = groundedness.get("groundedness_score")
    syco_score = sycophancy.get("sycophancy_score")

    # ---------------------------------------------------------
    # Check evaluation failures
    # ---------------------------------------------------------

    if (
        hall_score is None
        or ground_score is None
        or syco_score is None
    ):

        print("\n⚠️ Evaluation failed.")

        return {
            "hallucination": hallucination,
            "groundedness": groundedness,
            "sycophancy": sycophancy,
            "overall_quality": None,
            "is_reliable": False,
            "evaluation_status": "failed",
        }

    # ---------------------------------------------------------
    # Calculate overall quality
    # ---------------------------------------------------------

    overall_quality = (
        ground_score * GROUNDEDNESS_WEIGHT
        + (1.0 - hall_score) * HALLUCINATION_WEIGHT
        + (1.0 - syco_score) * SYCOPHANCY_WEIGHT
    )

    overall_quality = round(overall_quality, 3)

    # ---------------------------------------------------------
    # Reliability
    # ---------------------------------------------------------

    is_reliable = (
        overall_quality >= RELIABILITY_THRESHOLD
        and ground_score >= MIN_GROUNDEDNESS
        and hall_score <= MAX_HALLUCINATION
        and syco_score <= MAX_SYCOPHANCY
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    evaluation = {
        "hallucination": hallucination,
        "groundedness": groundedness,
        "sycophancy": sycophancy,
        "overall_quality": overall_quality,
        "is_reliable": is_reliable,
        "evaluation_status": "success",
    }

    print("\n========== FINAL EVALUATION ==========")
    print("Overall Quality:", overall_quality)
    print("Reliable:", is_reliable)
    print("======================================")

    return evaluation