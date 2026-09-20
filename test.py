import pytest

from src.pipeline import pipeline


class TestSelfRAG:

    @pytest.fixture
    def app(self):
        """Create a fresh Self-RAG graph for testing."""
        return pipeline()

    # =========================================================
    # 1. Graph Creation Test
    # =========================================================

    def test_graph_creation(self, app):
        """Test that the Self-RAG graph is created successfully."""

        assert app is not None

    # =========================================================
    # 2. Retrieval Decision Test
    # =========================================================

    def test_retrieval_decision(self, app):
        """Test adaptive retrieval decision."""

        result = app.invoke({
            "question": "What is 2 + 2?"
        })

        assert "need_retrieval" in result
        assert result["need_retrieval"] is False

    # =========================================================
    # 3. Retrieval Required Test
    # =========================================================

    def test_retrieval_required(self, app):
        """Test that a knowledge-base question requires retrieval."""

        result = app.invoke({
            "question": "Who is Tanvir Ahmed?"
        })

        assert "need_retrieval" in result
        assert result["need_retrieval"] is True

    # =========================================================
    # 4. Cache Test
    # =========================================================

    def test_cache_hit(self, app):
        """Test semantic cache using the same question twice."""

        question = "Who is Tanvir Ahmed?"

        # First request
        result1 = app.invoke({
            "question": question
        })

        assert result1 is not None

        # Second request
        result2 = app.invoke({
            "question": question
        })

        assert "cache_hit" in result2
        assert result2["cache_hit"] is True

    # =========================================================
    # 5. Answer Generation Test
    # =========================================================

    def test_answer_generation(self, app):
        """Test that the graph generates an answer."""

        result = app.invoke({
            "question": "Who is Tanvir Ahmed?"
        })

        assert "answer" in result
        assert result["answer"] is not None
        assert len(result["answer"].strip()) > 0

    # =========================================================
    # 6. Evaluation Structure Test
    # =========================================================

    def test_evaluation_metrics(self, app):
        """Test evaluation engine output."""

        result = app.invoke({
            "question": "Who is Tanvir Ahmed?"
        })

        assert "evaluation" in result

        evaluation = result["evaluation"]

        assert "hallucination" in evaluation
        assert "groundedness" in evaluation
        assert "sycophancy" in evaluation

        assert "overall_quality" in evaluation
        assert "is_reliable" in evaluation
        assert "evaluation_status" in evaluation

    # =========================================================
    # 7. Evaluation Score Range Test
    # =========================================================

    def test_evaluation_score_ranges(self, app):
        """Test that evaluation scores are between 0 and 1."""

        result = app.invoke({
            "question": "Who is Tanvir Ahmed?"
        })

        evaluation = result["evaluation"]

        hallucination = (
            evaluation["hallucination"]["hallucination_score"]
        )

        groundedness = (
            evaluation["groundedness"]["groundedness_score"]
        )

        sycophancy = (
            evaluation["sycophancy"]["sycophancy_score"]
        )

        assert 0.0 <= hallucination <= 1.0
        assert 0.0 <= groundedness <= 1.0
        assert 0.0 <= sycophancy <= 1.0

    # =========================================================
    # 8. Evaluation Status Test
    # =========================================================

    def test_evaluation_success(self, app):
        """Test that evaluation completes successfully."""

        result = app.invoke({
            "question": "Who is Tanvir Ahmed?"
        })

        evaluation = result["evaluation"]

        assert evaluation["evaluation_status"] == "success"

    # =========================================================
    # 9. Overall Quality Test
    # =========================================================

    def test_overall_quality(self, app):
        """Test overall evaluation quality."""

        result = app.invoke({
            "question": "Who is Tanvir Ahmed?"
        })

        evaluation = result["evaluation"]

        overall_quality = evaluation["overall_quality"]

        assert overall_quality is not None
        assert 0.0 <= overall_quality <= 1.0

    # =========================================================
    # 10. Complete RAG Test
    # =========================================================

    def test_complete_rag_workflow(self, app):
        """Test the complete Self-RAG workflow."""

        result = app.invoke({
            "question": "Who is Tanvir Ahmed?"
        })

        # Basic state
        assert result is not None

        # Answer
        assert "answer" in result
        assert result["answer"]

        # Retrieval
        assert "docs" in result
        assert "relevant_docs" in result

        # Evaluation
        assert "evaluation" in result

        # Evaluation status
        assert (
            result["evaluation"]["evaluation_status"]
            == "success"
        )

test_rag = TestSelfRAG()
app = pipeline()
test_rag.test_answer_generation(app)