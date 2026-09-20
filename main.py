from src.pipeline import pipeline
import time
from langchain_community.callbacks import get_openai_callback
from src.logger.production_logger import ProductionLogger


production_logger = ProductionLogger(
    log_file="logs/rag_production.log"
)

app = pipeline()

def run_rag(question: str):

    initial_state = {
        "question": question,
        "retrieval_query": "",
        "optimized_query": "",
        "keywords": [],
        "entities": [],
        "rewrite_tries": 2,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "",
        "evidence": [],
        "retries":1,
        "isuse": "",
        "evaluation": {},
        "cache_hit": False,
        "need_retrieval": None,
    }

    start_time = time.perf_counter()

    with get_openai_callback() as callback:

        final_state = app.invoke(initial_state)

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        metrics = {
            "latency_ms": latency_ms,
            "tokens_used": callback.total_tokens,
        }

    production_logger.log_query(
        state=final_state,
        metrics=metrics
    )

    return final_state

if __name__ == "__main__":
    final_state = run_rag(question="What department does Tanvir Ahmed belong to??")
    print("Final State", final_state)
    print("*"*50)
    print("\nAnswer:", final_state["answer"])
    # You can now use the compiled LangGraph application for further processing or querying
