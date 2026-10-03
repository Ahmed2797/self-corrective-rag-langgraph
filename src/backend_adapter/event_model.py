# src/adapters/rag_service.py

import time
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from langchain_community.callbacks import get_openai_callback
from src.pipeline import pipeline
from src.logger.production_logger import ProductionLogger

production_logger = ProductionLogger(log_file="logs/rag_production.log")

class PipelineEvent:
    def __init__(self, node: str, status: str, metadata: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.node = node
        self.status = status  # 'running', 'completed', 'failed'
        self.metadata = metadata or {}
        self.error = error

async def run_rag_stream(
    question: str, 
    thread_id: str, 
    chat_history: Optional[list] = None,
    project_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Executes RAG graph via astream, yielding status events and returning final state & metrics.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Build the graph only when this helper is actually called. Importing this
    # module must not create a test pipeline or require API credentials.
    app = pipeline(project_id or thread_id)

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
        "retries": 1,
        "isuse": "",
        "evaluation": {},
        "cache_hit": False,
        "need_retrieval": None,
        "messages": chat_history or []
    }

    start_time = time.perf_counter()
    last_node = None
    final_state = {}

    with get_openai_callback() as callback:
        # Stream graph updates node-by-node
        async for output in app.astream(initial_state, config=config, stream_mode="updates"):
            if not output or not isinstance(output, dict):
                continue

            for node_name, state_update in output.items():
                if not node_name or node_name in ["__start__", "__end__"]:
                    continue

                # Notify completion of previous node
                if last_node and last_node != node_name:
                    yield {"type": "event", "data": PipelineEvent(node=last_node, status="completed")}

                # Update current state reference
                if isinstance(state_update, dict):
                    final_state.update(state_update)
                    keys_updated = list(state_update.keys())
                else:
                    keys_updated = []

                last_node = node_name
                yield {
                    "type": "event", 
                    "data": PipelineEvent(node=node_name, status="running", metadata={"keys_updated": keys_updated})
                }

        # Flush final node completion
        if last_node:
            yield {"type": "event", "data": PipelineEvent(node=last_node, status="completed")}

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metrics = {
            "Prompt tokens": callback.prompt_tokens,
            "Completion tokens": callback.completion_tokens,
            "latency_ms": latency_ms,
            "tokens_used": callback.total_tokens,
            "query_cost": callback.total_cost,
        }

    # Log metrics to production log
    production_logger.log_query(state=final_state, metrics=metrics)

    # Yield final response payload
    yield {
        "type": "result",
        "final_state": final_state,
        "metrics": metrics
    }
