"""
RAG Service Adapter
===================
Bridges Streamlit UI with the existing LangGraph execution engine
and SQLite persistence layer.
"""

import logging
import time
from typing import Generator, Dict, Any, List, Optional

from langchain_community.callbacks import get_openai_callback
from sqlalchemy.orm import Session

from src.database.repository import MessageRepository, ChatRepository
from src.backend_adapter.event_model import PipelineEvent
from src.logger.production_logger import ProductionLogger


class RAGService:

    def __init__(
        self,
        compiled_graph_app,
        db: Session,
    ):
        """
        Initialize RAG service.

        Args:
            compiled_graph_app: Compiled LangGraph application.
            db: SQLAlchemy database session.
        """
        self.app = compiled_graph_app
        self.db = db

        self.message_repo = MessageRepository(db)
        self.chat_repo = ChatRepository(db)
        self.production_logger = ProductionLogger(log_file="logs/rag_production.log")

    @staticmethod
    def format_answer_with_citations(
        answer: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format the final assistant answer with source citations.

        Args:
            answer: Generated assistant answer.
            citations: List of citation dictionaries.

        Returns:
            Formatted answer with citations.
        """
        if not answer or not citations:
            return answer

        cleaned_answer = str(answer).strip()
        citation_lines = []

        for idx, citation in enumerate(citations[:5], 1):
            if not isinstance(citation, dict):
                continue

            source_name = str(citation.get("source") or "Document")
            snippet = str(citation.get("snippet") or "").strip()

            if snippet:
                citation_lines.append(f'[{idx}] {source_name}: "{snippet}"')

        if not citation_lines:
            return cleaned_answer

        return (
            f"{cleaned_answer}\n\n"
            f"**Sources**\n"
            + "\n".join(citation_lines)
        )

    def run_stream(
        self,
        chat_id: str,
        project_id: str,
        user_question: str,
    ) -> Generator[PipelineEvent, None, Optional[Dict[str, Any]]]:
        """
        Execute the RAG pipeline synchronously for Streamlit compatibility.

        Yields:
            PipelineEvent: Real-time pipeline execution updates.
            
        Returns:
            Dict[str, Any]: Final answer payload upon completion.
        """
        # 1. Save user question
        self.message_repo.create(
            chat_id=chat_id,
            role="user",
            content=user_question,
        )

        # 2. Load chat history
        chat_history = self.message_repo.get_by_chat(chat_id)
        formatted_messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in chat_history
        ]

        # 3. Build LangGraph initial state
        initial_state = {
            "question": user_question,
            "project_id": project_id,
            "chat_id": chat_id,
            "messages": formatted_messages,
            "retries": 0,
            "rewrite_tries": 0,
        }

        config = {
            "configurable": {
                "thread_id": chat_id,
            }
        }

        # LangGraph's "updates" stream contains node deltas, so retain the
        # initial query fields as well as each node's updates for logging.
        final_state: Dict[str, Any] = dict(initial_state)
        last_node: Optional[str] = None
        start_time = time.perf_counter()

        # 4. Execute LangGraph stream
        with get_openai_callback() as callback:
            try:
                for output in self.app.stream(
                    initial_state,
                    config=config,
                    stream_mode="updates",
                ):
                    if not output or not isinstance(output, dict):
                        continue

                    for node_name, state_update in output.items():
                        # Ignore internal nodes
                        if not node_name or node_name in {"__start__", "__end__"}:
                            continue

                        # Complete previous node
                        if last_node and last_node != node_name:
                            yield PipelineEvent(node=last_node, status="completed")

                        # Extract state updates
                        if isinstance(state_update, dict):
                            final_state.update(state_update)
                            keys_updated = list(state_update.keys())
                        else:
                            keys_updated = []

                        last_node = node_name
                        yield PipelineEvent(
                            node=node_name,
                            status="running",
                            metadata={"keys_updated": keys_updated},
                        )

                if last_node:
                    yield PipelineEvent(node=last_node, status="completed")

            except Exception as e:
                error_message = str(e)
                logging.exception("Error during LangGraph execution")
                self.production_logger.log_query(
                    state=final_state,
                    metrics=self._run_metrics(callback, start_time, status="failed", error=error_message),
                )
                yield PipelineEvent(
                    node=last_node or "rag_pipeline",
                    status="failed",
                    error=error_message,
                )
                return None

        run_metrics = self._run_metrics(callback, start_time, status="success")

        # 5. Extract final answer
        fallback_message = (
            "I can only answer based on the uploaded "
            "project documents. No relevant information "
            "was found for this question."
        )

        assistant_answer = str(
            final_state.get("answer", fallback_message) or fallback_message
        ).strip()

        if assistant_answer.lower() in {"no answer found.", fallback_message.lower()} or \
           "no relevant information was found" in assistant_answer.lower():
            assistant_answer = fallback_message

        # 6. Extract sources & citations
        source_names = final_state.get("sources", []) or []
        citations = final_state.get("citations", []) or []
        relevant_docs = final_state.get("relevant_docs", []) or []

        if not source_names:
            for doc in relevant_docs:
                metadata = getattr(doc, "metadata", {}) or {}
                source_names.append(
                    str(
                        metadata.get("source")
                        or metadata.get("file_name")
                        or metadata.get("name")
                        or "Document"
                    )
                )

        if not citations:
            for doc in relevant_docs:
                metadata = getattr(doc, "metadata", {}) or {}
                page_content = str(getattr(doc, "page_content", ""))
                citations.append(
                    {
                        "source": str(
                            metadata.get("source")
                            or metadata.get("file_name")
                            or metadata.get("name")
                            or "Document"
                        ),
                        "snippet": (
                            page_content[:180] + "..."
                            if len(page_content) > 180
                            else page_content
                        ),
                    }
                )

        # 7. Build metadata
        docs = final_state.get("docs", []) or []
        metadata = {
            "route": final_state.get("route"),
            "cache_hit": final_state.get("cache_hit", False),
            "retrieved_docs_count": len(relevant_docs or docs),
            "evaluation": final_state.get("evaluation", {}),
            "sources": source_names,
            "citations": citations,
            "confidence_score": final_state.get("confidence_score", 0.0),
        }

        # Log every completed Streamlit graph run, including token and latency
        # metrics, using the same production JSONL logger as main.py.
        self.production_logger.log_query(state=final_state, metrics=run_metrics)

        # 8. Format answer with source block
        assistant_answer = self.format_answer_with_citations(
            assistant_answer,
            citations,
        )

        # 9. Save assistant message to Database
        assistant_msg = self.message_repo.create(
            chat_id=chat_id,
            role="assistant",
            content=assistant_answer,
            metadata=metadata,
        )

        # 10. Update chat thread title if new
        chat = self.chat_repo.get_by_id(chat_id)
        if chat and chat.title in {"New Chat", "New Chat Thread"}:
            short_title = (
                user_question[:30] + "..."
                if len(user_question) > 30
                else user_question
            )
            if hasattr(self.chat_repo, "update_title"):
                self.chat_repo.update_title(chat_id, short_title)
            elif hasattr(self.chat_repo, "rename"):
                self.chat_repo.rename(chat_id, short_title)

        result_payload = {
            "message_id": assistant_msg.id,
            "answer": assistant_answer,
            "metadata": metadata,
        }

        # 11. Yield completion event containing final payload
        yield PipelineEvent(
            node="rag_complete",
            status="completed",
            metadata=result_payload,
        )

        return result_payload

    @staticmethod
    def _run_metrics(callback, start_time: float, status: str, error: str = "") -> Dict[str, Any]:
        return {
            "Prompt tokens": callback.prompt_tokens,
            "Completion tokens": callback.completion_tokens,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "tokens_used": callback.total_tokens,
            "query_cost": callback.total_cost,
            "status": status,
            "error": error,
        }

    def run(
        self,
        chat_id: str,
        project_id: str,
        user_question: str,
        event_callback=None,
    ) -> Dict[str, Any]:
        """
        Convenience synchronous wrapper.

        Args:
            chat_id: Active chat ID.
            project_id: Active project ID.
            user_question: User question.
            event_callback: Optional callback for pipeline events.

        Returns:
            Dict[str, Any]: Final answer and metadata.
        """
        final_result: Dict[str, Any] = {}

        for event in self.run_stream(
            chat_id=chat_id,
            project_id=project_id,
            user_question=user_question,
        ):
            if event_callback and callable(event_callback):
                result = event_callback(event)
                if result is not None:
                    final_result = result

            if event.node == "rag_complete" and event.status == "completed":
                final_result = event.metadata or {}

        return final_result
    
