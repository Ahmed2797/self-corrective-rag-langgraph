from src.logger import logging
import time
from typing import Generator, Dict, Any, List, Optional
import traceback

from langchain_community.callbacks import get_openai_callback
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.database.repository import MessageRepository, ChatRepository
from src.backend_adapter.event_model import PipelineEvent
from src.logger.production_logger import ProductionLogger

logger = logging.getLogger(__name__)


class RAGService:
    """
    Production-grade RAG service with comprehensive error handling.
    """

    def __init__(self, compiled_graph_app, db: Session):
        """
        Initialize RAG service.

        Args:
            compiled_graph_app: Compiled LangGraph application
            db: SQLAlchemy database session

        Raises:
            ValueError: If arguments are invalid
        """
        if not compiled_graph_app:
            raise ValueError("compiled_graph_app is required")
        if not db:
            raise ValueError("Database session is required")

        self.app = compiled_graph_app
        self.db = db

        try:
            self.message_repo = MessageRepository(db)
            self.chat_repo = ChatRepository(db)
            self.production_logger = ProductionLogger(log_file="logs/rag_production.log")
            logger.info("✅ RAGService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAGService: {e}")
            raise

    @staticmethod
    def format_answer_with_citations(
        answer: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format answer with source citations.

        Args:
            answer: Generated answer
            citations: List of citation dicts

        Returns:
            Formatted answer with citations
        """
        try:
            if not answer or not citations:
                return answer or ""

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
        except Exception as e:
            logger.warning(f"Error formatting citations: {e}")
            return answer or ""

    def run_stream(
        self,
        chat_id: str,
        project_id: str,
        user_question: str,
    ) -> Generator[PipelineEvent, None, Optional[Dict[str, Any]]]:
        """
        Execute RAG pipeline with comprehensive error handling.

        Yields:
            PipelineEvent: Real-time execution updates

        Returns:
            Final result payload
        """
        logger.info(f"Starting RAG execution: chat_id={chat_id}, project_id={project_id}")

        # =====================================================================
        # STEP 1: Save user question with error handling
        # =====================================================================
        try:
            self.message_repo.create(
                chat_id=chat_id,
                role="user",
                content=user_question,
            )
            logger.debug(f"✅ Saved user message to DB")
        except SQLAlchemyError as e:
            logger.error(f"❌ Failed to save user message: {e}")
            # Continue anyway - generation is more important than DB persistence
            yield PipelineEvent(
                node="message_save",
                status="failed",
                error=f"Failed to save message: {e}"
            )

        # =====================================================================
        # STEP 2: Load chat history
        # =====================================================================
        try:
            chat_history = self.message_repo.get_by_chat(chat_id)
            formatted_messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in chat_history
            ]
            logger.debug(f"✅ Loaded {len(formatted_messages)} messages from chat history")
        except SQLAlchemyError as e:
            logger.warning(f"⚠️ Failed to load chat history: {e}")
            formatted_messages = []  # Continue with empty history

        # =====================================================================
        # STEP 3: Build initial state
        # =====================================================================
        initial_state = {
            "question": user_question,
            "project_id": project_id,
            "chat_id": chat_id,
            "messages": formatted_messages,
            "retries": 0,
            "rewrite_tries": 0,
        }

        config = {"configurable": {"thread_id": chat_id}}

        final_state: Dict[str, Any] = dict(initial_state)
        last_node: Optional[str] = None
        start_time = time.perf_counter()
        pipeline_error: Optional[str] = None

        # =====================================================================
        # STEP 4: Execute LangGraph stream
        # =====================================================================
        logger.info("Starting LangGraph execution")

        with get_openai_callback() as callback:
            try:
                for output in self.app.stream(
                    initial_state,
                    config=config,
                    stream_mode="updates",
                ):
                    try:
                        if not output or not isinstance(output, dict):
                            continue

                        for node_name, state_update in output.items():
                            if not node_name or node_name in {"__start__", "__end__"}:
                                continue

                            # Yield completion of previous node
                            if last_node and last_node != node_name:
                                yield PipelineEvent(node=last_node, status="completed")

                            # Update final state
                            if isinstance(state_update, dict):
                                final_state.update(state_update)
                                keys_updated = list(state_update.keys())
                            else:
                                keys_updated = []

                            last_node = node_name
                            logger.debug(f"Node {node_name}: updated {keys_updated}")

                            yield PipelineEvent(
                                node=node_name,
                                status="running",
                                metadata={"keys_updated": keys_updated},
                            )

                    except Exception as e:
                        logger.error(f"Error processing stream output: {e}")
                        pipeline_error = str(e)
                        continue

                # Yield completion of last node
                if last_node:
                    yield PipelineEvent(node=last_node, status="completed")

                logger.info("✅ LangGraph execution completed successfully")

            except Exception as e:
                error_message = f"{type(e).__name__}: {str(e)}"
                logger.error(f"❌ Pipeline execution failed: {error_message}\n{traceback.format_exc()}")
                
                pipeline_error = error_message
                
                # Log failure to production logger
                try:
                    metrics = self._run_metrics(callback, start_time, status="failed", error=error_message)
                    self.production_logger.log_query(state=final_state, metrics=metrics)
                except Exception as log_err:
                    logger.warning(f"Failed to log failure: {log_err}")

                # Yield failure event
                yield PipelineEvent(
                    node=last_node or "rag_pipeline",
                    status="failed",
                    error=error_message,
                )
                return None

        # =====================================================================
        # STEP 5: Extract final answer
        # =====================================================================
        fallback_message = (
            "I can only answer based on the uploaded project documents. "
            "No relevant information was found for this question."
        )

        assistant_answer = str(
            final_state.get("answer") or fallback_message
        ).strip()

        if not assistant_answer or \
           assistant_answer.lower() == "no answer found." or \
           "no relevant information" in assistant_answer.lower():
            assistant_answer = fallback_message

        logger.debug(f"Final answer length: {len(assistant_answer)} chars")

        # =====================================================================
        # STEP 6: Extract sources & citations
        # =====================================================================
        try:
            source_names = final_state.get("sources", []) or []
            citations = final_state.get("citations", []) or []
            relevant_docs = final_state.get("relevant_docs", []) or []

            if not source_names:
                for doc in relevant_docs:
                    metadata = getattr(doc, "metadata", {}) or {}
                    source = str(
                        metadata.get("source")
                        or metadata.get("file_name")
                        or metadata.get("name")
                        or "Document"
                    )
                    if source not in source_names:
                        source_names.append(source)

            if not citations:
                for doc in relevant_docs:
                    metadata = getattr(doc, "metadata", {}) or {}
                    page_content = str(getattr(doc, "page_content", ""))
                    citations.append({
                        "source": str(
                            metadata.get("source")
                            or metadata.get("file_name")
                            or "Document"
                        ),
                        "snippet": (
                            page_content[:180] + "..."
                            if len(page_content) > 180
                            else page_content
                        ),
                    })

            logger.debug(f"Extracted {len(source_names)} sources, {len(citations)} citations")

        except Exception as e:
            logger.warning(f"Error extracting citations: {e}")
            source_names = []
            citations = []

        # =====================================================================
        # STEP 7: Build metadata
        # =====================================================================
        docs = final_state.get("docs", []) or []
        metadata = {
            "route": final_state.get("route"),
            "cache_hit": final_state.get("cache_hit", False),
            "retrieved_docs_count": len(final_state.get("relevant_docs", []) or docs),
            "evaluation": final_state.get("evaluation", {}),
            "sources": source_names,
            "citations": citations,
            "confidence_score": final_state.get("confidence_score", 0.0),
            "pipeline_error": pipeline_error,
        }

        # =====================================================================
        # STEP 8: Log execution metrics
        # =====================================================================
        try:
            run_metrics = self._run_metrics(callback, start_time, status="success")
            self.production_logger.log_query(state=final_state, metrics=run_metrics)
            logger.debug(f"✅ Logged metrics: {run_metrics}")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

        # =====================================================================
        # STEP 9: Format answer with citations
        # =====================================================================
        assistant_answer = self.format_answer_with_citations(
            assistant_answer,
            citations,
        )

        # =====================================================================
        # STEP 10: Save assistant message (with retry)
        # =====================================================================
        assistant_msg = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                assistant_msg = self.message_repo.create(
                    chat_id=chat_id,
                    role="assistant",
                    content=assistant_answer,
                    metadata=metadata
                )
                logger.info(f"✅ Saved assistant message: {assistant_msg.id}")
                break

            except SQLAlchemyError as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    logger.error(f"❌ Failed to save assistant message after {max_retries} attempts")
                    # Don't fail the whole operation - just warn user
                    metadata["save_error"] = str(e)

        # =====================================================================
        # STEP 11: Update chat title if new
        # =====================================================================
        try:
            chat = self.chat_repo.get_by_id(chat_id)
            if chat and chat.title in {"New Chat", "New Chat Thread"}:
                short_title = (
                    user_question[:40] + "..."
                    if len(user_question) > 40
                    else user_question
                )
                if hasattr(self.chat_repo, "rename"):
                    self.chat_repo.rename(chat_id, short_title)
                logger.debug(f"Updated chat title: {short_title}")
        except Exception as e:
            logger.warning(f"Failed to update chat title: {e}")

        # =====================================================================
        # STEP 12: Build result payload
        # =====================================================================
        result_payload = {
            "message_id": assistant_msg.id if assistant_msg else None,
            "answer": assistant_answer,
            "metadata": metadata,
        }

        # =====================================================================
        # STEP 13: Yield completion
        # =====================================================================
        yield PipelineEvent(
            node="rag_complete",
            status="completed",
            metadata=result_payload,
        )

        logger.info(f"✅ RAG execution completed: {len(assistant_answer)} chars")
        return result_payload

    @staticmethod
    def _run_metrics(
        callback,
        start_time: float,
        status: str,
        error: str = ""
    ) -> Dict[str, Any]:
        """Calculate execution metrics."""
        return {
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "total_tokens": callback.total_tokens,
            "total_cost": callback.total_cost,
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
        Synchronous wrapper for streaming execution.

        Args:
            chat_id: Active chat ID
            project_id: Active project ID
            user_question: User's question
            event_callback: Optional event handler

        Returns:
            Final result dict
        """
        final_result: Dict[str, Any] = {}

        try:
            for event in self.run_stream(
                chat_id=chat_id,
                project_id=project_id,
                user_question=user_question,
            ):
                if event_callback and callable(event_callback):
                    try:
                        result = event_callback(event)
                        if result is not None:
                            final_result = result
                    except Exception as e:
                        logger.warning(f"Error in event callback: {e}")

                if event.node == "rag_complete" and event.status == "completed":
                    final_result = event.metadata or {}

            return final_result

        except Exception as e:
            logger.error(f"Error in run(): {e}")
            return {"error": str(e), "answer": "An error occurred during processing."}