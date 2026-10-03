"""
Reusable UI Components
======================
Streamlit renderers for a self-corrective RAG workspace dashboard.
"""

import streamlit as st
from typing import Dict, Any, List, Callable, Iterable


def _safe_score(value: Any) -> str:
    """Render floats and numbers consistently in UI badges."""
    try:
        if value is None:
            return "N/A"
        numeric = float(value)
        return f"{numeric:.2f}"
    except (TypeError, ValueError):
        return str(value or "N/A")


def render_metric_card(title: str, value: Any, accent: str = "#8b5cf6"):
    """Render a compact metric card for the dashboard."""
    st.markdown(
        f"""
        <div class="metric-pill" style="border-left: 4px solid {accent};">
            <span class="label">{title}</span>
            <span class="value">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_status(events: Dict[str, str]):
    """Render a modern pipeline progress card for the current LangGraph execution."""
    if not events:
        return

    st.markdown("#### ⚙️ Real-time Execution Pipeline")
    cols = st.columns(min(len(events), 4))

    for idx, (node_name, status) in enumerate(events.items()):
        col = cols[idx % 4]
        normalized_status = str(status).lower()
        status_class = f"status-{normalized_status if normalized_status in ['running', 'completed', 'failed'] else 'running'}"
        icon = "⏳" if normalized_status == "running" else "✅" if normalized_status == "completed" else "❌"

        with col:
            col.markdown(
                f"""
                <div class="status-card {status_class}">
                    <div class="node-label">{icon} Node</div>
                    <div class="node-name">{node_name}</div>
                    <div class="node-label">{normalized_status.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_message_metadata(metadata: Dict[str, Any]):
    """Render a compact, readable metadata panel for a generated assistant answer."""
    if not metadata:
        return

    route = metadata.get("route", "N/A")
    cache_hit = "Yes" if metadata.get("cache_hit") else "No"
    docs_count = metadata.get("retrieved_docs_count", 0)
    confidence = metadata.get("confidence_score")
    evaluation = metadata.get("evaluation") or {}

    st.markdown(
        f"""
        <div class="meta-panel">
            <div>
                <span class="meta-badge">Route: <b>{route}</b></span>
                <span class="meta-badge">Cache: <b>{cache_hit}</b></span>
                <span class="meta-badge">Retrieved Docs: <b>{docs_count}</b></span>
                <span class="meta-badge">Confidence: <b>{_safe_score(confidence)}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Execution Details & Sources"):
        sources = metadata.get("sources", []) or []
        citations = metadata.get("citations", []) or []
        if sources:
            st.markdown("**Sources Used:**")
            for idx, src in enumerate(sources, 1):
                if isinstance(src, dict):
                    label = src.get("title") or src.get("source") or src.get("name") or f"Source {idx}"
                    st.markdown(f"- **[{idx}]** {label}")
                else:
                    st.markdown(f"- **[{idx}]** {src}")
        else:
            st.caption("No external sources retrieved for this turn.")

        if citations:
            st.markdown("**Exact Snippets:**")
            for idx, citation in enumerate(citations[:3], 1):
                source_name = citation.get("source") if isinstance(citation, dict) else "Document"
                snippet = citation.get("snippet") if isinstance(citation, dict) else str(citation)
                st.markdown(f"- **{source_name}**: \"{snippet}\"")

        if evaluation:
            st.markdown("**Evaluation Summary:**")
            score_cols = st.columns(min(len(evaluation), 4))
            for idx, (key, value) in enumerate(evaluation.items()):
                if idx >= 4:
                    break
                with score_cols[idx % 4]:
                    render_metric_card(str(key).replace("_", " ").title(), _safe_score(value), accent="#22d3ee")

            with st.container():
                st.json(evaluation)


def render_file_management_card(files: List[Any], on_delete_callback: Callable[[str], None]):
    """Render uploaded project files in a compact, clear management matrix."""
    if not files:
        st.info("No documents uploaded for this project yet.")
        return

    for file_rec in files:
        col1, col2, col3 = st.columns([5.5, 2.2, 1.3])
        with col1:
            st.markdown(
                f"""
                <div class="doc-card">
                    <div style="font-size: 1rem; font-weight: 700;">📄 {file_rec.display_name}</div>
                    <div style="color: #94a3b8; margin-top: 0.2rem;">{file_rec.file_type}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.caption(f"{round(file_rec.size / 1024, 1)} KB")
        with col3:
            if st.button("🗑️", key=f"del_file_{file_rec.id}", use_container_width=True):
                on_delete_callback(file_rec.id)
                st.rerun()


def render_project_summary(project_name: str, description: str = "", stats: Dict[str, Any] | None = None):
    """Render an executive summary card at the top of a project page."""
    st.markdown(f"## {project_name}")
    if description:
        st.caption(description)

    if not stats:
        stats = {}

    metric_cols = st.columns(4)
    default_values = {
        "Chats": stats.get("chats", 0),
        "Documents": stats.get("documents", 0),
        "Retrievals": stats.get("retrievals", 0),
        "Avg. Quality": stats.get("avg_quality", "--"),
    }

    for idx, (label, value) in enumerate(default_values.items()):
        with metric_cols[idx]:
            render_metric_card(label, value, accent=["#8b5cf6", "#22d3ee", "#34d399", "#fbbf24"][idx % 4])
