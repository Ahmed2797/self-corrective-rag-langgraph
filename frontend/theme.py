"""
UI Theme & Custom Styling Manager
=================================
Injects custom CSS to style the self-corrective RAG workspace with a modern AI dashboard aesthetic.
"""

import streamlit as st


def apply_custom_css():
    """Inject custom CSS rules into the Streamlit app session."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --bg2: #0b1729;
            --panel: rgba(15, 23, 42, 0.9);
            --panel-strong: #101b2d;
            --panel-soft: #12233d;
            --line: rgba(148, 163, 184, 0.18);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --primary: #8b5cf6;
            --primary-soft: rgba(139, 92, 246, 0.18);
            --cyan: #22d3ee;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #f87171;
            --shadow: 0 10px 30px rgba(15, 23, 42, 0.45);
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: radial-gradient(circle at top left, rgba(139, 92, 246, 0.18), transparent 28%),
                        radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.12), transparent 26%),
                        var(--bg);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(9, 16, 27, 0.98), rgba(13, 22, 36, 0.98));
            border-right: 1px solid var(--line);
            box-shadow: inset -1px 0 0 rgba(148, 163, 184, 0.08);
        }

        .stTabs [role="tablist"] {
            gap: 0.6rem;
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.4rem;
            margin-bottom: 1rem;
        }

        .stTabs [role="tab"] {
            border-radius: 10px;
            padding: 0.6rem 1rem;
            color: var(--muted);
            font-weight: 600;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.24), rgba(34, 211, 238, 0.12));
            color: var(--text);
            border: 1px solid rgba(139, 92, 246, 0.5);
        }

        .project-card,
        .doc-card,
        .status-card,
        .meta-panel,
        .metric-pill {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(10, 18, 30, 0.96));
            box-shadow: var(--shadow);
        }

        .project-card {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            margin-bottom: 0.7rem;
            transition: all 0.2s ease;
        }

        .project-card:hover {
            border-color: rgba(139, 92, 246, 0.7);
            transform: translateY(-1px);
        }

        .status-card {
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
            margin-bottom: 0.6rem;
            min-height: 72px;
        }

        .status-running {
            border-color: rgba(59, 130, 246, 0.7);
            background: linear-gradient(180deg, rgba(30, 64, 175, 0.2), rgba(15, 23, 42, 0.94));
        }

        .status-completed {
            border-color: rgba(16, 185, 129, 0.7);
            background: linear-gradient(180deg, rgba(6, 78, 59, 0.25), rgba(15, 23, 42, 0.96));
        }

        .status-failed {
            border-color: rgba(239, 68, 68, 0.7);
            background: linear-gradient(180deg, rgba(127, 29, 29, 0.22), rgba(15, 23, 42, 0.96));
        }

        .node-label {
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.2rem;
        }

        .node-name {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text);
        }

        .meta-panel {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 0.8rem;
            background: rgba(15, 23, 42, 0.8);
        }

        .meta-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(148, 163, 184, 0.08);
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-size: 0.72rem;
            font-weight: 600;
            margin: 0.15rem 0.35rem 0.15rem 0;
        }

        .metric-pill {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(17, 24, 39, 0.9));
        }

        .metric-pill .label {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            display: block;
            margin-bottom: 0.3rem;
        }

        .metric-pill .value {
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--text);
        }

        .doc-card {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }

        .source-list {
            margin: 0.5rem 0 0 0;
            padding-left: 1.1rem;
            color: var(--muted);
        }

        .source-list li {
            margin-bottom: 0.35rem;
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.94), rgba(15, 23, 42, 1));
            color: var(--text);
            font-weight: 600;
            transition: transform 0.18s ease, border-color 0.18s ease;
        }

        .stButton > button:hover {
            border-color: rgba(139, 92, 246, 0.7);
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.9), rgba(59, 130, 246, 0.9));
            border: none;
            color: white;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.7);
        }

        .stChatMessage {
            background: rgba(15, 23, 42, 0.66);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }

        .stChatInput {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.8);
        }

        .stAlert {
            border-radius: 12px;
        }

        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.7);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.35);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(139, 92, 246, 0.7);
        }
        </style>
        """,
        unsafe_allow_html=True
    )