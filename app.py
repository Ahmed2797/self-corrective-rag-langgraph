"""
Main Application Entrypoint (Streamlit)
=======================================
Multi-project, multi-chat RAG workspace.
Upload PDF/TXT/MD -> index per project -> ask questions -> see answers.
"""

import hashlib
import os
import shutil
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Self-Corrective RAG Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.database.db import init_db, SessionLocal
from src.database.repository import (
    ProjectRepository,
    ChatRepository,
    MessageRepository,
    FileRepository,
)
from src.backend_adapter.service import RAGService
from src.retrive import index_files, delete_index
from frontend.theme import apply_custom_css
from frontend.components import (
    render_pipeline_status,
    render_message_metadata,
    render_project_summary,
)

UPLOAD_ROOT = os.path.join(os.getcwd(), "uploads")


# ==============================================================================
# HELPERS
# ==============================================================================
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_graph(project_id: str):
    """Compile the LangGraph pipeline once per project."""
    from src.pipeline import pipeline

    return pipeline(project_id)


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash_on_disk(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return bytes_hash(f.read())


def rerun():
    st.rerun()


# ==============================================================================
# SIDEBAR: PROJECTS & CHATS
# ==============================================================================
def render_sidebar(db):
    project_repo = ProjectRepository(db)
    chat_repo = ChatRepository(db)

    with st.sidebar:
        st.title("🤖 RAG Workspace")

        # ---------------- Projects ----------------
        st.subheader("Projects")
        projects = project_repo.get_all()

        with st.expander("➕ New Project"):
            name = st.text_input("Project Name", key="new_proj_name")
            desc = st.text_area("Description (Optional)", key="new_proj_desc")
            if st.button("Create Project", use_container_width=True):
                if name.strip():
                    created = project_repo.create(name.strip(), desc.strip())
                    st.session_state.active_project_id = created.id
                    st.session_state.active_chat_id = None
                    st.toast(f"Project '{created.name}' created!", icon="✅")
                    rerun()
                else:
                    st.warning("Please enter a project name.")

        if projects:
            options = {p.id: p.name for p in projects}
            ids = list(options.keys())

            if st.session_state.active_project_id not in options:
                st.session_state.active_project_id = ids[0]

            col_proj, col_actions = st.columns([0.85, 0.15])

            with col_proj:
                selected = st.selectbox(
                    "Select Project",
                    options=ids,
                    format_func=lambda x: options[x],
                    index=ids.index(st.session_state.active_project_id),
                    label_visibility="collapsed",
                )
                if selected != st.session_state.active_project_id:
                    st.session_state.active_project_id = selected
                    st.session_state.active_chat_id = None

            with col_actions:
                active = next(
                    (p for p in projects if p.id == st.session_state.active_project_id),
                    None,
                )
                if active:
                    with st.popover("⚙️", help="Project Actions"):
                        st.markdown("**Project Settings**")
                        renamed = st.text_input(
                            "Project Name",
                            value=active.name,
                            key=f"rename_proj_{active.id}",
                        )
                        if st.button(
                            "Save Name",
                            key=f"save_proj_{active.id}",
                            use_container_width=True,
                        ):
                            if renamed.strip():
                                project_repo.rename(active.id, renamed.strip())
                                rerun()

                        st.divider()
                        st.markdown("**Danger Zone**")
                        st.caption(
                            "Deleting a project removes its chats, files, and index data."
                        )
                        if st.button(
                            "🗑️ Delete Project",
                            key=f"delete_proj_{active.id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            project_repo.delete(active.id)
                            shutil.rmtree(
                                os.path.join(UPLOAD_ROOT, active.id), ignore_errors=True
                            )
                            delete_index(active.id)
                            load_graph.clear()
                            st.session_state.active_project_id = None
                            st.session_state.active_chat_id = None
                            st.toast(f"Project '{active.name}' deleted.", icon="🗑️")
                            rerun()
        else:
            st.caption("No projects available. Create one to get started.")

        st.divider()

        # ---------------- Chats ----------------
        if st.session_state.active_project_id:
            st.subheader("Chats")

            if st.button("➕ New Chat Thread", use_container_width=True):
                new_chat = chat_repo.create(
                    project_id=st.session_state.active_project_id
                )
                st.session_state.active_chat_id = new_chat.id
                rerun()

            chats = chat_repo.get_by_project(st.session_state.active_project_id)
            if chats:
                chat_ids = [c.id for c in chats]
                if st.session_state.active_chat_id not in chat_ids:
                    st.session_state.active_chat_id = chats[0].id

                for c in chats:
                    is_active = c.id == st.session_state.active_chat_id
                    label = f"**{c.title}**" if is_active else c.title

                    col1, col2 = st.columns([0.8, 0.2])

                    with col1:
                        if st.button(
                            label, key=f"chat_btn_{c.id}", use_container_width=True
                        ):
                            st.session_state.active_chat_id = c.id
                            rerun()

                    with col2:
                        with st.popover("⚙️", help="Chat Options"):
                            st.markdown("**Rename Chat**")
                            new_title = st.text_input(
                                "New title",
                                value=c.title,
                                key=f"rename_input_{c.id}",
                                label_visibility="collapsed",
                            )
                            if st.button(
                                "Save",
                                key=f"save_rename_{c.id}",
                                use_container_width=True,
                            ):
                                if new_title.strip():
                                    chat_repo.rename(c.id, new_title.strip())
                                    rerun()

                            st.divider()
                            st.markdown("**Delete Chat**")
                            if st.button(
                                "🗑️ Confirm Delete",
                                key=f"delete_chat_{c.id}",
                                type="primary",
                                use_container_width=True,
                            ):
                                chat_repo.delete(c.id)
                                if st.session_state.active_chat_id == c.id:
                                    st.session_state.active_chat_id = None
                                rerun()
            else:
                st.caption("No chat threads in this project.")


# ==============================================================================
# TAB 1: CHAT
# ==============================================================================
def render_chat_tab(db, graph_app, active_project, active_chat, doc_count):
    message_repo = MessageRepository(db)

    if not active_chat:
        st.info("Select or create a chat thread from the sidebar to begin.")
        return

    if graph_app is None:
        st.error("RAG pipeline is unavailable. Check `src.pipeline` and the logs.")
        return

    st.subheader(f"Thread: {active_chat.title}")

    if doc_count == 0:
        st.warning(
            "This project has no documents yet. Upload some in the "
            "**Document Management** tab so answers can use them."
        )

    for msg in message_repo.get_by_chat(active_chat.id):
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.role == "assistant" and msg.metadata_dict:
                render_message_metadata(msg.metadata_dict)

    user_input = st.chat_input("Ask a question about your documents...")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    status_container = st.empty()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        rag_service = RAGService(graph_app, db)
        node_events = {}
        pipeline_error = None

        try:
            stream_gen = rag_service.run_stream(
                chat_id=active_chat.id,
                project_id=active_project.id,
                user_question=user_input,
            )

            final_result = None
            while True:
                try:
                    event = next(stream_gen)
                    node_events[event.node] = event.status
                    if event.status == "failed":
                        pipeline_error = event.error or "The RAG pipeline failed."
                    with status_container.container():
                        render_pipeline_status(node_events)
                except StopIteration as e:
                    final_result = e.value
                    break

            status_container.empty()
            if final_result:
                answer = str(final_result.get("answer") or "").strip()
                if answer:
                    message_placeholder.markdown(answer)
                else:
                    message_placeholder.warning(
                        "The pipeline finished without returning an answer. "
                        "Check the application logs for details."
                    )
                if final_result.get("metadata"):
                    render_message_metadata(final_result["metadata"])
            elif pipeline_error:
                message_placeholder.error(f"Could not generate an answer: {pipeline_error}")
            else:
                message_placeholder.error(
                    "The pipeline stopped before producing an answer. "
                    "Check the application logs for details."
                )

        except Exception as e:
            status_container.empty()
            st.error(f"Error during graph execution: {e}")
            return

    rerun()


# ==============================================================================
# TAB 2: DOCUMENTS (upload -> save -> index)
# ==============================================================================
def render_files_tab(db, active_project):
    file_repo = FileRepository(db)

    st.subheader("Project Documents")

    # Version counter resets the uploader after a successful run
    version_key = f"uploader_version_{active_project.id}"
    st.session_state.setdefault(version_key, 0)

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / MD files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
        key=f"file_uploader_{active_project.id}_{st.session_state[version_key]}",
    )

    existing = file_repo.get_by_project(active_project.id)
    if existing:
        with st.expander(f"📚 {len(existing)} indexed document(s)", expanded=True):
            for f in existing:
                st.write(f"• {f.display_name}")
    else:
        st.caption("No documents indexed yet.")

    if not uploaded_files:
        return

    st.info(f"{len(uploaded_files)} file(s) selected.")

    if not st.button(
        "📤 Process Files", key=f"process_{active_project.id}", type="primary"
    ):
        return

    upload_dir = os.path.join(UPLOAD_ROOT, active_project.id)
    os.makedirs(upload_dir, exist_ok=True)

    indexed, skipped, failed, empty = [], [], [], []

    with st.status("Processing documents...", expanded=True) as status:
        for u_file in uploaded_files:
            data = u_file.getvalue()
            safe_name = os.path.basename(u_file.name)
            file_path = os.path.join(upload_dir, safe_name)

            # Same name + same content already indexed -> skip
            if file_hash_on_disk(file_path) == bytes_hash(data):
                skipped.append(safe_name)
                st.write(f"⏭️ {safe_name}: already indexed")
                continue

            st.write(f"⏳ {safe_name}: extracting, chunking, embedding...")
            with open(file_path, "wb") as f:
                f.write(data)

            try:
                n_chunks = index_files(active_project.id, [file_path])
            except Exception as e:
                os.remove(file_path)  # so a retry isn't treated as a duplicate
                failed.append(safe_name)
                st.write(f"❌ {safe_name}: {e}")
                continue

            if n_chunks == 0:
                os.remove(file_path)
                empty.append(safe_name)
                st.write(f"⚠️ {safe_name}: no text found (scanned PDF?)")
                continue

            # DB record only after indexing succeeded
            file_repo.create(
                project_id=active_project.id,
                original_name=u_file.name,
                display_name=u_file.name,
                storage_path=file_path,
                file_type=u_file.type or safe_name.rsplit(".", 1)[-1],
                size=u_file.size,
            )
            indexed.append(safe_name)
            st.write(f"✅ {safe_name}: {n_chunks} chunks indexed")

        status.update(label="Processing finished", state="complete", expanded=False)

    if indexed:
        load_graph.clear()  # rebuild retriever so it sees the new chunks
        st.toast(f"Indexed {len(indexed)} file(s).", icon="✅")
    if skipped:
        st.toast(f"Skipped duplicates: {', '.join(skipped)}", icon="⚠️")
    if failed or empty:
        # Keep the uploader as is, so the user can see what failed and retry
        st.error(
            "Some files could not be indexed: "
            + ", ".join(failed + empty)
        )
        if indexed:
            st.button("Continue", on_click=lambda: None)
        return

    st.session_state[version_key] += 1  # clears the uploader
    rerun()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    init_db()
    apply_custom_css()

    st.session_state.setdefault("active_project_id", None)
    st.session_state.setdefault("active_chat_id", None)

    db = SessionLocal()
    try:
        render_sidebar(db)

        if not st.session_state.active_project_id:
            st.warning("Please create or select a project from the sidebar to begin.")
            st.stop()

        project_repo = ProjectRepository(db)
        chat_repo = ChatRepository(db)
        file_repo = FileRepository(db)

        active_project = project_repo.get_by_id(st.session_state.active_project_id)
        if active_project is None:
            st.session_state.active_project_id = None
            st.rerun()

        active_chat = (
            chat_repo.get_by_id(st.session_state.active_chat_id)
            if st.session_state.active_chat_id
            else None
        )

        try:
            graph_app = load_graph(active_project.id)
        except Exception as e:
            st.error(f"Could not build the RAG pipeline: {e}")
            graph_app = None

        project_chats = chat_repo.get_by_project(active_project.id)
        project_files = file_repo.get_by_project(active_project.id)

        render_project_summary(
            project_name=active_project.name,
            description=active_project.description or "",
            stats={
                "chats": len(project_chats),
                "documents": len(project_files),
                "retrievals": max(len(project_chats), 1),
                "avg_quality": "--",
            },
        )

        tab_chat, tab_files = st.tabs(["💬 Chat Workspace", "📄 Document Management"])

        with tab_chat:
            render_chat_tab(db, graph_app, active_project, active_chat, len(project_files))

        with tab_files:
            render_files_tab(db, active_project)
    finally:
        db.close()


main()
