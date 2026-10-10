import hashlib
import os
import shutil
import sys
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any
import traceback

import streamlit as st

# =====================================================
# LOGGING SETUP
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Self-Corrective RAG Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.database.db import init_db, SessionLocal, engine
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


UPLOAD_ROOT = Path(os.getenv("RAG_UPLOAD_ROOT", "./uploads")).resolve()
MAX_FILE_SIZE_MB = int(os.getenv("RAG_MAX_FILE_SIZE_MB", "100"))
MAX_FILES_PER_UPLOAD = int(os.getenv("RAG_MAX_FILES_PER_UPLOAD", "10"))

# Create directories
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

logger.info(f"Upload root: {UPLOAD_ROOT}")
logger.info(f"Max file size: {MAX_FILE_SIZE_MB}MB")



@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


class GraphCacheManager:
    """Manages graph caching with version tracking"""
    
    def __init__(self):
        self.cache = {}
        self.versions = {}
    
    def get_version_key(self, project_id: str) -> str:
        """Get unique key that changes when files are updated"""
        key = f"graph_{project_id}"
        if key not in self.versions:
            self.versions[key] = 0
        return f"{key}_v{self.versions[key]}"
    
    def get(self, project_id: str):
        """Get cached graph if exists"""
        key = self.get_version_key(project_id)
        return self.cache.get(key)
    
    def set(self, project_id: str, graph):
        """Cache graph with current version"""
        key = self.get_version_key(project_id)
        self.cache[key] = graph
    
    def invalidate(self, project_id: str):
        """Invalidate cache for project (when files change)"""
        self.versions[f"graph_{project_id}"] = self.versions.get(f"graph_{project_id}", 0) + 1
        logger.info(f"Invalidated graph cache for project {project_id}")


graph_cache_manager = GraphCacheManager()


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_graph(project_id: str, cache_version: int):
    """Compile the LangGraph pipeline once per project (with version tracking)"""
    try:
        logger.info(f"Loading graph for project {project_id} (v{cache_version})")
        from src.pipeline import pipeline
        graph = pipeline(project_id)
        graph_cache_manager.set(project_id, graph)
        logger.info(f"Graph loaded successfully for {project_id}")
        return graph
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        st.error(f"Failed to load RAG pipeline: {e}")
        raise


def safe_index_files(project_id: str, file_paths: list) -> Optional[int]:
    """Safe file indexing with error recovery"""
    try:
        n_chunks = index_files(project_id, file_paths)
        return n_chunks
    except Exception as e:
        logger.error(f"Indexing error for {file_paths}: {e}")
        # Try to recover by invalidating cache
        graph_cache_manager.invalidate(project_id)
        raise


def validate_uploaded_file(u_file) -> tuple[bool, str]:
    """Validate uploaded file"""
    # Check size
    if u_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File too large (max {MAX_FILE_SIZE_MB}MB)"
    
    # Check extension
    allowed = {"pdf", "txt", "md"}
    ext = u_file.name.rsplit(".", 1)[-1].lower() if "." in u_file.name else ""
    if ext not in allowed:
        return False, f"File type not allowed (use: {', '.join(allowed)})"
    
    # Check if readable
    try:
        u_file.seek(0)
        _ = u_file.read(100)  # Try to read first 100 bytes
        u_file.seek(0)
    except Exception as e:
        return False, f"File is not readable: {e}"
    
    return True, ""


def bytes_hash(data: bytes) -> str:
    """Hash file content"""
    return hashlib.sha256(data).hexdigest()


def file_hash_on_disk(path: str) -> Optional[str]:
    """Get hash of file on disk"""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return bytes_hash(f.read())
    except Exception as e:
        logger.warning(f"Could not hash file {path}: {e}")
        return None


# =====================================================
# SIDEBAR: PROJECTS & CHATS
# =====================================================
def render_sidebar(db):
    """Render sidebar with projects and chats"""
    project_repo = ProjectRepository(db)
    chat_repo = ChatRepository(db)

    with st.sidebar:
        st.title("🤖 RAG Workspace")

        # ---- Projects ----
        st.subheader("Projects")
        
        try:
            projects = project_repo.get_all()
        except Exception as e:
            logger.error(f"Failed to load projects: {e}")
            st.error("Could not load projects from database")
            projects = []

        with st.expander("➕ New Project"):
            name = st.text_input("Project Name", key="new_proj_name")
            desc = st.text_area("Description (Optional)", key="new_proj_desc")
            if st.button("Create Project", use_container_width=True):
                if not name.strip():
                    st.warning("Please enter a project name")
                    return
                
                try:
                    created = project_repo.create(name.strip(), desc.strip())
                    st.session_state.active_project_id = created.id
                    st.session_state.active_chat_id = None
                    st.toast(f"Project '{created.name}' created!", icon="✅")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to create project: {e}")
                    st.error(f"Could not create project: {e}")

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
                                try:
                                    project_repo.rename(active.id, renamed.strip())
                                    st.rerun()
                                except Exception as e:
                                    logger.error(f"Failed to rename: {e}")
                                    st.error(f"Could not rename project: {e}")

                        st.divider()
                        st.markdown("**Danger Zone**")
                        st.caption("Deleting a project removes its chats, files, and data.")
                        
                        if st.button(
                            "🗑️ Delete Project",
                            key=f"delete_proj_{active.id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                project_repo.delete(active.id)
                                
                                try:
                                    delete_index(active.id)
                                except Exception as e:
                                    logger.error(f"Failed to delete index: {e}")
                                    st.warning("Index cleanup incomplete (non-critical)")
                                
                                # Clean up files
                                project_dir = UPLOAD_ROOT / active.id
                                if project_dir.exists():
                                    shutil.rmtree(project_dir, ignore_errors=True)
                                
                                # Invalidate cache
                                graph_cache_manager.invalidate(active.id)
                                
                                st.session_state.active_project_id = None
                                st.session_state.active_chat_id = None
                                st.toast(f"Project '{active.name}' deleted.", icon="🗑️")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to delete project: {e}")
                                st.error(f"Could not delete project: {e}")
        else:
            st.caption("No projects available. Create one to get started.")

        st.divider()

        # ---- Chats ----
        if st.session_state.active_project_id:
            st.subheader("Chats")

            if st.button("➕ New Chat Thread", use_container_width=True):
                try:
                    new_chat = chat_repo.create(
                        project_id=st.session_state.active_project_id
                    )
                    st.session_state.active_chat_id = new_chat.id
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to create chat: {e}")
                    st.error(f"Could not create chat: {e}")

            try:
                chats = chat_repo.get_by_project(st.session_state.active_project_id)
            except Exception as e:
                logger.error(f"Failed to load chats: {e}")
                st.error("Could not load chats")
                chats = []

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
                            st.rerun()

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
                                    try:
                                        chat_repo.rename(c.id, new_title.strip())
                                        st.rerun()
                                    except Exception as e:
                                        logger.error(f"Failed to rename chat: {e}")
                                        st.error(f"Could not rename: {e}")

                            st.divider()
                            st.markdown("**Delete Chat**")
                            if st.button(
                                "🗑️ Confirm Delete",
                                key=f"delete_chat_{c.id}",
                                type="primary",
                                use_container_width=True,
                            ):
                                try:
                                    chat_repo.delete(c.id)
                                    if st.session_state.active_chat_id == c.id:
                                        st.session_state.active_chat_id = None
                                    st.rerun()
                                except Exception as e:
                                    logger.error(f"Failed to delete chat: {e}")
                                    st.error(f"Could not delete chat: {e}")
            else:
                st.caption("No chat threads in this project.")


# =====================================================
# TAB 1: CHAT
# =====================================================
def render_chat_tab(db, graph_app, active_project, active_chat, doc_count):
    """Render chat tab with streaming support"""
    message_repo = MessageRepository(db)

    if not active_chat:
        st.info("Select or create a chat thread from the sidebar to begin.")
        return

    if graph_app is None:
        st.error(
            "RAG pipeline is unavailable. "
            "Check that documents are indexed and try refreshing the page."
        )
        return

    st.subheader(f"Thread: {active_chat.title}")

    if doc_count == 0:
        st.warning(
            "This project has no documents yet. "
            "Upload some in the **Document Management** tab."
        )

    try:
        messages = message_repo.get_by_chat(active_chat.id)
    except Exception as e:
        logger.error(f"Failed to load messages: {e}")
        st.error("Could not load chat history")
        messages = []

    for msg in messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.role == "assistant" and msg.metadata_dict:
                try:
                    render_message_metadata(msg.metadata_dict)
                except Exception as e:
                    logger.warning(f"Could not render metadata: {e}")

    user_input = st.chat_input("Ask a question about your documents...")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    status_container = st.empty()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            rag_service = RAGService(graph_app, db)
            node_events = {}
            pipeline_error = None
            final_result = None

            stream_gen = rag_service.run_stream(
                chat_id=active_chat.id,
                project_id=active_project.id,
                user_question=user_input,
            )

            for event in stream_gen:
                node_events[event.node] = event.status
                if event.status == "failed":
                    pipeline_error = event.error or "Pipeline failed"
                
                with status_container.container():
                    try:
                        render_pipeline_status(node_events)
                    except Exception as e:
                        logger.warning(f"Could not render pipeline status: {e}")

            status_container.empty()

            # The generator should have returned the result via return/StopIteration
            if hasattr(stream_gen, '__value__'):
                final_result = stream_gen.__value__

            if final_result:
                answer = str(final_result.get("answer") or "").strip()
                if answer:
                    message_placeholder.markdown(answer)
                    
                    # Save message to DB
                    try:
                        message_repo.create(
                            chat_id=active_chat.id,
                            role="assistant",
                            content=answer,
                            metadata=final_result.get("metadata", {})
                        )
                    except Exception as e:
                        logger.error(f"Failed to save message: {e}")
                else:
                    message_placeholder.warning(
                        "Pipeline finished but returned no answer. "
                        "Check application logs."
                    )
                
                if final_result.get("metadata"):
                    try:
                        render_message_metadata(final_result["metadata"])
                    except Exception as e:
                        logger.warning(f"Could not render metadata: {e}")
            
            elif pipeline_error:
                message_placeholder.error(f"Pipeline error: {pipeline_error}")
            else:
                message_placeholder.error(
                    "Pipeline did not produce an answer. "
                    "Check the application logs."
                )

        except Exception as e:
            status_container.empty()
            logger.error(f"Error during execution: {e}\n{traceback.format_exc()}")
            st.error(f"Error: {e}")
            return

    st.rerun()


# =====================================================
# TAB 2: DOCUMENTS
# =====================================================
def render_files_tab(db, active_project):
    """Render file upload and management tab"""
    file_repo = FileRepository(db)

    st.subheader("Project Documents")

    # Version counter resets uploader after successful run
    version_key = f"uploader_version_{active_project.id}"
    st.session_state.setdefault(version_key, 0)

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / MD files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
        key=f"file_uploader_{active_project.id}_{st.session_state[version_key]}",
    )

    if uploaded_files and len(uploaded_files) > MAX_FILES_PER_UPLOAD:
        st.error(f"Max {MAX_FILES_PER_UPLOAD} files per upload")
        uploaded_files = uploaded_files[:MAX_FILES_PER_UPLOAD]

    try:
        existing = file_repo.get_by_project(active_project.id)
    except Exception as e:
        logger.error(f"Failed to load files: {e}")
        st.error("Could not load document list")
        existing = []

    if existing:
        with st.expander(f"📚 {len(existing)} indexed document(s)", expanded=True):
            for f in existing:
                st.write(f"• {f.display_name} ({f.size} bytes)")
    else:
        st.caption("No documents indexed yet.")

    if not uploaded_files:
        return

    st.info(f"{len(uploaded_files)} file(s) selected.")

    if not st.button(
        "📤 Process Files", key=f"process_{active_project.id}", type="primary"
    ):
        return

    upload_dir = UPLOAD_ROOT / active_project.id
    upload_dir.mkdir(parents=True, exist_ok=True)

    indexed, skipped, failed, empty = [], [], [], []

    with st.status("Processing documents...", expanded=True) as status:
        for u_file in uploaded_files:
            is_valid, error_msg = validate_uploaded_file(u_file)
            if not is_valid:
                failed.append((u_file.name, error_msg))
                st.write(f"❌ {u_file.name}: {error_msg}")
                continue

            data = u_file.getvalue()
            safe_name = os.path.basename(u_file.name)
            file_path = upload_dir / safe_name

            # Check if already indexed
            if file_hash_on_disk(str(file_path)) == bytes_hash(data):
                skipped.append(safe_name)
                st.write(f"⏭️ {safe_name}: already indexed")
                continue

            st.write(f"⏳ {safe_name}: extracting, chunking, embedding...")
            
            try:
                with open(file_path, "wb") as f:
                    f.write(data)

                n_chunks = safe_index_files(active_project.id, [str(file_path)])
                
                if n_chunks == 0:
                    file_path.unlink()
                    empty.append(safe_name)
                    st.write(f"⚠️ {safe_name}: no text found")
                    continue

                # Save to DB only after indexing succeeded
                try:
                    file_repo.create(
                        project_id=active_project.id,
                        original_name=u_file.name,
                        display_name=u_file.name,
                        storage_path=str(file_path),
                        file_type=u_file.type or safe_name.rsplit(".", 1)[-1],
                        size=u_file.size,
                    )
                except Exception as e:
                    logger.error(f"Failed to save file record: {e}")
                    st.write(f"⚠️ {safe_name}: indexed but DB record failed")
                    continue

                indexed.append(safe_name)
                st.write(f"✅ {safe_name}: {n_chunks} chunks indexed")
                
            except Exception as e:
                logger.error(f"Failed to index {safe_name}: {e}")
                if file_path.exists():
                    file_path.unlink()
                failed.append((safe_name, str(e)))
                st.write(f"❌ {safe_name}: {e}")
                continue

        status.update(label="Processing finished", state="complete", expanded=False)

    if indexed:
        graph_cache_manager.invalidate(active_project.id)
        st.toast(f"Indexed {len(indexed)} file(s).", icon="✅")
    
    if skipped:
        st.toast(f"Skipped {len(skipped)} duplicate(s).", icon="⚠️")
    
    if failed or empty:
        error_msg = ", ".join([f"{n} ({e})" for n, e in failed] + empty)
        st.error(f"Failed: {error_msg}")
        if indexed:
            st.button("Continue", on_click=lambda: None)
        return

    st.session_state[version_key] += 1  # Clear uploader
    st.rerun()


# =====================================================
# MAIN
# =====================================================
def main():
    """Main application"""
    try:
        init_db()
        apply_custom_css()

        st.session_state.setdefault("active_project_id", None)
        st.session_state.setdefault("active_chat_id", None)

        render_sidebar(SessionLocal())

        if not st.session_state.active_project_id:
            st.warning("Create or select a project from the sidebar.")
            st.stop()

        with get_db_session() as db:
            project_repo = ProjectRepository(db)
            chat_repo = ChatRepository(db)
            file_repo = FileRepository(db)

            try:
                active_project = project_repo.get_by_id(st.session_state.active_project_id)
            except Exception as e:
                logger.error(f"Failed to load project: {e}")
                st.error("Could not load project")
                st.stop()

            if not active_project:
                st.session_state.active_project_id = None
                st.rerun()

            try:
                active_chat = (
                    chat_repo.get_by_id(st.session_state.active_chat_id)
                    if st.session_state.active_chat_id
                    else None
                )
            except Exception as e:
                logger.error(f"Failed to load chat: {e}")
                active_chat = None

            try:
                cache_version = hash(st.session_state.active_project_id) % (2**31)
                graph_app = load_graph(active_project.id, cache_version)
            except Exception as e:
                logger.error(f"Failed to build graph: {e}")
                st.error(f"Could not build RAG pipeline: {e}")
                graph_app = None

            try:
                project_chats = chat_repo.get_by_project(active_project.id)
                project_files = file_repo.get_by_project(active_project.id)
            except Exception as e:
                logger.error(f"Failed to load project data: {e}")
                project_chats = []
                project_files = []

            render_project_summary(
                project_name=active_project.name,
                description=active_project.description or "",
                stats={
                    "chats": len(project_chats),
                    "documents": len(project_files),
                    "retrievals": "--",
                    "avg_quality": "--",
                },
            )

            tab_chat, tab_files = st.tabs(["💬 Chat Workspace", "📄 Document Management"])

            with tab_chat:
                render_chat_tab(db, graph_app, active_project, active_chat, len(project_files))

            with tab_files:
                render_files_tab(db, active_project)

    except Exception as e:
        logger.error(f"Fatal error in main: {e}\n{traceback.format_exc()}")
        st.error(f"Fatal application error: {e}")
        st.stop()


if __name__ == "__main__":
    main()