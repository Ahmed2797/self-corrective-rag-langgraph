import uuid
from datetime import datetime
import json
from src.logger import logging

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Integer, Text, BigInteger,
    UniqueConstraint, CheckConstraint, Index, event
)
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from src.database.db import Base

logger = logging.getLogger(__name__)


def generate_uuid():
    """Generate UUID for primary keys"""
    return str(uuid.uuid4())


class Project(Base):
    """Project model with validation and constraints"""
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Add constraints
    __table_args__ = (
        UniqueConstraint('name', name='uq_project_name'),
        CheckConstraint('LENGTH(name) > 0', name='ck_project_name_not_empty'),
        Index('idx_project_created', 'created_at'),
    )

    # Relationships (Cascading deletes)
    chats = relationship("Chat", back_populates="project", cascade="all, delete-orphan")
    files = relationship("ProjectFile", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project id={self.id} name={self.name}>"


class Chat(Base):
    """Chat model with proper foreign key handling"""
    __tablename__ = "chats"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint('LENGTH(title) > 0', name='ck_chat_title_not_empty'),
        Index('idx_chat_project_created', 'project_id', 'created_at'),
    )

    project = relationship("Project", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    execution_logs = relationship("ExecutionLog", back_populates="chat", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Chat id={self.id} project_id={self.project_id} title={self.title}>"


class Message(Base):
    """Message model with metadata and audit trail"""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name='ck_message_role_valid'
        ),
        CheckConstraint('LENGTH(content) > 0', name='ck_message_content_not_empty'),
        Index('idx_message_chat_created', 'chat_id', 'created_at'),
    )

    chat = relationship("Chat", back_populates="messages")

    @property
    def metadata_dict(self):
        """Helper to get metadata as dict."""
        try:
            return json.loads(self.metadata_json or "{}")
        except Exception as e:
            logger.warning(f"Failed to parse metadata JSON: {e}")
            return {}

    @metadata_dict.setter
    def metadata_dict(self, value: dict):
        """Helper to set metadata from dict."""
        try:
            self.metadata_json = json.dumps(value or {})
        except Exception as e:
            logger.error(f"Failed to serialize metadata: {e}")
            self.metadata_json = "{}"

    def __repr__(self):
        return f"<Message id={self.id} chat_id={self.chat_id} role={self.role}>"


class ProjectFile(Base):
    """File model with metadata"""
    __tablename__ = "files"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    original_name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    storage_path = Column(Text, nullable=False)
    file_type = Column(String(50), nullable=False)
    size = Column(BigInteger, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint('LENGTH(original_name) > 0', name='ck_file_original_name_not_empty'),
        CheckConstraint('LENGTH(display_name) > 0', name='ck_file_display_name_not_empty'),
        CheckConstraint('size >= 0', name='ck_file_size_positive'),
        Index('idx_file_project_created', 'project_id', 'created_at'),
    )

    project = relationship("Project", back_populates="files")

    def __repr__(self):
        return f"<ProjectFile id={self.id} project_id={self.project_id} name={self.display_name}>"


class ExecutionLog(Base):
    """Execution log for audit trail and debugging"""
    __tablename__ = "execution_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(String(36), nullable=True)
    node_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # 'pending', 'running', 'completed', 'failed'
    metadata_json = Column(Text, nullable=True, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name='ck_log_status_valid'
        ),
        Index('idx_log_chat_created', 'chat_id', 'created_at'),
        Index('idx_log_status', 'status'),
    )

    chat = relationship("Chat", back_populates="execution_logs")

    @property
    def metadata_dict(self):
        try:
            return json.loads(self.metadata_json or "{}")
        except Exception:
            return {}

    @metadata_dict.setter
    def metadata_dict(self, value: dict):
        self.metadata_json = json.dumps(value or {})

    def __repr__(self):
        return f"<ExecutionLog id={self.id} chat_id={self.chat_id} node={self.node_name} status={self.status}>"


# =====================================================================
# CONNECTION POOL CONFIGURATION
# =====================================================================

def configure_database_engine(database_url: str, echo: bool = False):
    """
    Configure SQLAlchemy engine with production-ready settings.
    
    Args:
        database_url: Database connection URL
        echo: Whether to log SQL statements
        
    Returns:
        SQLAlchemy engine
    """
    engine = create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=20,              # Connections in pool
        max_overflow=10,           # Extra connections allowed
        pool_timeout=30,           # Wait 30s for connection
        pool_recycle=3600,         # Recycle every 1 hour
        echo=echo,
        connect_args={
            "timeout": 10,         # Connection timeout
            "check_same_thread": False,  # For SQLite
        }
    )
    
    # Log pool events for monitoring
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        logger.debug("Database connection opened")

    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_conn, connection_record):
        logger.debug("Database connection returned to pool")

    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Database connection checked out from pool")

    @event.listens_for(engine, "close")
    def receive_close(dbapi_conn, connection_record):
        logger.debug("Database connection closed")

    @event.listens_for(engine, "detach")
    def receive_detach(dbapi_conn, connection_record):
        logger.debug("Database connection detached")

    logger.info(f"Configured database engine with pool_size=20, max_overflow=10")
    
    return engine