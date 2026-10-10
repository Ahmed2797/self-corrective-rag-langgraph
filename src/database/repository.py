from src.logger import logging
from typing import List, Optional
import json

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from src.database.models import Project, Chat, Message, ProjectFile, ExecutionLog

logger = logging.getLogger(__name__)


# =====================================================================
# PROJECT REPOSITORY
# =====================================================================

class ProjectRepository:
    """Repository for Project CRUD operations"""
    
    def __init__(self, db: Session):
        if not db:
            raise ValueError("Database session is required")
        self.db = db

    def create(self, name: str, description: Optional[str] = None) -> Project:
        """
        Create new project with validation.
        
        Args:
            name: Project name (required, non-empty)
            description: Optional description
            
        Returns:
            Created Project
            
        Raises:
            ValueError: If name is empty
            IntegrityError: If name is duplicate
        """
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")
        
        try:
            project = Project(
                name=name.strip(),
                description=description.strip() if description else None
            )
            self.db.add(project)
            self.db.flush()  # Validate before commit
            self.db.commit()
            self.db.refresh(project)
            
            logger.info(f"✅ Created project: id={project.id} name={project.name}")
            return project
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"❌ Duplicate project name: {name}")
            raise ValueError(f"Project name '{name}' already exists") from e
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error creating project: {e}")
            raise
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Unexpected error creating project: {e}")
            raise

    def get_all(self) -> List[Project]:
        """Get all projects ordered by creation date."""
        try:
            projects = self.db.query(Project).order_by(
                Project.created_at.desc()
            ).all()
            logger.debug(f"Retrieved {len(projects)} projects")
            return projects
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error fetching projects: {e}")
            raise

    def get_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        if not project_id:
            logger.warning("Attempted get_by_id with empty project_id")
            return None
            
        try:
            project = self.db.query(Project).filter(
                Project.id == project_id
            ).first()
            
            if not project:
                logger.warning(f"Project not found: {project_id}")
            return project
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error fetching project {project_id}: {e}")
            raise

    def rename(self, project_id: str, new_name: str) -> Optional[Project]:
        """Rename project."""
        if not project_id:
            raise ValueError("Project ID is required")
        if not new_name or not new_name.strip():
            raise ValueError("New project name cannot be empty")
        
        try:
            project = self.get_by_id(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            old_name = project.name
            project.name = new_name.strip()
            
            self.db.flush()
            self.db.commit()
            self.db.refresh(project)
            
            logger.info(f"✅ Renamed project {project_id}: '{old_name}' → '{project.name}'")
            return project
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"❌ Duplicate project name: {new_name}")
            raise ValueError(f"Project name '{new_name}' already exists") from e
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error renaming project {project_id}: {e}")
            raise

    def delete(self, project_id: str) -> bool:
        """Delete project and all related data (cascading)."""
        if not project_id:
            raise ValueError("Project ID is required")
        
        try:
            project = self.get_by_id(project_id)
            if not project:
                logger.warning(f"Attempted to delete non-existent project: {project_id}")
                return False
            
            self.db.delete(project)
            self.db.commit()
            
            logger.info(f"✅ Deleted project: {project_id} (cascaded to chats, files)")
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error deleting project {project_id}: {e}")
            raise


# =====================================================================
# CHAT REPOSITORY
# =====================================================================

class ChatRepository:
    """Repository for Chat CRUD operations"""
    
    def __init__(self, db: Session):
        if not db:
            raise ValueError("Database session is required")
        self.db = db

    def create(self, project_id: str, title: str = "New Chat") -> Chat:
        """Create new chat thread."""
        if not project_id:
            raise ValueError("Project ID is required")
        if not title or not title.strip():
            title = "New Chat"
        
        try:
            chat = Chat(
                project_id=project_id,
                title=title.strip()
            )
            self.db.add(chat)
            self.db.flush()
            self.db.commit()
            self.db.refresh(chat)
            
            logger.info(f"✅ Created chat: id={chat.id} project_id={project_id}")
            return chat
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error creating chat: {e}")
            raise

    def get_by_project(self, project_id: str) -> List[Chat]:
        """Get all chats for a project."""
        if not project_id:
            logger.warning("Attempted get_by_project with empty project_id")
            return []
        
        try:
            chats = self.db.query(Chat).filter(
                Chat.project_id == project_id
            ).order_by(Chat.created_at.desc()).all()
            
            logger.debug(f"Retrieved {len(chats)} chats for project {project_id}")
            return chats
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error fetching chats for {project_id}: {e}")
            raise

    def get_by_id(self, chat_id: str) -> Optional[Chat]:
        """Get chat by ID."""
        if not chat_id:
            return None
        
        try:
            chat = self.db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                logger.warning(f"Chat not found: {chat_id}")
            return chat
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error fetching chat {chat_id}: {e}")
            raise

    def rename(self, chat_id: str, new_title: str) -> Optional[Chat]:
        """Rename chat thread."""
        if not chat_id:
            raise ValueError("Chat ID is required")
        if not new_title or not new_title.strip():
            raise ValueError("Chat title cannot be empty")
        
        try:
            chat = self.get_by_id(chat_id)
            if not chat:
                raise ValueError(f"Chat {chat_id} not found")
            
            old_title = chat.title
            chat.title = new_title.strip()
            
            self.db.flush()
            self.db.commit()
            self.db.refresh(chat)
            
            logger.info(f"✅ Renamed chat {chat_id}: '{old_title}' → '{chat.title}'")
            return chat
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error renaming chat {chat_id}: {e}")
            raise

    def delete(self, chat_id: str) -> bool:
        """Delete chat and all messages."""
        if not chat_id:
            raise ValueError("Chat ID is required")
        
        try:
            chat = self.get_by_id(chat_id)
            if not chat:
                logger.warning(f"Attempted to delete non-existent chat: {chat_id}")
                return False
            
            self.db.delete(chat)
            self.db.commit()
            
            logger.info(f"✅ Deleted chat: {chat_id} (cascaded to messages, logs)")
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error deleting chat {chat_id}: {e}")
            raise


# =====================================================================
# MESSAGE REPOSITORY
# =====================================================================

class MessageRepository:
    """Repository for Message CRUD operations"""
    
    def __init__(self, db: Session):
        if not db:
            raise ValueError("Database session is required")
        self.db = db

    def create(
        self,
        chat_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> Message:
        """Create new message."""
        if not chat_id:
            raise ValueError("Chat ID is required")
        if not role or role not in {'user', 'assistant', 'system'}:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty")
        
        try:
            message = Message(
                chat_id=chat_id,
                role=role,
                content=content.strip(),
                metadata_json=json.dumps(metadata or {})
            )
            self.db.add(message)
            self.db.flush()
            self.db.commit()
            self.db.refresh(message)
            
            logger.debug(f"✅ Created message: id={message.id} role={role} len={len(content)}")
            return message
            
        except ValueError as e:
            self.db.rollback()
            logger.error(f"❌ Validation error: {e}")
            raise
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error creating message: {e}")
            raise

    def get_by_chat(self, chat_id: str) -> List[Message]:
        """Get all messages in a chat."""
        if not chat_id:
            logger.warning("Attempted get_by_chat with empty chat_id")
            return []
        
        try:
            messages = self.db.query(Message).filter(
                Message.chat_id == chat_id
            ).order_by(Message.created_at.asc()).all()
            
            logger.debug(f"Retrieved {len(messages)} messages for chat {chat_id}")
            return messages
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error fetching messages for {chat_id}: {e}")
            raise


# =====================================================================
# FILE REPOSITORY
# =====================================================================

class FileRepository:
    """Repository for ProjectFile CRUD operations"""
    
    def __init__(self, db: Session):
        if not db:
            raise ValueError("Database session is required")
        self.db = db

    def create(
        self,
        project_id: str,
        original_name: str,
        display_name: str,
        storage_path: str,
        file_type: str,
        size: int
    ) -> ProjectFile:
        """Create file record."""
        if not project_id:
            raise ValueError("Project ID is required")
        if not original_name or not original_name.strip():
            raise ValueError("Original filename cannot be empty")
        if not storage_path or not storage_path.strip():
            raise ValueError("Storage path cannot be empty")
        if size < 0:
            raise ValueError("File size cannot be negative")
        
        try:
            file_rec = ProjectFile(
                project_id=project_id,
                original_name=original_name.strip(),
                display_name=(display_name or original_name).strip(),
                storage_path=storage_path.strip(),
                file_type=file_type.strip() if file_type else "unknown",
                size=size
            )
            self.db.add(file_rec)
            self.db.flush()
            self.db.commit()
            self.db.refresh(file_rec)
            
            logger.info(f"✅ Created file record: {file_rec.id} {original_name} ({size} bytes)")
            return file_rec
            
        except ValueError as e:
            self.db.rollback()
            logger.error(f"❌ Validation error: {e}")
            raise
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error creating file record: {e}")
            raise

    def get_by_project(self, project_id: str) -> List[ProjectFile]:
        """Get all files for a project."""
        if not project_id:
            logger.warning("Attempted get_by_project with empty project_id")
            return []
        
        try:
            files = self.db.query(ProjectFile).filter(
                ProjectFile.project_id == project_id
            ).order_by(ProjectFile.created_at.desc()).all()
            
            logger.debug(f"Retrieved {len(files)} files for project {project_id}")
            return files
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error fetching files for {project_id}: {e}")
            raise

    def get_by_id(self, file_id: str) -> Optional[ProjectFile]:
        """Get file by ID."""
        if not file_id:
            return None
        
        try:
            file_rec = self.db.query(ProjectFile).filter(
                ProjectFile.id == file_id
            ).first()
            
            if not file_rec:
                logger.warning(f"File not found: {file_id}")
            return file_rec
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error fetching file {file_id}: {e}")
            raise

    def delete(self, file_id: str) -> bool:
        """Delete file record."""
        if not file_id:
            raise ValueError("File ID is required")
        
        try:
            file_rec = self.get_by_id(file_id)
            if not file_rec:
                logger.warning(f"Attempted to delete non-existent file: {file_id}")
                return False
            
            self.db.delete(file_rec)
            self.db.commit()
            
            logger.info(f"✅ Deleted file record: {file_id}")
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Database error deleting file {file_id}: {e}")
            raise