"""
Data Access Layer (Repository Pattern)
=======================================
Provides CRUD operations for Projects, Chats, Messages, Files, and Logs.
"""

from typing import List, Optional
import json
from sqlalchemy.orm import Session

from src.database.models import Project, Chat, Message, ProjectFile, ExecutionLog


class ProjectRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, name: str, description: Optional[str] = None) -> Project:
        project = Project(name=name, description=description)
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project

    def get_all(self) -> List[Project]:
        return self.db.query(Project).order_by(Project.created_at.desc()).all()

    def get_by_id(self, project_id: str) -> Optional[Project]:
        return self.db.query(Project).filter(Project.id == project_id).first()

    def rename(self, project_id: str, new_name: str) -> Optional[Project]:
        project = self.get_by_id(project_id)
        if project:
            project.name = new_name
            self.db.commit()
            self.db.refresh(project)
        return project

    def update_name(self, project_id: str, new_name: str) -> Optional[Project]:
        return self.rename(project_id, new_name)

    def delete(self, project_id: str) -> bool:
        project = self.get_by_id(project_id)
        if project:
            self.db.delete(project)
            self.db.commit()
            return True
        return False


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, project_id: str, title: str = "New Chat") -> Chat:
        chat = Chat(project_id=project_id, title=title)
        self.db.add(chat)
        self.db.commit()
        self.db.refresh(chat)
        return chat

    def get_by_project(self, project_id: str) -> List[Chat]:
        return self.db.query(Chat).filter(Chat.project_id == project_id).order_by(Chat.created_at.desc()).all()

    def get_by_id(self, chat_id: str) -> Optional[Chat]:
        return self.db.query(Chat).filter(Chat.id == chat_id).first()

    def rename(self, chat_id: str, new_title: str) -> Optional[Chat]:
        chat = self.get_by_id(chat_id)
        if chat:
            chat.title = new_title
            self.db.commit()
            self.db.refresh(chat)
        return chat

    def delete(self, chat_id: str) -> bool:
        chat = self.get_by_id(chat_id)
        if chat:
            self.db.delete(chat)
            self.db.commit()
            return True
        return False


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, chat_id: str, role: str, content: str, metadata: Optional[dict] = None) -> Message:
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
            metadata_json=json.dumps(metadata or {})
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_by_chat(self, chat_id: str) -> List[Message]:
        return self.db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()


class FileRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, project_id: str, original_name: str, display_name: str, storage_path: str, file_type: str, size: int) -> ProjectFile:
        file_rec = ProjectFile(
            project_id=project_id,
            original_name=original_name,
            display_name=display_name,
            storage_path=storage_path,
            file_type=file_type,
            size=size
        )
        self.db.add(file_rec)
        self.db.commit()
        self.db.refresh(file_rec)
        return file_rec

    def get_by_project(self, project_id: str) -> List[ProjectFile]:
        return self.db.query(ProjectFile).filter(ProjectFile.project_id == project_id).order_by(ProjectFile.created_at.desc()).all()

    def get_by_id(self, file_id: str) -> Optional[ProjectFile]:
        return self.db.query(ProjectFile).filter(ProjectFile.id == file_id).first()

    def rename_display_name(self, file_id: str, new_display_name: str) -> Optional[ProjectFile]:
        file_rec = self.get_by_id(file_id)
        if file_rec:
            file_rec.display_name = new_display_name
            self.db.commit()
            self.db.refresh(file_rec)
        return file_rec

    def delete(self, file_id: str) -> bool:
        file_rec = self.get_by_id(file_id)
        if file_rec:
            self.db.delete(file_rec)
            self.db.commit()
            return True
        return False