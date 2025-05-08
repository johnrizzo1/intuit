"""
Document class for vector store.
"""
from typing import Dict, Any
from pydantic import BaseModel

class Document(BaseModel):
    """Represents a document in the vector store."""
    content: str
    metadata: Dict[str, Any]
    id: str 