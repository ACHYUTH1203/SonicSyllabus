from typing import TypedDict, List
from pydantic import BaseModel

class GraphState(TypedDict):
    """
    Represents the state of our SonicSyllabus graph.
    """
    question: str
    context: List[str]
    is_valid: bool
    target_language: str
    english_script: str
    translated_script: str

class AudioRequest(BaseModel):
    topic: str
    language: str
    user_id: str  