from typing import Optional
from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 128
    min_length: Optional[int] = 16
