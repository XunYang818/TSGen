import json
from typing import Any, List, Union


def convert_to_json(data: Union[List, Any]) -> str:
    """
    Convert data to JSON string (for LLM prompts)
    
    Args:
        data: Pydantic models, dicts, or lists
        
    Returns:
        JSON string
    """
    def to_dict(item):
        if isinstance(item, dict):
            return item
        # Pydantic v2: model_dump()
        if hasattr(item, 'model_dump'):
            return item.model_dump()
        # Fallback
        return item
    
    if isinstance(data, list):
        converted = [to_dict(item) for item in data]
    else:
        converted = to_dict(data)
    
    return json.dumps(converted, ensure_ascii=False, indent=2)
    