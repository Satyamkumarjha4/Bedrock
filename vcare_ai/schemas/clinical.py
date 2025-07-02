from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Union, Any



class LabResult(BaseModel):
    """Model for lab result values that handles string or numeric values"""
    value: Union[float, str]
    unit: Optional[str] = None
    
    @field_validator('value', mode='before')
    def parse_numeric_values(cls, v):
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                # Try to convert to float if it looks like a number
                if '.' in v:
                    return float(v)
                else:
                    return int(v)
            except (ValueError, TypeError):
                # Keep as string if not convertible
                return v
        return v



class ClinicalRecommendationPayload(BaseModel):
    age: int = Field(..., ge=0, le=120)
    conditions: List[str]
    lab_results: Dict[str, Union[LabResult, str, float]]
    medications: Optional[List[str]] = []
    use_template: Optional[str] = None
    disable_cache: Optional[bool] = False
    
    @field_validator('lab_results', mode='before')
    def normalize_lab_results(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Lab results must be a dictionary")
            
        # Convert simple string/number values to LabResult objects
        result = {}
        for key, value in v.items():
            if isinstance(value, (str, int, float)):
                result[key] = LabResult(value=value)
            elif isinstance(value, dict):
                result[key] = LabResult(**value)
            else:
                result[key] = value
        return result
    
    def to_ai_payload(self) -> Dict[str, Any]:
        """Convert to a dict suitable for AI model input"""
        # Process lab results to a simple dict
        processed_lab_results = {}
        for key, lab in self.lab_results.items():
            if isinstance(lab, LabResult):
                if lab.unit:
                    processed_lab_results[key] = f"{lab.value} {lab.unit}"
                else:
                    processed_lab_results[key] = lab.value
            else:
                processed_lab_results[key] = lab
                
        return {
            "age": self.age,
            "conditions": self.conditions,
            "lab_results": processed_lab_results,
            "medications": self.medications
        }
    
