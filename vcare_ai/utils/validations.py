from vcare_ai.schemas import USECASE_SCHEMAS
import uuid
from typing import Any
from vcare_ai.schemas.shared import ClinicalRecommendationResult

def validate_payload(usecase: str, payload: dict) -> Any:
    """Validate payload against the appropriate schema"""
    schema = USECASE_SCHEMAS.get(usecase)
    if not schema:
        raise ValidationError(f"Unknown usecase: {usecase}")
    
    # For recommend_careplan, we need to handle the clinical_recommendation field specially
    if usecase == "recommend_careplan" and "clinical_recommendation" in payload:
        # Convert to ClinicalRecommendationResult for validation
        clinical_rec = ClinicalRecommendationResult.from_dict(payload["clinical_recommendation"])
        # Convert back to dict for the schema
        payload["clinical_recommendation"] = clinical_rec.to_json()
    
    return schema(**payload)

def generate_request_id() -> str:
    """Generate a unique request ID for tracing"""
    return f"req_{uuid.uuid4().hex[:8]}"