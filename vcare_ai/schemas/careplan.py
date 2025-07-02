from pydantic import BaseModel
from typing import Optional
from src.vcare_ai.schemas.shared import ClinicalRecommendationResult

class RecommendCarePlanPayload(BaseModel):
    patientId: str
    carePlanId: Optional[str] = None
    clinical_recommendation: ClinicalRecommendationResult

    def to_ai_payload(self, careplan_summary: dict) -> dict:
        """
        Combines validated clinical recommendation and careplan summary for prompt construction.
        """
        return {
            "careplan_summary": careplan_summary,
            "clinical_recommendation": self.clinical_recommendation.to_json()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RecommendCarePlanPayload':
        """Create a RecommendCarePlanPayload from a dictionary"""
        if isinstance(data, cls):
            return data
        # Don't convert clinical_recommendation here since it's already handled in validate_payload
        return cls(**data)