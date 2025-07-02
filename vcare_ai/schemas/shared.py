from pydantic import BaseModel
from typing import List


class ClinicalRecommendationResult(BaseModel):
    prescriptions: List[str]
    tests: List[str]
    referrals: List[str]
    reasoning: str

    def to_ai_payload(self) -> dict:
        """Convert the model to a dictionary for AI processing"""
        return self.dict()

    def to_json(self) -> dict:
        """Convert the model to a JSON-serializable dictionary"""
        return {
            "prescriptions": self.prescriptions,
            "tests": self.tests,
            "referrals": self.referrals,
            "reasoning": self.reasoning
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ClinicalRecommendationResult':
        """Create a ClinicalRecommendationResult from a dictionary"""
        if isinstance(data, cls):
            return data
        return cls(**data)