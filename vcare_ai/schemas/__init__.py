from src.vcare_ai.schemas.clinical import ClinicalRecommendationPayload
from src.vcare_ai.schemas.careplan import RecommendCarePlanPayload

USECASE_SCHEMAS = {
    "generate_clinical_recommendation": ClinicalRecommendationPayload,
    "recommend_careplan": RecommendCarePlanPayload
}