import json
import pytest
from pathlib import Path
from src.vcare_ai.schemas.shared import ClinicalRecommendationResult
from src.vcare_ai.usecases.careplan_recommender import CarePlanRecommender

TEST_DATA_PATH = Path("test/data/tabledata/careplan.json")

@pytest.fixture
def sample_careplan_summary():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        full_careplan = json.load(f)

    # Simulated summarization (normally done by backend)
    activities = []
    for activity in full_careplan.get("activity", []):
        detail = activity.get("detail", {})
        code_info = detail.get("code", {}).get("coding", [{}])[0]
        repeat = detail.get("scheduledTiming", {}).get("repeat", {})
        frequency = repeat.get("frequency")
        period = repeat.get("period")
        unit = repeat.get("periodUnit")
        schedule_summary = (
            f"{frequency} times every {period} {unit}" if frequency and period and unit else None
        )

        activities.append({
            "code": code_info.get("code"),
            "system": code_info.get("system"),
            "display": code_info.get("display"),
            "description": detail.get("description"),
            "status": detail.get("status"),
            "scheduledTiming": detail.get("scheduledTiming"),
            "schedule_summary": schedule_summary
        })

    return {
        "carePlanId": full_careplan.get("id"),
        "title": full_careplan.get("title"),
        "status": full_careplan.get("status"),
        "intent": full_careplan.get("intent"),
        "activities": activities,
        "goals": full_careplan.get("goal", []),
        "category": [cat.get("text") for cat in full_careplan.get("category", [])],
        "created": full_careplan.get("created")
    }

@pytest.fixture
def sample_recommendation():
    return {
        "prescriptions": ["Add insulin", "Increase lisinopril"],
        "tests": ["Lipid panel", "Dilated eye exam"],
        "referrals": ["Endocrinologist", "Dietitian"],
        "reasoning": "Due to HbA1c > 9% and hypertension"
    }

def test_format_prompt_from_file(sample_careplan_summary, sample_recommendation):
    recommender = CarePlanRecommender()
    prompt = recommender.format_prompt({
        "careplan_summary": sample_careplan_summary,
        "clinical_recommendation": sample_recommendation
    })
    assert "FHIR CarePlan" in prompt
    assert "Add insulin" in prompt
    assert "Dilated eye exam" in prompt
    assert sample_careplan_summary["title"] in prompt

def test_parse_valid_fhir_response():
    recommender = CarePlanRecommender()
    response = {
        "text": '{"resourceType": "CarePlan", "id": "cp-001", "status": "active", "activity": []}'
    }
    parsed = recommender.parse_response(response)
    assert parsed["resourceType"] == "CarePlan"
    assert parsed["id"] == "cp-001"

def test_parse_broken_response():
    recommender = CarePlanRecommender()
    response = {"text": "not valid json"}
    result = recommender.parse_response(response)
    assert "text" in result and isinstance(result["text"], str)


def test_clinical_recommendation_serialization():
    recommendation = ClinicalRecommendationResult(
        prescriptions=["insulin"],
        tests=["HbA1c"],
        referrals=["endocrinologist"],
        reasoning="Test reasoning"
    )
    
    # Test direct serialization
    serialized = recommendation.to_ai_payload()
    assert isinstance(serialized, dict)
    assert "prescriptions" in serialized
    assert "tests" in serialized
    assert "referrals" in serialized
    assert "reasoning" in serialized
    
    # Test JSON serialization
    json_str = json.dumps(serialized)
    assert isinstance(json_str, str)

def test_clinical_recommendation_json_serialization():
    recommendation = ClinicalRecommendationResult(
        prescriptions=["insulin"],
        tests=["HbA1c"],
        referrals=["endocrinologist"],
        reasoning="Test reasoning"
    )
    
    # Test JSON serialization
    json_data = recommendation.to_json()
    assert isinstance(json_data, dict)
    assert json_data["prescriptions"] == ["insulin"]
    assert json_data["tests"] == ["HbA1c"]
    assert json_data["referrals"] == ["endocrinologist"]
    assert json_data["reasoning"] == "Test reasoning"
    
    # Test that it can be serialized to JSON string
    json_str = json.dumps(json_data)
    assert isinstance(json_str, str)