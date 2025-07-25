import json
import sys
import os
import logging

# Ensure the functions directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../functions')))

from functions.Stage_AiRecommendation import lambda_handler

logger = logging.getLogger(__name__)

def test_lambda_handler_valid_input():
    # Mock event for generate_clinical_recommendation
    event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "usecase": "generate_clinical_recommendation",
            "pg_payload": {
                "age": 45,
                "conditions": ["hypertension", "diabetes"],
                "lab_results": {
                    "HbA1c": {"value": 7.2, "unit": "%"},
                    "LDL": {"value": 120, "unit": "mg/dL"}
                },
                "medications": ["metformin", "lisinopril"],
                "use_template": None,
                "disable_cache": False
            }
        })
    }
    context = None  # You can mock context if needed

    response = lambda_handler(event, context)
    assert isinstance(response, dict)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "result" in body
    assert "request_id" in body

def test_lambda_handler_missing_usecase():
    event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "pg_payload": {
                "age": 45,
                "conditions": ["hypertension"],
                "lab_results": {"HbA1c": {"value": 7.2, "unit": "%"}},
                "medications": []
            }
        })
    }
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body

def test_lambda_handler_invalid_payload():
    event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "usecase": "generate_clinical_recommendation",
            "pg_payload": {
                "age": -1,  # Invalid age
                "conditions": [],
                "lab_results": {},
                "medications": []
            }
        })
    }
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body
    assert "details" in body

def test_lambda_handler_recommend_careplan(monkeypatch):
    # Mock event for recommend_careplan
    event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "usecase": "recommend_careplan",
            "pg_payload": {
                "patientId": "test-patient-id",
                "carePlanId": "test-careplan-id",
                "clinical_recommendation": {
                    "prescriptions": ["insulin glargine", "metformin"],
                    "tests": ["HbA1c", "Lipid Panel"],
                    "referrals": ["endocrinologist"],
                    "reasoning": "Given elevated HbA1c and cardiovascular risk factors"
                }
            }
        })
    }

    # Mock DynamoDBClient.get_careplan_summary
    class MockDynamoDBClient:
        def get_careplan_summary(self, careplan_id):
            return {
                "carePlanId": careplan_id,
                "title": "Test Plan",
                "activities": [
                    {"code": "12345-6", "description": "Blood Pressure"}
                ]
            }
        def update_careplan(self, patient_id, careplan):
            return {"status": "success"}

    # Mock BedrockClient.invoke
    class MockBedrockClient:
        def invoke(self, prompt, use_cache=True):
            return {
                "text": json.dumps({
                    "resourceType": "CarePlan",
                    "id": "new-id",
                    "status": "active",
                    "intent": "plan",
                    "subject": {"reference": "Patient/test-patient-id"},
                    "activity": [
                        {
                            "detail": {
                                "code": {"text": "Blood Pressure Monitoring"},
                                "status": "scheduled",
                                "scheduledTiming": {"frequency": "daily"}
                            }
                        }
                    ]
                })
            }

    # Patch imports in the lambda handler
    monkeypatch.setattr("src.vcare_py.DynamoDBClient.DynamoDBClient", MockDynamoDBClient)
    monkeypatch.setattr("src.vcare_ai.client.BedrockClient", MockBedrockClient)

    response = lambda_handler(event, None)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "new_careplan" in body
    assert "resourceType" in body["new_careplan"]

def test_combined_recommendation_to_careplan(monkeypatch):
    """Test the end-to-end flow from clinical recommendation to careplan generation"""
    from functions.Stage_AiRecommendation import lambda_handler
    from src.vcare_ai.schemas.shared import ClinicalRecommendationResult

    # Step 1: Generate Clinical Recommendation
    clinical_event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "usecase": "generate_clinical_recommendation",
            "pg_payload": {
                "age": 60,
                "conditions": ["type 2 diabetes", "hypertension"],
                "lab_results": {
                    "hba1c": 9.2,
                    "fasting_glucose": 110,
                    "blood_pressure_systolic": 145,
                    "blood_pressure_diastolic": 90,
                    "ldl_cholesterol": 110,
                    "hdl_cholesterol": 38,
                    "egfr": 68
                },
                "medications": ["metformin 500mg twice daily", "lisinopril 10mg daily"]
            }
        })
    }

    # Mock responses
    class MockBedrockClient:
        def invoke(self, prompt, use_cache=True):
            if "CarePlan" in prompt:
                return {
                    "text": json.dumps({
                        "resourceType": "CarePlan",
                        "id": "new-id",
                        "status": "active",
                        "intent": "plan",
                        "subject": {"reference": "Patient/test-patient-id"},
                        "activity": [
                            {
                                "detail": {
                                    "code": {"text": "Blood Pressure Monitoring"},
                                    "status": "scheduled",
                                    "scheduledTiming": {"frequency": "daily"}
                                }
                            }
                        ]
                    })
                }
            else:
                return {
                    "text": json.dumps({
                        "prescriptions": ["insulin glargine", "metformin"],
                        "tests": ["HbA1c", "Lipid Panel"],
                        "referrals": ["endocrinologist"],
                        "reasoning": "Given elevated HbA1c and cardiovascular risk factors"
                    })
                }

    # Apply mocks
    monkeypatch.setattr("src.vcare_ai.client.BedrockClient", MockBedrockClient)

    # Step 1: Get Clinical Recommendation
    clinical_response = lambda_handler(clinical_event, None)
    assert clinical_response["statusCode"] == 200
    clinical_body = json.loads(clinical_response["body"])
    assert "result" in clinical_body

    # Step 2: Generate CarePlan
    careplan_event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "usecase": "recommend_careplan",
            "pg_payload": {
                "patientId": "test-patient-id",
                "carePlanId": "test-careplan-id",
                "clinical_recommendation": clinical_body["result"]
            }
        })
    }

    careplan_response = lambda_handler(careplan_event, None)
    assert careplan_response["statusCode"] == 200
    careplan_body = json.loads(careplan_response["body"])
    assert "new_careplan" in careplan_body
    assert "resourceType" in careplan_body["new_careplan"]
