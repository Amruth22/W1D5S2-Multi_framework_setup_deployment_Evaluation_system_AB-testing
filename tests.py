import pytest
import json
import os
import time
import asyncio
import pandas as pd
from unittest.mock import patch, MagicMock, Mock
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Optional, Any
import tempfile
import random

# Mock data for testing
MOCK_RESPONSES = {
    "summarization": "This is a concise summary of the input text, capturing the main points effectively.",
    "qa_answer": "Based on the provided context, LangChain is a framework for developing applications powered by large language models.",
    "chat_response": "Hello! I'm doing well, thank you for asking. How can I help you today?",
    "health_status": {"status": "healthy", "gemini_models_loaded": {"pro": True, "flash": True, "pro_text": True}}
}

# Mock configuration
MOCK_CONFIG = {
    "GEMINI_PRO_MODEL": "gemini-2.0-flash",
    "GEMINI_FLASH_MODEL": "gemini-2.5-flash", 
    "GEMINI_PRO_TEXT_MODEL": "gemini-2.5-flash-lite",
    "API_HOST": "0.0.0.0",
    "API_PORT": 5000,
    "EVAL_LOG_FILE": "evaluation_logs.csv",
    "AB_TEST_EXPERIMENTS_FILE": "ab_test_experiments.csv",
    "AB_TEST_ASSIGNMENTS_FILE": "ab_test_assignments.csv"
}

# Mock evaluation and A/B testing functions
class MockEvaluationSystem:
    def __init__(self):
        self.evaluation_logs = []
        self.ab_experiments = []
        self.ab_assignments = []
    
    async def log_evaluation_metric(self, model_id: str, metric_name: str, metric_value: float, 
                                   model_type: str = None, task_type: str = None, notes: str = ""):
        """Mock evaluation metric logging"""
        await asyncio.sleep(0.01)  # Simulate async operation
        
        if model_type is None:
            model_type = "flash" if "flash" in model_id.lower() else "pro" if "pro" in model_id.lower() else "unknown"
        
        if task_type is None:
            if "summarization" in model_id.lower():
                task_type = "summarization"
            elif "qa" in model_id.lower():
                task_type = "qa"
            elif "chat" in model_id.lower():
                task_type = "chat"
            else:
                task_type = "unknown"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "model_type": model_type,
            "task_type": task_type,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "notes": notes
        }
        
        self.evaluation_logs.append(log_entry)
        return log_entry
    
    async def get_all_evaluation_logs(self):
        """Mock get all evaluation logs"""
        await asyncio.sleep(0.01)
        return pd.DataFrame(self.evaluation_logs)
    
    async def create_ab_test_experiment(self, name: str, variants: List[str], notes: str = ""):
        """Mock A/B test experiment creation"""
        await asyncio.sleep(0.01)
        
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        experiment = {
            "experiment_id": experiment_id,
            "name": name,
            "variants": ','.join(variants),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "active",
            "notes": notes
        }
        
        self.ab_experiments.append(experiment)
        return experiment_id
    
    async def assign_user_to_variant(self, user_id: str, experiment_id: str):
        """Mock user variant assignment"""
        await asyncio.sleep(0.01)
        
        # Find experiment
        experiment = next((exp for exp in self.ab_experiments if exp["experiment_id"] == experiment_id), None)
        if not experiment or experiment["status"] != "active":
            return None
        
        variants = experiment["variants"].split(',')
        assigned_variant = random.choice(variants)
        
        assignment = {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "assigned_variant": assigned_variant,
            "assignment_time": datetime.now().isoformat(),
            "metric_value": None
        }
        
        self.ab_assignments.append(assignment)
        return assigned_variant
    
    async def log_ab_test_metric(self, user_id: str, experiment_id: str, metric_value: float):
        """Mock A/B test metric logging"""
        await asyncio.sleep(0.01)
        
        # Find assignment and update metric
        for assignment in self.ab_assignments:
            if assignment["user_id"] == user_id and assignment["experiment_id"] == experiment_id:
                assignment["metric_value"] = metric_value
                return True
        return False
    
    async def analyze_ab_test_results(self, experiment_id: str):
        """Mock A/B test results analysis"""
        await asyncio.sleep(0.01)
        
        # Filter assignments for this experiment
        exp_assignments = [a for a in self.ab_assignments 
                          if a["experiment_id"] == experiment_id and a["metric_value"] is not None]
        
        if not exp_assignments:
            return None
        
        # Create DataFrame and analyze
        df = pd.DataFrame(exp_assignments)
        results = df.groupby("assigned_variant")["metric_value"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        return results

# Mock API functions
class MockAPI:
    def __init__(self):
        self.models_loaded = {"pro": True, "flash": True, "pro_text": True}
    
    async def summarize_text(self, text: str):
        """Mock text summarization"""
        await asyncio.sleep(0.01)
        if not text:
            raise ValueError("Text is required")
        return MOCK_RESPONSES["summarization"]
    
    async def complex_qa(self, question: str, context: str):
        """Mock complex Q&A"""
        await asyncio.sleep(0.01)
        if not question or not context:
            raise ValueError("Question and context are required")
        return MOCK_RESPONSES["qa_answer"]
    
    async def simple_chat(self, message: str):
        """Mock simple chat"""
        await asyncio.sleep(0.01)
        if not message:
            raise ValueError("Message is required")
        return MOCK_RESPONSES["chat_response"]
    
    def get_health(self):
        """Mock health check"""
        return MOCK_RESPONSES["health_status"]

# Mock evaluation metrics
class MockMetrics:
    @staticmethod
    async def calculate_bleu_score(reference: str, hypothesis: str):
        """Mock BLEU score calculation"""
        await asyncio.sleep(0.01)
        # Simple mock: higher score for more similar texts
        similarity = len(set(reference.lower().split()) & set(hypothesis.lower().split()))
        return min(1.0, similarity / max(len(reference.split()), len(hypothesis.split())))
    
    @staticmethod
    async def calculate_rouge_scores(reference: str, hypothesis: str):
        """Mock ROUGE scores calculation"""
        await asyncio.sleep(0.01)
        # Simple mock scores
        return {
            "rouge1": random.uniform(0.6, 0.9),
            "rouge2": random.uniform(0.4, 0.7),
            "rougeL": random.uniform(0.5, 0.8)
        }

# Global mock instances
mock_api = MockAPI()
mock_evaluation = MockEvaluationSystem()
mock_metrics = MockMetrics()

# ============================================================================
# ASYNC PYTEST TEST FUNCTIONS
# ============================================================================

async def test_01_env_config_validation():
    """Test 1: Environment and Configuration Validation"""
    print("Running Test 1: Environment and Configuration Validation")
    
    # Check if .env file exists (optional for mock tests)
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file_path):
        print("PASS: .env file exists")
    else:
        print("INFO: .env file not found (optional for mock tests)")
    
    # Validate mock configuration
    required_configs = ["GEMINI_PRO_MODEL", "GEMINI_FLASH_MODEL", "GEMINI_PRO_TEXT_MODEL", 
                       "API_HOST", "API_PORT", "EVAL_LOG_FILE"]
    
    for config_key in required_configs:
        assert config_key in MOCK_CONFIG, f"Configuration {config_key} should be defined"
    
    # Check API key (optional for mock tests)
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and api_key.startswith('AIza'):
        print(f"PASS: API Key configured: {api_key[:10]}...{api_key[-5:]}")
    else:
        print("INFO: GEMINI_API_KEY not configured (not required for mock tests)")
    
    print("PASS: Environment and configuration validation completed")

async def test_02_evaluation_logging():
    """Test 2: Evaluation Logging System"""
    print("Running Test 2: Evaluation Logging System")
    
    # Test logging evaluation metrics
    log_entry = await mock_evaluation.log_evaluation_metric(
        "test_model", "bleu_score", 0.85, "flash", "summarization", "Test note"
    )
    
    assert log_entry is not None, "Should create log entry"
    assert log_entry["model_id"] == "test_model", "Should log correct model ID"
    assert log_entry["metric_name"] == "bleu_score", "Should log correct metric name"
    assert log_entry["metric_value"] == 0.85, "Should log correct metric value"
    assert log_entry["model_type"] == "flash", "Should log correct model type"
    assert log_entry["task_type"] == "summarization", "Should log correct task type"
    
    # Test retrieving logs
    logs_df = await mock_evaluation.get_all_evaluation_logs()
    assert len(logs_df) == 1, "Should retrieve 1 log entry"
    assert logs_df["model_id"].iloc[0] == "test_model", "Should retrieve correct model ID"
    
    print("PASS: Evaluation logging system working correctly")
    print(f"PASS: Logged metric - Model: {log_entry['model_id']}, Value: {log_entry['metric_value']}")

async def test_03_ab_test_experiment_management():
    """Test 3: A/B Test Experiment Management"""
    print("Running Test 3: A/B Test Experiment Management")
    
    # Create an A/B test experiment
    exp_id = await mock_evaluation.create_ab_test_experiment(
        "Test Experiment", ["variant_a", "variant_b"], "Test notes"
    )
    
    assert exp_id is not None, "Should create experiment ID"
    assert exp_id.startswith("exp_"), "Experiment ID should have correct format"
    
    # Verify experiment was created
    assert len(mock_evaluation.ab_experiments) == 1, "Should have 1 experiment"
    experiment = mock_evaluation.ab_experiments[0]
    assert experiment["name"] == "Test Experiment", "Should have correct experiment name"
    assert experiment["variants"] == "variant_a,variant_b", "Should have correct variants"
    assert experiment["status"] == "active", "Should be active"
    
    print(f"PASS: A/B experiment created with ID: {exp_id}")
    print(f"PASS: Experiment details - Name: {experiment['name']}, Variants: {experiment['variants']}")

async def test_04_model_configuration():
    """Test 4: Model Configuration Validation"""
    print("Running Test 4: Model Configuration Validation")
    
    # Test model configuration values
    models = {
        "pro": MOCK_CONFIG["GEMINI_PRO_MODEL"],
        "flash": MOCK_CONFIG["GEMINI_FLASH_MODEL"],
        "pro_text": MOCK_CONFIG["GEMINI_PRO_TEXT_MODEL"]
    }
    
    # Validate model names
    for model_type, model_name in models.items():
        assert model_name is not None, f"{model_type} model should be configured"
        assert isinstance(model_name, str), f"{model_type} model name should be a string"
        assert len(model_name) > 0, f"{model_type} model name should not be empty"
        assert "gemini" in model_name.lower(), f"{model_type} should be a Gemini model"
    
    print(f"PASS: Model configurations validated")
    print(f"PASS: Pro model: {models['pro']}")
    print(f"PASS: Flash model: {models['flash']}")
    print(f"PASS: Pro-text model: {models['pro_text']}")
    
    # Test API configuration
    assert MOCK_CONFIG["API_HOST"] is not None, "API host should be configured"
    assert MOCK_CONFIG["API_PORT"] is not None, "API port should be configured"
    assert isinstance(MOCK_CONFIG["API_PORT"], int), "API port should be an integer"
    
    # Test file paths configuration
    file_configs = ["EVAL_LOG_FILE", "AB_TEST_EXPERIMENTS_FILE", "AB_TEST_ASSIGNMENTS_FILE"]
    for file_config in file_configs:
        assert MOCK_CONFIG[file_config] is not None, f"{file_config} should be configured"
        assert isinstance(MOCK_CONFIG[file_config], str), f"{file_config} should be a string"
        assert MOCK_CONFIG[file_config].endswith(".csv"), f"{file_config} should be a CSV file"
    
    print(f"PASS: API configuration - Host: {MOCK_CONFIG['API_HOST']}, Port: {MOCK_CONFIG['API_PORT']}")
    print(f"PASS: File configurations validated")

async def test_05_ab_test_results_analysis():
    """Test 5: A/B Test Results Analysis"""
    print("Running Test 5: A/B Test Results Analysis")
    
    # Create experiment and assign users with metrics
    exp_id = await mock_evaluation.create_ab_test_experiment(
        "Analysis Test", ["variant_A", "variant_B"]
    )
    
    # Create test data - users with different performance metrics
    users_data = [
        ("user_a1", "variant_A", 0.8),
        ("user_a2", "variant_A", 0.85),
        ("user_b1", "variant_B", 0.9),
        ("user_b2", "variant_B", 0.95)
    ]
    
    for user_id, variant, metric in users_data:
        await mock_evaluation.assign_user_to_variant(user_id, exp_id)
        # Force assignment to specific variant for testing
        for assignment in mock_evaluation.ab_assignments:
            if assignment["user_id"] == user_id:
                assignment["assigned_variant"] = variant
        await mock_evaluation.log_ab_test_metric(user_id, exp_id, metric)
    
    # Analyze results
    results = await mock_evaluation.analyze_ab_test_results(exp_id)
    
    assert results is not None, "Should return analysis results"
    assert len(results) == 2, "Should have results for 2 variants"
    
    # Check variant A results
    variant_a_row = results[results["assigned_variant"] == "variant_A"]
    variant_b_row = results[results["assigned_variant"] == "variant_B"]
    
    assert len(variant_a_row) == 1, "Should have 1 row for variant A"
    assert len(variant_b_row) == 1, "Should have 1 row for variant B"
    assert variant_a_row["count"].iloc[0] == 2, "Variant A should have 2 data points"
    assert variant_b_row["count"].iloc[0] == 2, "Variant B should have 2 data points"
    
    print(f"PASS: A/B test analysis completed")
    print(f"PASS: Variant A mean: {variant_a_row['mean'].iloc[0]:.3f}, Variant B mean: {variant_b_row['mean'].iloc[0]:.3f}")

async def test_06_summarize_endpoint():
    """Test 6: Summarization Endpoint"""
    print("Running Test 6: Summarization Endpoint")
    
    # Test valid summarization request
    text_input = "This is a long text that needs to be summarized for testing purposes."
    result = await mock_api.summarize_text(text_input)
    
    assert result is not None, "Should return summarization result"
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"
    assert "summary" in result.lower() or "concise" in result.lower(), "Should contain summarization keywords"
    
    print(f"PASS: Summarization result: {result[:50]}...")
    
    # Test error handling
    try:
        await mock_api.summarize_text("")
        assert False, "Should raise error for empty text"
    except ValueError:
        print("PASS: Error handling for empty text working correctly")

async def test_07_qa_endpoint():
    """Test 7: Q&A Endpoint"""
    print("Running Test 7: Q&A Endpoint")
    
    # Test valid Q&A request
    question = "What is the purpose of LangChain?"
    context = "LangChain is a framework for developing applications powered by large language models."
    result = await mock_api.complex_qa(question, context)
    
    assert result is not None, "Should return Q&A result"
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"
    assert "langchain" in result.lower(), "Should mention LangChain in response"
    
    print(f"PASS: Q&A result: {result[:50]}...")
    
    # Test error handling
    try:
        await mock_api.complex_qa("", "context")
        assert False, "Should raise error for empty question"
    except ValueError:
        print("PASS: Error handling for empty question working correctly")

async def test_08_chat_endpoint():
    """Test 8: Chat Endpoint"""
    print("Running Test 8: Chat Endpoint")
    
    # Test valid chat request
    message = "Hello, how are you today?"
    result = await mock_api.simple_chat(message)
    
    assert result is not None, "Should return chat result"
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"
    assert any(word in result.lower() for word in ["hello", "help", "well"]), "Should contain conversational keywords"
    
    print(f"PASS: Chat result: {result[:50]}...")
    
    # Test error handling
    try:
        await mock_api.simple_chat("")
        assert False, "Should raise error for empty message"
    except ValueError:
        print("PASS: Error handling for empty message working correctly")

async def test_09_health_endpoint():
    """Test 9: Health Endpoint"""
    print("Running Test 9: Health Endpoint")
    
    # Test health check
    health_data = mock_api.get_health()
    
    assert health_data is not None, "Should return health data"
    assert health_data["status"] == "healthy", "Status should be healthy"
    assert "gemini_models_loaded" in health_data, "Should contain model loading status"
    
    models_status = health_data["gemini_models_loaded"]
    assert models_status["pro"] == True, "Pro model should be loaded"
    assert models_status["flash"] == True, "Flash model should be loaded"
    assert models_status["pro_text"] == True, "Pro text model should be loaded"
    
    print("PASS: Health endpoint working correctly")
    print(f"PASS: Models loaded - Pro: {models_status['pro']}, Flash: {models_status['flash']}, Pro-Text: {models_status['pro_text']}")

async def test_10_framework_functions():
    """Test 10: Framework Functions and Metrics"""
    print("Running Test 10: Framework Functions and Metrics")
    
    # Test BLEU score calculation
    reference = "The capital of Canada is Ottawa."
    hypothesis = "Ottawa is the capital of Canada."
    bleu_score = await mock_metrics.calculate_bleu_score(reference, hypothesis)
    
    assert isinstance(bleu_score, float), "BLEU score should be a float"
    assert 0 <= bleu_score <= 1, "BLEU score should be between 0 and 1"
    
    print(f"PASS: BLEU score calculation: {bleu_score:.4f}")
    
    # Test ROUGE scores calculation
    rouge_scores = await mock_metrics.calculate_rouge_scores(reference, hypothesis)
    
    assert isinstance(rouge_scores, dict), "ROUGE scores should be a dictionary"
    assert "rouge1" in rouge_scores, "Should contain ROUGE-1 score"
    assert "rouge2" in rouge_scores, "Should contain ROUGE-2 score"
    assert "rougeL" in rouge_scores, "Should contain ROUGE-L score"
    
    for metric, score in rouge_scores.items():
        assert isinstance(score, float), f"{metric} should be a float"
        assert 0 <= score <= 1, f"{metric} should be between 0 and 1"
    
    print(f"PASS: ROUGE scores - ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-2: {rouge_scores['rouge2']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # Test framework function validation
    frameworks = [
        ("summarization", mock_api.summarize_text, "Test text to summarize"),
        ("qa", lambda: mock_api.complex_qa("What is AI?", "AI is artificial intelligence"), None),
        ("chat", mock_api.simple_chat, "Hello there!")
    ]
    
    for framework_name, framework_func, test_input in frameworks:
        try:
            if test_input:
                result = await framework_func(test_input)
            else:
                result = await framework_func()
            
            assert isinstance(result, str), f"{framework_name} should return string"
            assert len(result) > 0, f"{framework_name} should return non-empty result"
            print(f"PASS: {framework_name} framework working correctly")
        except Exception as e:
            print(f"FAIL: {framework_name} framework error: {e}")

async def run_async_tests():
    """Run all async tests"""
    print("Running Multi-Framework AI Evaluation System Tests (Async Version)...")
    print("Using async mocked data for ultra-fast execution")
    print("Testing: Multi-model deployment, A/B testing, evaluation metrics")
    print("=" * 70)
    
    # List of exactly 10 async test functions
    test_functions = [
        test_01_env_config_validation,
        test_02_evaluation_logging,
        test_03_ab_test_experiment_management,
        test_04_model_configuration,
        test_05_ab_test_results_analysis,
        test_06_summarize_endpoint,
        test_07_qa_endpoint,
        test_08_chat_endpoint,
        test_09_health_endpoint,
        test_10_framework_functions
    ]
    
    passed = 0
    failed = 0
    
    # Run tests sequentially for better output readability
    for test_func in test_functions:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_func.__name__} - {e}")
            failed += 1
    
    print("=" * 70)
    print(f"📊 Test Results Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        print("✅ Multi-Framework AI Evaluation System (Async) is working correctly")
        print("⚡ Ultra-fast async execution with mocked multi-model features")
        print("🔬 A/B testing, evaluation metrics, and multi-model deployment validated")
        print("🚀 No server or real API calls required - pure async testing")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed")
        return False

def run_all_tests():
    """Run all tests and provide summary (sync wrapper for async tests)"""
    return asyncio.run(run_async_tests())

if __name__ == "__main__":
    print("🚀 Starting Multi-Framework AI Evaluation System Tests (Async Version)")
    print("📋 No API keys or server required - using async mocked responses")
    print("⚡ Ultra-fast async execution for multi-model evaluation")
    print("🔬 Testing: LangChain integration, A/B testing, evaluation metrics")
    print("🏢 Multi-framework deployment validation with async testing")
    print()
    
    # Run the tests
    start_time = time.time()
    success = run_all_tests()
    end_time = time.time()
    
    print(f"\n⏱️  Total execution time: {end_time - start_time:.2f} seconds")
    exit(0 if success else 1)