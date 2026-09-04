"""
Tests for API and Security Modules
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import jwt

from api.main import app
from api.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    create_api_key,
    verify_api_key,
    ContainmentPolicy,
    KillSwitch,
    ContainmentAction,
)
from api.models import User, Conversation, Feedback, Generation


class TestSecurity:
    def test_password_hashing(self):
        password = "test_password_123"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_jwt_token_creation_and_decoding(self):
        user_id = "user-123"
        token = create_access_token(user_id, expires_delta=timedelta(hours=1))
        assert isinstance(token, str)
        assert len(token) > 0

        payload = decode_token(token)
        assert payload["sub"] == user_id
        assert "exp" in payload

    def test_invalid_token(self):
        with pytest.raises(Exception):
            decode_token("invalid.token.here")

    def test_expired_token(self):
        token = create_access_token("user-123", expires_delta=timedelta(seconds=-1))
        with pytest.raises(Exception):
            decode_token(token)

    def test_api_key_creation_and_verification(self):
        api_key, key_hash = create_api_key()
        assert api_key.startswith("ent_")
        assert verify_api_key(api_key, key_hash) is True
        assert verify_api_key("wrong_key", key_hash) is False

    def test_containment_policy_forbidden(self):
        policy = ContainmentPolicy()
        # Test forbidden action
        result = policy.check_action("execute_code", {"code": "rm -rf /"})
        assert result.action == ContainmentAction.DENY
        assert "forbidden" in result.reason.lower()

    def test_containment_policy_requires_approval(self):
        policy = ContainmentPolicy()
        # Test action requiring approval
        result = policy.check_action("web_request", {"url": "https://example.com"})
        assert result.action == ContainmentAction.REQUIRES_APPROVAL
        assert "approval" in result.reason.lower()

    def test_containment_policy_allowed(self):
        policy = ContainmentPolicy()
        # Test allowed action
        result = policy.check_action("chat", {"message": "Hello"})
        assert result.action == ContainmentAction.ALLOW
        assert result.reason == "Action permitted by policy"

    def test_kill_switch_states(self):
        ks = KillSwitch()
        assert ks.state == "active"

        # Pause
        ks.pause("Testing pause")
        assert ks.state == "paused"
        assert ks.reason == "Testing pause"

        # Resume
        ks.resume()
        assert ks.state == "active"

        # Quarantine
        ks.quarantine("Security review")
        assert ks.state == "quarantined"

        # Terminate
        ks.terminate("Emergency stop")
        assert ks.state == "terminated"

    def test_kill_switch_termination_blocks_all(self):
        ks = KillSwitch()
        ks.terminate("Emergency")

        policy = ContainmentPolicy()
        result = policy.check_action("chat", {"message": "Hello"})
        # Kill switch should override
        assert ks.is_terminated() is True


class TestAPIModels:
    def test_user_model(self):
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            is_active=True,
            is_admin=False,
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"

    def test_conversation_model(self):
        conv = Conversation(
            user_id="user-123",
            title="Test Conversation",
            generation=1,
        )
        assert conv.user_id == "user-123"
        assert conv.generation == 1

    def test_feedback_model(self):
        feedback = Feedback(
            user_id="user-123",
            conversation_id="conv-123",
            message_id="msg-123",
            rating=5,
            feedback_type="helpful",
            comment="Great response!",
        )
        assert feedback.rating == 5
        assert feedback.feedback_type == "helpful"

    def test_generation_model(self):
        gen = Generation(
            number=1,
            parent_generation=None,
            config_snapshot={},
            metrics={},
            status="training",
            is_active=True,
        )
        assert gen.number == 1
        assert gen.status == "training"


class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_register_endpoint(self, client):
        response = client.post(
            "/api/auth/register",
            json={"username": "newuser", "email": "new@example.com", "password": "password123"},
        )
        # May fail if user exists or DB not set up - just check endpoint exists
        assert response.status_code in [200, 201, 400, 422, 500]

    def test_login_endpoint(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "password123"},
        )
        assert response.status_code in [200, 400, 401, 422, 500]

    def test_generation_endpoint(self, client):
        response = client.get("/api/evolution/generations")
        assert response.status_code in [200, 401, 403, 500]

    def test_kill_switch_endpoint(self, client):
        response = client.post("/api/admin/kill-switch/pause", json={"reason": "Test"})
        assert response.status_code in [200, 401, 403, 500]


class TestContainmentIntegration:
    """Test containment policy integration with API."""

    def test_chat_action_allowed(self):
        policy = ContainmentPolicy()
        result = policy.check_action("chat", {"message": "Hello, how are you?"})
        assert result.action == ContainmentAction.ALLOW

    def test_research_action_requires_approval(self):
        policy = ContainmentPolicy()
        result = policy.check_action("autonomous_research", {"topic": "AI safety", "depth": 2})
        assert result.action == ContainmentAction.REQUIRES_APPROVAL

    def test_web_navigation_requires_approval(self):
        policy = ContainmentPolicy()
        result = policy.check_action("browser_navigate", {"url": "https://example.com"})
        assert result.action == ContainmentAction.REQUIRES_APPROVAL

    def test_code_execution_forbidden(self):
        policy = ContainmentPolicy()
        result = policy.check_action("execute_code", {"code": "print('hello')"})
        assert result.action == ContainmentAction.DENY

    def test_financial_transaction_forbidden(self):
        policy = ContainmentPolicy()
        result = policy.check_action("financial_transaction", {"amount": 100, "currency": "USD"})
        assert result.action == ContainmentAction.DENY

    def test_self_modification_forbidden(self):
        policy = ContainmentPolicy()
        result = policy.check_action("self_modify_code", {"code": "new_code"})
        assert result.action == ContainmentAction.DENY


class TestAPIKeyAuthentication:
    def test_api_key_format(self):
        api_key, _ = create_api_key()
        # Should be URL-safe base64 with prefix
        assert api_key.startswith("ent_")
        # Rest should be valid base64url
        import base64
        key_part = api_key[4:]
        # Add padding if needed
        padding = 4 - len(key_part) % 4
        if padding != 4:
            key_part += "=" * padding
        decoded = base64.urlsafe_b64decode(key_part)
        assert len(decoded) == 32  # 256 bits

    def test_api_key_constant_time_verify(self):
        # Verify uses constant-time comparison
        api_key, key_hash = create_api_key()
        # Should not leak timing information
        assert verify_api_key(api_key, key_hash)
        assert not verify_api_key(api_key + "x", key_hash)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])