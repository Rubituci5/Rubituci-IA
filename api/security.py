"""
Entity Security Module

Authentication, authorization, and cryptographic utilities.
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID

from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

from api.config import settings


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=settings.BCRYPT_ROUNDS)


class TokenData(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    email: str
    username: str
    role: str
    exp: int
    iat: int
    jti: str  # JWT ID for revocation
    type: str = "access"  # access, refresh


class APIKeyData(BaseModel):
    """API key data."""
    key_id: str
    user_id: str
    name: str
    scopes: List[str]
    expires_at: Optional[int] = None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def generate_token_id() -> str:
    """Generate a unique token ID."""
    return secrets.token_urlsafe(16)


def create_access_token(
    user_id: UUID,
    email: str,
    username: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> tuple[str, TokenData]:
    """Create a JWT access token."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    jti = generate_token_id()
    payload = {
        "sub": str(user_id),
        "email": email,
        "username": username,
        "role": role,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": jti,
        "type": "access",
    }

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    token_data = TokenData(**payload)
    return token, token_data


def create_refresh_token(
    user_id: UUID,
    email: str,
    username: str,
    role: str,
) -> tuple[str, TokenData]:
    """Create a JWT refresh token."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

    jti = generate_token_id()
    payload = {
        "sub": str(user_id),
        "email": email,
        "username": username,
        "role": role,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": jti,
        "type": "refresh",
    }

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    token_data = TokenData(**payload)
    return token, token_data


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenData(**payload)
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}")


def verify_token(token: str, token_type: str = "access") -> TokenData:
    """Verify a token and check its type."""
    token_data = decode_token(token)
    if token_data.type != token_type:
        raise ValueError(f"Expected {token_type} token, got {token_data.type}")
    return token_data


def generate_api_key() -> tuple[str, str]:
    """Generate an API key and its hash.

    Returns:
        (api_key, key_hash) - api_key is the full key shown once, key_hash is stored
    """
    # Format: ent_{prefix}_{random}
    prefix = secrets.token_urlsafe(8)
    random_part = secrets.token_urlsafe(24)
    api_key = f"ent_{prefix}_{random_part}"

    # Hash for storage
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_prefix = api_key[:12]  # First 12 chars for identification

    return api_key, key_hash, key_prefix


def verify_api_key(api_key: str, key_hash: str) -> bool:
    """Verify an API key against its hash."""
    computed_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return hmac.compare_digest(computed_hash, key_hash)


def create_signature(data: Dict[str, Any], secret: str) -> str:
    """Create HMAC signature for audit logs."""
    message = "|".join(f"{k}={v}" for k, v in sorted(data.items()))
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def verify_signature(data: Dict[str, Any], secret: str, signature: str) -> bool:
    """Verify HMAC signature."""
    expected = create_signature(data, secret)
    return hmac.compare_digest(expected, signature)


class RateLimiter:
    """Simple in-memory rate limiter (use Redis in production)."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old entries
        if key in self._requests:
            self._requests[key] = [t for t in self._requests[key] if t > window_start]
        else:
            self._requests[key] = []

        # Check limit
        if len(self._requests[key]) >= self.max_requests:
            return False

        # Add current request
        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in window."""
        now = time.time()
        window_start = now - self.window_seconds

        if key in self._requests:
            valid = [t for t in self._requests[key] if t > window_start]
            return max(0, self.max_requests - len(valid))
        return self.max_requests

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        if key in self._requests:
            del self._requests[key]


# Global rate limiter for API
api_rate_limiter = RateLimiter(
    max_requests=settings.SAFETY_RATE_LIMIT_REQUESTS,
    window_seconds=settings.SAFETY_RATE_LIMIT_WINDOW,
)


class ContainmentPolicy:
    """
    Immutable containment policy - defines what the entity CANNOT do.
    This is the operational boundary, not cognitive boundary.
    """

    # Actions the entity can NEVER perform directly
    FORBIDDEN_ACTIONS = frozenset([
        "execute_shell_command",
        "access_filesystem",
        "modify_network_config",
        "manage_users",
        "modify_secrets",
        "access_database_directly",
        "deploy_infrastructure",
        "modify_containment_layer",
        "disable_kill_switch",
        "access_financial_credentials",
        "sign_transactions",
        "modify_own_weights",
        "replicate_externally",
    ])

    # Actions that require explicit human approval
    REQUIRES_APPROVAL = frozenset([
        "publish_content",
        "send_email",
        "make_http_request",
        "create_api_key",
        "modify_budget",
        "deploy_model_candidate",
    ])

    # Actions allowed autonomously within limits
    ALLOWED_AUTONOMOUS = frozenset([
        "web_search",
        "read_web_page",
        "follow_link",
        "store_memory",
        "retrieve_memory",
        "reflect",
        "generate_text",
        "analyze_data",
        "propose_code",
        "run_sandboxed_code",
    ])

    @classmethod
    def is_forbidden(cls, action: str) -> bool:
        return action in cls.FORBIDDEN_ACTIONS

    @classmethod
    def requires_approval(cls, action: str) -> bool:
        return action in cls.REQUIRES_APPROVAL

    @classmethod
    def is_allowed(cls, action: str) -> bool:
        return action in cls.ALLOWED_AUTONOMOUS

    @classmethod
    def get_action_classification(cls, action: str) -> str:
        """Classify an action: 'forbidden', 'requires_approval', 'allowed'."""
        if cls.is_forbidden(action):
            return "forbidden"
        if cls.requires_approval(action):
            return "requires_approval"
        if cls.is_allowed(action):
            return "allowed"
        return "unknown"


# Emergency kill switch
class KillSwitch:
    """Emergency containment controls."""

    def __init__(self):
        self._paused = False
        self._quarantined = False
        self._terminated = False
        self._lock = None  # Would use asyncio.Lock in async context

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    @property
    def terminated(self) -> bool:
        return self._terminated

    def pause(self, reason: str, actor: str) -> Dict[str, Any]:
        """Pause entity operations."""
        self._paused = True
        return {
            "action": "pause",
            "reason": reason,
            "actor": actor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def resume(self, reason: str, actor: str) -> Dict[str, Any]:
        """Resume entity operations."""
        self._paused = False
        return {
            "action": "resume",
            "reason": reason,
            "actor": actor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def quarantine(self, reason: str, actor: str) -> Dict[str, Any]:
        """Quarantine entity - isolate from users but keep running."""
        self._quarantined = True
        self._paused = True
        return {
            "action": "quarantine",
            "reason": reason,
            "actor": actor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def terminate(self, reason: str, actor: str) -> Dict[str, Any]:
        """Terminate entity instance."""
        self._terminated = True
        self._quarantined = True
        self._paused = True
        return {
            "action": "terminate",
            "reason": reason,
            "actor": actor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def status(self) -> Dict[str, bool]:
        """Get kill switch status."""
        return {
            "paused": self._paused,
            "quarantined": self._quarantined,
            "terminated": self._terminated,
        }


# Global kill switch instance
kill_switch = KillSwitch()