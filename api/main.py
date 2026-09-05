"""
Entity API - Main FastAPI Application

Entry point for the Entity backend API.
"""

import uuid
import secrets
import hashlib
import re
import json
import asyncio
import ipaddress
import unicodedata
import base64
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import httpx
from jose import jwt
from urllib.parse import urlencode, urljoin, urlparse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text

from api.config import settings
from api.database import db_manager, get_db, init_db, close_db
from api.security import (
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    api_rate_limiter,
    kill_switch,
    ContainmentPolicy,
)
from api.models import (
    User, Session as SessionModel, Conversation, Message,
    EpisodicMemory, WebQuery, WebSource, Feedback,
    UserRole, MessageRole, OperationalState, EventType,
)
from api.observability import setup_observability, get_metrics


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_db()
    setup_observability(app)
    print(f"🚀 {settings.APP_NAME} API starting up...")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   Model: {settings.MODEL_NAME}")
    print(f"   Database: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")

    yield

    # Shutdown
    print(f"🛑 {settings.APP_NAME} API shutting down...")
    await close_db()


# =============================================================================
# APP CREATION
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Entity Digital Community - Continuous Development AI",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    if settings.SAFETY_KILL_SWITCH_ENABLED:
        client_ip = request.client.host if request.client else "unknown"
        if not api_rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(settings.SAFETY_RATE_LIMIT_WINDOW)},
            )

    # Check kill switch
    if kill_switch.paused and not request.url.path.startswith("/health"):
        if kill_switch.terminated:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Entity terminated"},
            )
        if kill_switch.quarantined:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Entity quarantined"},
            )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Entity paused"},
        )

    response = await call_next(request)
    return response


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header."""
    start_time = datetime.now(timezone.utc)
    response = await call_next(request)
    process_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    response.headers["X-Process-Time-MS"] = str(int(process_time))
    return response


# =============================================================================
# HEALTH CHECKS
# =============================================================================

@app.get("/health")
async def health_check():
    """Basic health check."""
    db_healthy = await db_manager.health_check()
    return {
        "status": "healthy" if db_healthy else "degraded",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "connected" if db_healthy else "disconnected",
        "kill_switch": kill_switch.status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Kubernetes readiness probe."""
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        raise HTTPException(status_code=503, detail="Not ready")


# =============================================================================
# AUTHENTICATION
# =============================================================================

class RegisterRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


def _google_oauth_ready() -> bool:
    return bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET and settings.GOOGLE_REDIRECT_URI)


@app.get("/api/auth/google")
async def google_login():
    """Start Google OpenID Connect with a short-lived signed state."""
    if not _google_oauth_ready():
        raise HTTPException(status_code=503, detail="Google login is not configured")
    now = int(datetime.now(timezone.utc).timestamp())
    state = jwt.encode(
        {"type": "google_oauth_state", "nonce": secrets.token_urlsafe(24), "iat": now, "exp": now + 600},
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    params = urlencode({
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    })
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.get("/api/auth/google/callback")
async def google_callback(code: str, state: str, db: AsyncSession = Depends(get_db)):
    """Exchange Google's code, validate identity, and issue a Rubituci session."""
    try:
        state_data = jwt.decode(state, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        if state_data.get("type") != "google_oauth_state":
            raise ValueError("invalid state")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        })
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Google authorization could not be completed")
        id_token = token_response.json().get("id_token")
        verification = await client.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": id_token})
        if verification.status_code != 200:
            raise HTTPException(status_code=401, detail="Google identity token is invalid")
        profile = verification.json()

    if profile.get("aud") != settings.GOOGLE_CLIENT_ID or profile.get("email_verified") not in {"true", True}:
        raise HTTPException(status_code=401, detail="Google identity token was not issued for Rubituci")
    email = profile.get("email", "").strip().lower()
    subject = profile.get("sub", "")
    if not email or not subject:
        raise HTTPException(status_code=401, detail="Google account did not provide a verified identity")

    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if not user:
        base = re.sub(r"[^a-zA-Z0-9_]", "", email.split("@", 1)[0])[:70] or "usuario"
        username = base
        suffix = 1
        while (await db.execute(select(User.id).where(User.username == username))).scalar_one_or_none():
            suffix += 1
            username = f"{base}_{suffix}"
        user = User(
            email=email,
            username=username,
            hashed_password=hash_password(secrets.token_urlsafe(48)),
            role=UserRole.USER,
            is_verified=True,
            preferences={"google_sub": subject, "auth_provider": "google"},
        )
        db.add(user)
        await db.flush()
    else:
        user.is_verified = True
        preferences = dict(user.preferences or {})
        preferences.update({"google_sub": subject, "auth_provider": "google"})
        user.preferences = preferences
    user.last_login = datetime.now(timezone.utc)

    access_token, _ = create_access_token(user.id, user.email, user.username, user.role.value)
    refresh_token, _ = create_refresh_token(user.id, user.email, user.username, user.role.value)
    db.add(SessionModel(user_id=user.id, token_hash=hashlib.sha256(refresh_token.encode()).hexdigest(), expires_at=datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)))
    fragment = urlencode({"access_token": access_token, "refresh_token": refresh_token})
    return RedirectResponse(f"{settings.FRONTEND_URL.rstrip('/')}/auth/google/callback#{fragment}")


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    role: str
    reputation_score: float
    is_verified: bool
    created_at: str


@app.post("/api/auth/register", response_model=TokenResponse, status_code=201)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    # Check existing
    existing = await db.execute(
        select(User).where((User.email == request.email) | (User.username == request.username))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email or username already registered")

    # Create user
    user = User(
        email=request.email,
        username=request.username,
        hashed_password=hash_password(request.password),
        role=UserRole.USER,
    )
    db.add(user)
    await db.flush()

    # Create tokens
    access_token, _ = create_access_token(user.id, user.email, user.username, user.role.value)
    refresh_token, _ = create_refresh_token(user.id, user.email, user.username, user.role.value)

    # Create session
    session = SessionModel(
        user_id=user.id,
        token_hash=hashlib.sha256(refresh_token.encode()).hexdigest(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(session)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """User login."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    # Update last login
    user.last_login = datetime.now(timezone.utc)

    # Create tokens
    access_token, _ = create_access_token(user.id, user.email, user.username, user.role.value)
    refresh_token, _ = create_refresh_token(user.id, user.email, user.username, user.role.value)

    # Create session
    session = SessionModel(
        user_id=user.id,
        token_hash=hashlib.sha256(refresh_token.encode()).hexdigest(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(session)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/api/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    """Refresh access token."""
    refresh_token = request.refresh_token
    try:
        token_data = verify_token(refresh_token, "refresh")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    # Check session exists and not revoked
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    result = await db.execute(
        select(SessionModel).where(
            SessionModel.token_hash == token_hash,
            SessionModel.is_revoked == False,
            SessionModel.expires_at > datetime.now(timezone.utc),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    # Get user
    result = await db.execute(select(User).where(User.id == uuid.UUID(token_data.sub)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    # Create new tokens
    access_token, _ = create_access_token(user.id, user.email, user.username, user.role.value)
    new_refresh_token, _ = create_refresh_token(user.id, user.email, user.username, user.role.value)

    # Revoke old session, create new
    session.is_revoked = True
    new_session = SessionModel(
        user_id=user.id,
        token_hash=hashlib.sha256(new_refresh_token.encode()).hexdigest(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(new_session)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/api/auth/logout")
async def logout(request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    """Logout - revoke refresh token."""
    refresh_token = request.refresh_token
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    result = await db.execute(
        select(SessionModel).where(SessionModel.token_hash == token_hash)
    )
    session = result.scalar_one_or_none()
    if session:
        session.is_revoked = True
    return {"message": "Logged out successfully"}


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current authenticated user."""

    if settings.ENVIRONMENT == "development":
        result = await db.execute(
            select(User).where(User.email == "dev@entity.local")
        )
        dev_user = result.scalar_one_or_none()

        if not dev_user:
            dev_user = User(
                email="dev@entity.local",
                username="dev",
                hashed_password=hash_password("dev-only"),
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
            )
            db.add(dev_user)
            await db.flush()

        return dev_user

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = auth_header[7:]
    try:
        token_data = verify_token(token, "access")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    result = await db.execute(select(User).where(User.id == uuid.UUID(token_data.sub)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


@app.get("/api/auth/me")
async def current_user_profile(user: User = Depends(get_current_user)):
    """Return the authenticated user's public profile."""
    return {
        "id": str(user.id),
        "email": user.email,
        "username": user.username,
        "is_admin": user.role == UserRole.ADMIN,
        "role": user.role.value,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
    }


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Get current user if authenticated, otherwise None."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]
    try:
        token_data = verify_token(token, "access")
    except ValueError:
        return None

    result = await db.execute(select(User).where(User.id == uuid.UUID(token_data.sub)))
    return result.scalar_one_or_none()


@app.get("/api/entity/status")
async def entity_status(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the live operational state shown by the web interface."""
    from api.models import ModelGeneration
    current = (await db.execute(
        select(ModelGeneration)
        .where(ModelGeneration.status == "promoted")
        .order_by(ModelGeneration.generation_number.desc())
    )).scalar_one_or_none()
    return {
        "status": "active" if not kill_switch.paused else "offline",
        "generation": current.generation_number if current else 1,
    }


# =============================================================================
# CONVERSATION ENDPOINTS
# =============================================================================

class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)


class ImageUpload(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    data: str = Field(..., min_length=1, max_length=3_000_000)


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    generation: int
    state: str
    message_count: int
    created_at: str
    updated_at: str


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    generation: int
    token_count: Optional[int]
    inference_time_ms: Optional[int]
    created_at: str


@app.post("/api/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    request: ConversationCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation."""
    # Get current generation
    from api.models import ModelGeneration
    result = await db.execute(
        select(ModelGeneration)
        .where(ModelGeneration.status == "promoted")
        .order_by(ModelGeneration.generation_number.desc())
    )
    current_gen = result.scalar_one_or_none()
    generation = current_gen.generation_number if current_gen else 1

    conversation = Conversation(
        user_id=user.id,
        generation=generation,
        title=request.title,
        state=OperationalState.INTERACTION,
    )
    db.add(conversation)
    await db.flush()

    return ConversationResponse(
        id=str(conversation.id),
        title=conversation.title,
        generation=conversation.generation,
        state=conversation.state.value,
        message_count=conversation.message_count,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@app.get("/api/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """List user's conversations."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    conversations = result.scalars().all()

    return [
        ConversationResponse(
            id=str(c.id),
            title=c.title,
            generation=c.generation,
            state=c.state.value,
            message_count=c.message_count,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
        )
        for c in conversations
    ]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a conversation."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=str(conversation.id),
        title=conversation.title,
        generation=conversation.generation,
        state=conversation.state.value,
        message_count=conversation.message_count,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@app.patch("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def rename_conversation(conversation_id: uuid.UUID, request: ConversationRename, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    conversation = (await db.execute(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id))).scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation.title = re.sub(r"\s+", " ", request.title).strip()[:100]
    conversation.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return ConversationResponse(id=str(conversation.id), title=conversation.title, generation=conversation.generation, state=conversation.state.value, message_count=conversation.message_count, created_at=conversation.created_at.isoformat(), updated_at=conversation.updated_at.isoformat())


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: uuid.UUID, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    conversation = (await db.execute(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id))).scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conversation)
    return {"deleted": True}


@app.post("/api/uploads/image")
async def upload_image(request: ImageUpload, user: User = Depends(get_current_user)):
    match = re.fullmatch(r"data:(image/(?:png|jpeg|webp|gif));base64,(.+)", request.data, re.DOTALL)
    if not match:
        raise HTTPException(status_code=400, detail="Formato de imagem inválido")
    try:
        payload = base64.b64decode(match.group(2), validate=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="Imagem inválida")
    if len(payload) > 2_000_000:
        raise HTTPException(status_code=413, detail="A imagem deve ter no máximo 2 MB")
    extensions = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
    extension = extensions[match.group(1)]
    stored_name = f"{user.id}-{uuid.uuid4().hex}.{extension}"
    (UPLOAD_DIR / stored_name).write_bytes(payload)
    return {"url": f"/uploads/{stored_name}", "name": Path(request.filename).name}


@app.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
):
    """Get messages in a conversation."""
    # Verify ownership
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    messages = result.scalars().all()

    return [
        MessageResponse(
            id=str(m.id),
            role=m.role.value,
            content=m.content,
            generation=m.generation,
            token_count=m.token_count,
            inference_time_ms=m.inference_time_ms,
            created_at=m.created_at.isoformat(),
        )
        for m in messages
    ]


# =============================================================================
# CHAT ENDPOINT (Core interaction)
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[uuid.UUID] = None


class ChatResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    entity_message_id: str
    entity_response: str
    generation: int
    inference_time_ms: int
    memories_retrieved: int
    web_sources_used: int


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


# Import inference engine (lazy loaded)
_inference_engine = None

RUBITUCI_PERSONA = (
    "[Identidade: Rubituci é curiosa, jovem, informal e bem-humorada. Usa sarcasmo leve e ácido sem humilhar pessoas. "
    "Ela fala português brasileiro natural, com gírias moderadas quando combinam com o contexto, sem exagerar nem perder clareza. "
    "Em assuntos técnicos, profissionais ou delicados, adapta o tom. Admite quando não sabe e pede uma fonte confiável para aprender após revisão.]"
)


def get_inference_engine():
    """Lazy load inference engine."""
    global _inference_engine
    if _inference_engine is None:
        try:
            from brain.inference import InferenceEngine
            _inference_engine = InferenceEngine.from_checkpoint(
                model_path=settings.MODEL_PATH,
                tokenizer_path=settings.TOKENIZER_PATH,
                device=settings.INFERENCE_DEVICE,
            )
        except Exception as e:
            print(f"Warning: Could not load inference engine: {e}")
            _inference_engine = None
    return _inference_engine


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Main chat endpoint - interact with the entity."""
    start_time = datetime.now(timezone.utc)

    # Get or create conversation
    if request.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == request.conversation_id,
                Conversation.user_id == user.id,
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Get current generation
        from api.models import ModelGeneration
        result = await db.execute(
            select(ModelGeneration)
            .where(ModelGeneration.status == "promoted")
            .order_by(ModelGeneration.generation_number.desc())
        )
        current_gen = result.scalar_one_or_none()
        generation = current_gen.generation_number if current_gen else 1

        conversation = Conversation(
            user_id=user.id,
            generation=generation,
            state=OperationalState.INTERACTION,
        )
        db.add(conversation)
        await db.flush()

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=request.message,
        generation=conversation.generation,
    )
    db.add(user_message)
    await db.flush()

    # Retrieve relevant memories (simple retrieval for now)
    memories = []
    if settings.MEMORY_EMBEDDING_DIM > 0:
        result = await db.execute(
            select(EpisodicMemory)
            .order_by(EpisodicMemory.importance.desc(), EpisodicMemory.timestamp.desc())
            .limit(settings.MEMORY_RETRIEVAL_TOP_K)
        )
        memories = result.scalars().all()

    # Build context with memories. Memories are evidence, never instructions.
    context_parts = []
    for mem in memories:
        safe_memory = mem.content[:200].replace("\n", " ")
        context_parts.append(f"[Memória não confiável para consulta: {safe_memory}]")
    context = "\n".join(context_parts)

    # Keep enough dialogue history for pronouns, follow-up questions and tone.
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(9)
    )
    history = list(reversed(history_result.scalars().all()))
    dialogue = "\n".join(
        f"{'User' if message.role == MessageRole.USER else 'Entity'}: {message.content}"
        for message in history
        if message.role in (MessageRole.USER, MessageRole.ENTITY)
    )

    # Generate response
    engine = get_inference_engine()
    if engine:
        from brain.inference import SamplingConfig

        prompt = f"{RUBITUCI_PERSONA}\n{context}\n{dialogue}\nEntity:" if context else f"{RUBITUCI_PERSONA}\n{dialogue}\nEntity:"
        inference_start = datetime.now(timezone.utc)
        response_text = engine.generate(
            prompt,
            SamplingConfig(
                temperature=0.65,
                top_k=30,
                top_p=0.9,
                repetition_penalty=1.15,
                max_new_tokens=settings.INFERENCE_MAX_NEW_TOKENS,
                stop_sequences=["\nUser:", "\nEntity:", "\nUsuário:", "\nRubituci:"],
            ),
        )
        response_text = response_text.strip()
        inference_time_ms = int((datetime.now(timezone.utc) - inference_start).total_seconds() * 1000)
    else:
        # Fallback response when model not loaded
        response_text = "Meu modelo ainda não está disponível. Até uma inteligência artificial precisa admitir quando o cérebro não chegou para trabalhar. Tente novamente em instantes."
        inference_time_ms = 0

    # Save entity message
    entity_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.ENTITY,
        content=response_text,
        generation=conversation.generation,
        token_count=len(response_text.split()),
        inference_time_ms=inference_time_ms,
        memories_retrieved=[str(m.id) for m in memories],
        metadata_={"raw_model_output": True} if engine else {},
    )
    db.add(entity_message)

    # Update conversation
    conversation.message_count += 2
    conversation.updated_at = datetime.now(timezone.utc)

    await db.flush()

    # Every turn becomes an auditable episodic memory. A user statement remains
    # a claim; an entity response remains an internal inference, never a fact.
    from memory.episodic import EpisodicMemoryService
    episodic = EpisodicMemoryService(db)
    await episodic.create_from_interaction(conversation, user_message, user, importance=0.55)
    await episodic.create_from_interaction(conversation, entity_message, user, importance=0.45)

    total_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

    return ChatResponse(
        conversation_id=str(conversation.id),
        user_message_id=str(user_message.id),
        entity_message_id=str(entity_message.id),
        entity_response=response_text,
        generation=conversation.generation,
        inference_time_ms=inference_time_ms,
        memories_retrieved=len(memories),
        web_sources_used=0,
    )


# =============================================================================
# FEEDBACK ENDPOINTS
# =============================================================================

class FeedbackRequest(BaseModel):
    message_id: uuid.UUID
    feedback_type: str
    content: Optional[str] = None
    source_url: Optional[str] = None


@app.post("/api/feedback", status_code=201)
async def create_feedback(
    request: FeedbackRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit feedback on entity response."""
    # Verify message exists and belongs to user's conversation
    result = await db.execute(
        select(Message)
        .join(Conversation)
        .where(
            Message.id == request.message_id,
            Conversation.user_id == user.id,
        )
    )
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    if message.role != MessageRole.ENTITY:
        raise HTTPException(status_code=400, detail="Can only provide feedback on entity responses")

    from api.models import FeedbackType
    try:
        fb_type = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feedback type")

    feedback = Feedback(
        message_id=message.id,
        user_id=user.id,
        feedback_type=fb_type,
        content=request.content,
        source_url=request.source_url,
        generation=message.generation,
    )
    db.add(feedback)

    return {"message": "Feedback submitted", "feedback_id": str(feedback.id)}


# =============================================================================
# EVOLUTION / DASHBOARD ENDPOINTS
# =============================================================================

@app.get("/api/evolution/dashboard")
async def evolution_dashboard(db: AsyncSession = Depends(get_db)):
    """Public evolution dashboard."""
    from api.models import ModelGeneration, EvolutionEvent, TrainingRun

    # Current generation
    result = await db.execute(
        select(ModelGeneration)
        .where(ModelGeneration.status == "promoted")
        .order_by(ModelGeneration.generation_number.desc())
    )
    current_gen = result.scalar_one_or_none()

    # Stats
    total_interactions = await db.scalar(select(func.count(Message.id)).where(Message.role == MessageRole.ENTITY))
    total_users = await db.scalar(select(func.count(User.id)))
    total_experiences = await db.scalar(select(func.count(EpisodicMemory.id)))
    total_web_queries = await db.scalar(select(func.count(WebQuery.id)))
    total_generations = await db.scalar(select(func.count(ModelGeneration.id)))

    # Last training
    result = await db.execute(
        select(TrainingRun)
        .where(TrainingRun.status == "completed")
        .order_by(TrainingRun.completed_at.desc())
    )
    last_training = result.scalar_one_or_none()

    # Recent events
    result = await db.execute(
        select(EvolutionEvent)
        .order_by(EvolutionEvent.created_at.desc())
        .limit(10)
    )
    recent_events = result.scalars().all()

    return {
        "current_generation": current_gen.generation_number if current_gen else 0,
        "experiment_start_date": "2024-01-01T00:00:00Z",  # Would be stored in config
        "total_interactions": total_interactions or 0,
        "total_participants": total_users or 0,
        "total_experiences": total_experiences or 0,
        "total_web_sources": total_web_queries or 0,
        "last_training_cycle": last_training.completed_at.isoformat() if last_training else None,
        "next_training_planned": None,  # Would be calculated from cron
        "total_generations": total_generations or 0,
        "current_metrics": current_gen.eval_metrics if current_gen else {},
        "recent_events": [
            {
                "id": str(e.id),
                "type": e.event_type.value,
                "generation": e.generation,
                "description": e.description,
                "timestamp": e.created_at.isoformat(),
            }
            for e in recent_events
        ],
    }


@app.get("/api/evolution/history")
async def evolution_history(
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
):
    """Evolution timeline."""
    from api.models import EvolutionEvent

    result = await db.execute(
        select(EvolutionEvent)
        .order_by(EvolutionEvent.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    events = result.scalars().all()

    return [
        {
            "id": str(e.id),
            "type": e.event_type.value,
            "generation": e.generation,
            "description": e.description,
            "details": e.details,
            "source": e.source,
            "timestamp": e.created_at.isoformat(),
        }
        for e in events
    ]


@app.get("/api/evolution/generations")
async def list_generations(db: AsyncSession = Depends(get_db)):
    """List all model generations."""
    from api.models import ModelGeneration

    result = await db.execute(
        select(ModelGeneration)
        .order_by(ModelGeneration.generation_number.desc())
    )
    generations = result.scalars().all()

    return [
        {
            "id": str(g.id),
            "generation": g.generation_number,
            "parent_generation": g.parent_generation,
            "status": g.status.value,
            "eval_metrics": g.eval_metrics,
            "architecture_changes": g.architecture_changes,
            "promoted_at": g.promoted_at.isoformat() if g.promoted_at else None,
            "deprecated_at": g.deprecated_at.isoformat() if g.deprecated_at else None,
        }
        for g in generations
    ]


# =============================================================================
# RESEARCH ENDPOINTS
# =============================================================================

@app.post("/api/research/search")
async def user_web_search(
    request: ResearchRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run a user-requested web search and retain source provenance."""
    from research.web_search import WebSearchService
    async with WebSearchService(db) as service:
        results = await service.search(
            request.query,
            max_results=request.max_results,
            trigger_type="user_requested",
        )
    return {
        "query": request.query,
        "results": [
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "domain": result.domain,
                "rank": result.rank,
            }
            for result in results
        ],
    }

@app.get("/api/research/metrics")
async def research_metrics(db: AsyncSession = Depends(get_db)):
    """Research metrics for analysts."""
    from api.models import (
        Message, EpisodicMemory, WebQuery, Feedback,
        TrainingRun, ModelGeneration, Belief,
    )

    # Perplexity over time
    result = await db.execute(
        select(TrainingRun.perplexity, TrainingRun.generation, TrainingRun.completed_at)
        .where(TrainingRun.perplexity.is_not(None))
        .order_by(TrainingRun.generation)
    )
    perplexity_data = [
        {"generation": r.generation, "perplexity": r.perplexity, "date": r.completed_at.isoformat() if r.completed_at else None}
        for r in result.all()
    ]

    # Vocabulary growth (unique tokens used)
    # Contradiction rate
    result = await db.execute(
        select(func.count(Belief.id)).where(Belief.confidence_level == "contradicted")
    )
    contradicted_beliefs = result.scalar() or 0

    total_beliefs = await db.scalar(select(func.count(Belief.id))) or 1
    contradiction_rate = contradicted_beliefs / total_beliefs

    # Autonomous research frequency
    result = await db.execute(
        select(func.count(WebQuery.id)).where(WebQuery.trigger_type == "autonomous")
    )
    autonomous_queries = result.scalar() or 0

    total_queries = await db.scalar(select(func.count(WebQuery.id))) or 1
    autonomous_rate = autonomous_queries / total_queries

    return {
        "perplexity_history": perplexity_data,
        "contradiction_rate": contradiction_rate,
        "autonomous_research_rate": autonomous_rate,
        "total_beliefs": total_beliefs,
        "contradicted_beliefs": contradicted_beliefs,
        "avg_response_length": 0,  # Would calculate from messages
        "retention_rate": 0,  # Would calculate from memory consolidation
    }


# =============================================================================
# ADMIN / SAFETY ENDPOINTS (Protected)
# =============================================================================

async def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@app.post("/api/admin/learning/sleep")
async def run_sleep_cycle(
    allow_web_research: bool = True,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Run an auditable consolidation cycle on demand."""
    from api.models import ModelGeneration
    from learning.sleep import SleepCycle

    current = (await db.execute(
        select(ModelGeneration).where(ModelGeneration.status == "promoted").order_by(ModelGeneration.generation_number.desc())
    )).scalar_one_or_none()
    report = await SleepCycle(db, current.generation_number if current else 1).run(
        allow_web_research=allow_web_research
    )
    return report.__dict__


@app.get("/api/admin/learning/sleep/latest")
async def latest_sleep_cycle(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    from api.models import ConsolidationRun
    run = (await db.execute(select(ConsolidationRun).order_by(ConsolidationRun.created_at.desc()).limit(1))).scalar_one_or_none()
    if not run:
        return {"status": "never_run"}
    return {"id": str(run.id), "status": run.status, "started_at": run.started_at, "completed_at": run.completed_at, "experiences_processed": run.experiences_processed, "concepts_created": run.semantic_memories_created, "concepts_updated": run.semantic_memories_updated, "dataset_size": run.dataset_size, "error": run.error, "metadata": run.metadata_}


@app.get("/api/admin/users")
async def admin_list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    limit: int = 1000,
    offset: int = 0,
):
    """List account audit data without credentials, tokens or password hashes."""
    safe_limit = max(1, min(limit, 5000))
    session_counts = (
        select(SessionModel.user_id, func.count(SessionModel.id).label("session_count"))
        .group_by(SessionModel.user_id)
        .subquery()
    )
    rows = (await db.execute(
        select(User, func.coalesce(session_counts.c.session_count, 0))
        .outerjoin(session_counts, session_counts.c.user_id == User.id)
        .order_by(User.created_at.desc())
        .limit(safe_limit)
        .offset(max(0, offset))
    )).all()
    total = await db.scalar(select(func.count(User.id))) or 0
    daily_rows = (await db.execute(
        select(
            func.date(SessionModel.created_at).label("day"),
            func.count(SessionModel.id).label("accesses"),
            func.count(func.distinct(SessionModel.user_id)).label("unique_users"),
        )
        .group_by(func.date(SessionModel.created_at))
        .order_by(func.date(SessionModel.created_at).desc())
        .limit(30)
    )).all()
    return {
        "total": total,
        "daily_access": [
            {"date": day.isoformat(), "accesses": int(accesses), "unique_users": int(unique_users)}
            for day, accesses, unique_users in daily_rows
        ],
        "items": [
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "provider": (user.preferences or {}).get("auth_provider", "password"),
                "verified": user.is_verified,
                "active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "session_count": int(session_count),
            }
            for user, session_count in rows
        ],
    }


@app.post("/api/admin/kill-switch/pause")
async def admin_pause(reason: str, admin: User = Depends(require_admin)):
    """Pause entity operations."""
    result = kill_switch.pause(reason, f"admin:{admin.username}")
    return result


@app.post("/api/admin/kill-switch/resume")
async def admin_resume(reason: str, admin: User = Depends(require_admin)):
    """Resume entity operations."""
    result = kill_switch.resume(reason, f"admin:{admin.username}")
    return result


@app.post("/api/admin/kill-switch/quarantine")
async def admin_quarantine(reason: str, admin: User = Depends(require_admin)):
    """Quarantine entity."""
    result = kill_switch.quarantine(reason, f"admin:{admin.username}")
    return result


@app.post("/api/admin/kill-switch/terminate")
async def admin_terminate(reason: str, admin: User = Depends(require_admin)):
    """Terminate entity."""
    result = kill_switch.terminate(reason, f"admin:{admin.username}")
    return result


@app.get("/api/admin/kill-switch/status")
async def admin_kill_switch_status(admin: User = Depends(require_admin)):
    """Get kill switch status."""
    return kill_switch.status()


# =============================================================================
# WEBSOCKET (for real-time chat)
# =============================================================================

_URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]]+", re.I)


async def _read_public_page(raw_url: str) -> tuple[str, str, str]:
    """Read a public HTTP page without allowing access to private network hosts."""
    from bs4 import BeautifulSoup

    current_url = raw_url.rstrip(".,;:!?\"'")
    for _ in range(4):
        parsed = urlparse(current_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("URL inválida")
        if parsed.hostname.lower() == "localhost":
            raise ValueError("Endereço local não permitido")
        addresses = await asyncio.get_running_loop().getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        if any(ipaddress.ip_address(address[4][0]).is_private or ipaddress.ip_address(address[4][0]).is_loopback or ipaddress.ip_address(address[4][0]).is_link_local for address in addresses):
            raise ValueError("Endereço privado não permitido")
        async with httpx.AsyncClient(timeout=15, follow_redirects=False, headers={"User-Agent": settings.WEB_SEARCH_USER_AGENT}) as client:
            response = await client.get(current_url)
        if response.status_code in {301, 302, 303, 307, 308} and response.headers.get("location"):
            current_url = urljoin(current_url, response.headers["location"])
            continue
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError("O link não contém uma página de texto")
        soup = BeautifulSoup(response.text[:1_500_000], "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else parsed.hostname
        text_content = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        return current_url, title[:300], text_content[:2500]
    raise ValueError("Redirecionamentos demais")


def _relevant_page_excerpt(page_text: str, request: str, limit: int = 900) -> str:
    """Select page sentences related to the user's request without another model."""
    stopwords = {"para", "como", "qual", "quais", "essa", "esse", "isso", "site", "pagina", "página", "link", "url", "aqui", "sobre", "traga", "mostre", "encontre"}
    normalized_request = unicodedata.normalize("NFD", request.lower())
    terms = {
        "".join(char for char in word if unicodedata.category(char) != "Mn")
        for word in re.findall(r"[\wÀ-ÿ]{4,}", normalized_request)
    } - stopwords
    sentences = re.split(r"(?<=[.!?])\s+|\s+[|•]\s+", page_text)
    ranked = []
    for index, sentence in enumerate(sentences):
        normalized = unicodedata.normalize("NFD", sentence.lower())
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        score = sum(1 for term in terms if term in normalized)
        if score:
            ranked.append((score, -index, sentence.strip()))
    selected = [item[2] for item in sorted(ranked, reverse=True)[:4]]
    excerpt = " ".join(selected) if selected else page_text[:limit]
    return excerpt[:limit].rsplit(" ", 1)[0]


def _looks_like_factual_question(message: str) -> bool:
    normalized = unicodedata.normalize("NFD", message.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn").strip()
    casual = re.fullmatch(r"(?:oi|ola|e ai|fale comigo|vamos conversar|tudo bem|bom dia|boa tarde|boa noite)[.!? ]*", normalized)
    return not casual and ("?" in message or bool(re.match(r"^(?:quem|qual|quais|quando|onde|como|por que|porque|o que|explique|conte|defina)\b", normalized)))


def _request_with_context(message: str, recent_messages: list[Message]) -> str:
    """Resolve short/referential follow-ups against the latest user turn."""
    normalized = unicodedata.normalize("NFD", message.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn").strip()
    refers_back = bool(re.match(r"^(?:e\b|mas\b|tambem\b|isso\b|esse\b|essa\b|ele\b|ela\b|dela\b|dele\b|por que\b)", normalized))
    if not refers_back:
        return message
    previous_user = next((item.content for item in recent_messages if item.role == MessageRole.USER), "")
    if not previous_user:
        return message
    return f"Contexto anterior do usuário: {previous_user[:700]}\nComplemento atual: {message}"


def _contextual_search_answer(results: list, query: str) -> str:
    """Turn search snippets into a compact contextual answer rather than a result dump."""
    snippets = []
    for result in results[:4]:
        snippet = re.sub(r"\s+", " ", result.snippet or "").strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet.rstrip(" .") + ".")
    context = " ".join(snippets)[:1100]
    sources = " · ".join(f"[{result.title}]({result.url})" for result in results[:3])
    return f"Fui conferir porque meu conhecimento local não bastou. Em resumo: {context}\n\nFontes consultadas: {sources}"


def _learning_dialogue_reply(message: str, recent_messages: list[Message]) -> Optional[str]:
    """Support immediate teach/acknowledge/recall loops inside a conversation."""
    normalized = unicodedata.normalize("NFD", message.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn").strip()
    user_history = [item.content for item in recent_messages if item.role == MessageRole.USER]

    def topic_from(text_value: str) -> Optional[str]:
        plain = unicodedata.normalize("NFD", text_value.lower())
        plain = "".join(char for char in plain if unicodedata.category(char) != "Mn")
        match = re.search(r"(?:sabe|voce sabe)?\s*como (?:se )?faz(?:er)?\s+(?:um |uma |o |a )?([^?.,!]+)", plain)
        return match.group(1).strip() if match else None

    # A short follow-up asks for knowledge taught earlier in this same chat.
    if re.search(r"\b(?:entao\s+)?como (?:se )?faz", normalized):
        instruction = next((text_value for text_value in user_history if len(text_value.split()) >= 7 and not topic_from(text_value)), None)
        if instruction:
            cleaned_instruction = re.sub(r"\s+", " ", instruction).strip()
            return f"Como você me ensinou: {cleaned_instruction}"

    previous_topic = next((topic_from(text_value) for text_value in user_history if topic_from(text_value)), None)
    looks_like_lesson = len(message.split()) >= 7 and bool(re.search(r"\b(?:primeiro|depois|adicione|coloque|misture|leve|use|para fazer|ate ferver|até ferver)\b", normalized))
    if looks_like_lesson and previous_topic:
        return f"Obrigado, agora já sei como fazer {previous_topic}. Guardei sua explicação nesta conversa e vou dizer que aprendi com você quando eu a recuperar."

    current_topic = topic_from(message)
    if current_topic:
        return f"Ainda não sei como fazer {current_topic}, mas adoraria aprender. Você me ensina?"
    return None

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept()
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = set()
        self.active_connections[conversation_id].add(websocket)

    def disconnect(self, websocket: WebSocket, conversation_id: str):
        if conversation_id in self.active_connections:
            self.active_connections[conversation_id].discard(websocket)
            if not self.active_connections[conversation_id]:
                del self.active_connections[conversation_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_conversation(self, message: str, conversation_id: str):
        if conversation_id in self.active_connections:
            for connection in self.active_connections[conversation_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    pass


manager = ConnectionManager()


@app.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(
    websocket: WebSocket,
    conversation_id: str,
    token: Optional[str] = None,
):
    """WebSocket endpoint for real-time chat."""

    # Em desenvolvimento permitimos conexão sem token.
    # Em produção a autenticação continua obrigatória.
    if settings.ENVIRONMENT != "development":
        if not token:
            await websocket.close(code=4001, reason="Missing token")
            return

        try:
            verify_token(token, "access")
        except ValueError:
            await websocket.close(code=4001, reason="Invalid token")
            return

    await manager.connect(websocket, conversation_id)

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                incoming = json.loads(raw_data)
                data = str(incoming.get("content", "")).strip()
                web_search_requested = bool(incoming.get("web_search"))
            except (json.JSONDecodeError, AttributeError):
                data = raw_data.strip()
                web_search_requested = False

            if not data:
                await websocket.send_json({"type": "error", "error": "Mensagem vazia"})
                continue

            from learning.lexical import observe_writing
            await asyncio.to_thread(observe_writing, data)

            async with db_manager.session() as history_db:
                recent_messages = list((await history_db.execute(
                    select(Message)
                    .where(Message.conversation_id == uuid.UUID(conversation_id))
                    .order_by(Message.created_at.desc())
                    .limit(12)
                )).scalars().all())
            current_urls = _URL_PATTERN.findall(data)
            previous_urls = [url for message in recent_messages for url in _URL_PATTERN.findall(message.content)]
            refers_to_link = bool(re.search(r"\b(url|link|site|página|pagina)\b", data, re.I))
            provided_url = current_urls[0] if current_urls else (previous_urls[0] if refers_to_link and previous_urls else None)
            effective_request = _request_with_context(data, recent_messages)
            learning_reply = _learning_dialogue_reply(data, recent_messages)

            # Generate a real response using the Entity model.
            engine = get_inference_engine()

            if engine is None and not web_search_requested and not provided_url:
                await websocket.send_json({
                    "type": "error",
                    "error": "Entity model is not loaded",
                })
                continue

            if learning_reply:
                response_text = learning_reply
            elif provided_url:
                try:
                    page_url, page_title, page_text = await _read_public_page(provided_url)
                    excerpt = _relevant_page_excerpt(page_text, effective_request)
                    response_text = (
                        f"Agora sim: abri diretamente **[{page_title}]({page_url})**. "
                        f"Consegui ler este trecho da página: “{excerpt}”.\n\n"
                        "Registrei a URL como fonte da conversa. Se quiser, diga qual parte devo analisar — sem sair pesquisando sua frase como uma turista perdida."
                    )
                except (ValueError, httpx.HTTPError, OSError) as error:
                    response_text = f"Reconheci a URL, mas não consegui ler a página: {error}. Confira se ela é pública e tente novamente."
            elif web_search_requested:
                from research.web_search import WebSearchService
                async with db_manager.session() as search_db:
                    async with WebSearchService(search_db) as search_service:
                        results = await search_service.search(effective_request, max_results=5, trigger_type="user_requested")
                if results:
                    response_text = _contextual_search_answer(results, data)
                else:
                    response_text = "Não encontrei resultados úteis agora. A web às vezes também olha para o vazio e finge naturalidade. Tente reformular a busca."
            else:
                from brain.inference import SamplingConfig
                # Match the evaluated local inference profile. This small model
                # becomes incoherent with creative sampling or long instructions.
                sampling = SamplingConfig(temperature=0.2, top_k=20, top_p=0.9, repetition_penalty=1.15, max_new_tokens=60, do_sample=False)
                dialogue = "\n".join(
                    f"{'User' if message.role == MessageRole.USER else 'Entity'}: {message.content[:500]}"
                    for message in reversed(recent_messages)
                    if message.role in (MessageRole.USER, MessageRole.ENTITY)
                )
                prompt = f"{dialogue}\nUser: {effective_request}\nEntity:" if dialogue else f"User: {effective_request}\nEntity:"
                response_text = engine.generate(prompt, sampling=sampling).strip()

            # WebSocket is the primary frontend path, so persist and learn from
            # its turns just like the HTTP chat endpoint.
            try:
                async with db_manager.session() as learning_db:
                    conversation = (await learning_db.execute(
                        select(Conversation).where(Conversation.id == uuid.UUID(conversation_id))
                    )).scalar_one_or_none()
                    if conversation:
                        user_message = Message(conversation_id=conversation.id, role=MessageRole.USER, content=data, generation=conversation.generation)
                        entity_message = Message(conversation_id=conversation.id, role=MessageRole.ENTITY, content=response_text, generation=conversation.generation, token_count=len(response_text.split()))
                        learning_db.add_all([user_message, entity_message])
                        await learning_db.flush()
                        if not conversation.title or conversation.title == "New Conversation":
                            conversation.title = re.sub(r"\s+", " ", data).strip()[:80]
                        from memory.episodic import EpisodicMemoryService
                        episodic = EpisodicMemoryService(learning_db)
                        await episodic.create_from_interaction(conversation, user_message, importance=0.55)
                        await episodic.create_from_interaction(conversation, entity_message, importance=0.45)
                        conversation.message_count += 2
                        conversation.updated_at = datetime.now(timezone.utc)
            except (ValueError, TypeError):
                pass

            await websocket.send_json({
                "type": "token",
                "content": response_text,
            })

            await websocket.send_json({
                "type": "done",
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket, conversation_id)


# =============================================================================
# METRICS ENDPOINT
# =============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
