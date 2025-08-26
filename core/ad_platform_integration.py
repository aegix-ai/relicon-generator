"""
Ad Platform API Integration Framework for Phase 2
Provides standardized integration with Meta, TikTok, and Google Ads platforms.
"""

import asyncio
import aiohttp
import time
import hashlib
import secrets
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from urllib.parse import urlencode, parse_qs, urlparse
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# SQLAlchemy imports for OAuth token storage
try:
    from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON, Index, UniqueConstraint
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.sql import func
    from sqlalchemy.orm import Session
    import uuid
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from core.config import config
from core.logger import get_logger
from core.monitoring import AdPerformanceMetrics, record_ad_performance_metrics
from core.exceptions import ConfigurationError

# Import database setup from brand_intelligence if available
try:
    from core.brand_intelligence import Base, get_db_session
    DATABASE_AVAILABLE = True
except ImportError:
    Base = None
    DATABASE_AVAILABLE = False
    def get_db_session():
        return None

logger = get_logger(__name__)

# OAuth2 Security Configuration
class OAuth2Security:
    """Security utilities for OAuth2 token management."""
    
    def __init__(self):
        # Generate or use existing encryption key
        self._encryption_key = self._get_encryption_key()
        self._cipher_suite = Fernet(self._encryption_key)
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for token storage."""
        # In production, this should come from secure key management
        password = config.database.password.encode() if config.database.password else b"relicon-oauth-key"
        salt = b"relicon-salt-2024"  # In production, use random salt stored securely
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt token for secure storage."""
        try:
            return self._cipher_suite.encrypt(token.encode()).decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt token for use."""
        try:
            return self._cipher_suite.decrypt(encrypted_token.encode()).decode()
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise
    
    def generate_state_token(self) -> str:
        """Generate secure state token for OAuth2 flow."""
        return secrets.token_urlsafe(32)

# Global security instance
oauth_security = OAuth2Security()

# Database Models for OAuth Token Storage
if SQLALCHEMY_AVAILABLE and Base is not None:
    class OAuthToken(Base):
        """OAuth tokens table for secure credential storage."""
        __tablename__ = 'oauth_tokens'
        
        # Primary key and platform info
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid())
        platform = Column(String(50), nullable=False, index=True)  # meta, tiktok, google_ads
        account_id = Column(String(255), nullable=False)  # Platform-specific account identifier
        
        # Encrypted token data
        access_token_encrypted = Column(Text, nullable=False)
        refresh_token_encrypted = Column(Text, nullable=True)
        token_type = Column(String(50), default='Bearer')
        
        # Token metadata
        expires_at = Column(DateTime, nullable=True)
        scope = Column(String(500), nullable=True)
        token_issued_at = Column(DateTime, default=func.current_timestamp())
        
        # Security and audit fields
        is_active = Column(Boolean, default=True)
        last_used = Column(DateTime, nullable=True)
        usage_count = Column(Integer, default=0)
        last_refresh = Column(DateTime, nullable=True)
        
        # OAuth flow metadata
        authorization_code = Column(String(500), nullable=True)
        state_token = Column(String(100), nullable=True)
        redirect_uri = Column(String(500), nullable=True)
        
        # Timestamps
        created_at = Column(DateTime, default=func.current_timestamp(), server_default=func.current_timestamp())
        updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
        
        # Indexes for performance and security
        __table_args__ = (
            Index('idx_oauth_tokens_platform_account', 'platform', 'account_id'),
            Index('idx_oauth_tokens_expires_at', 'expires_at'),
            Index('idx_oauth_tokens_active', 'is_active'),
            Index('idx_oauth_tokens_state_token', 'state_token'),
            UniqueConstraint('platform', 'account_id', name='uq_oauth_platform_account'),
        )
        
        def set_access_token(self, token: str):
            """Set encrypted access token."""
            self.access_token_encrypted = oauth_security.encrypt_token(token)
            self.last_refresh = func.current_timestamp()
        
        def get_access_token(self) -> str:
            """Get decrypted access token."""
            if not self.access_token_encrypted:
                return None
            return oauth_security.decrypt_token(self.access_token_encrypted)
        
        def set_refresh_token(self, token: str):
            """Set encrypted refresh token."""
            if token:
                self.refresh_token_encrypted = oauth_security.encrypt_token(token)
        
        def get_refresh_token(self) -> Optional[str]:
            """Get decrypted refresh token."""
            if not self.refresh_token_encrypted:
                return None
            return oauth_security.decrypt_token(self.refresh_token_encrypted)
        
        def is_expired(self) -> bool:
            """Check if token is expired."""
            if not self.expires_at:
                return False
            return datetime.utcnow() > self.expires_at
        
        def expires_soon(self, minutes: int = 30) -> bool:
            """Check if token expires within specified minutes."""
            if not self.expires_at:
                return False
            return datetime.utcnow() + timedelta(minutes=minutes) > self.expires_at
    
    class OAuthAuditLog(Base):
        """OAuth audit log for security monitoring."""
        __tablename__ = 'oauth_audit_logs'
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        platform = Column(String(50), nullable=False, index=True)
        account_id = Column(String(255), nullable=True)
        event_type = Column(String(100), nullable=False)  # auth_success, auth_failed, token_refresh, etc.
        event_details = Column(JSON, nullable=True)
        ip_address = Column(String(45), nullable=True)
        user_agent = Column(String(500), nullable=True)
        timestamp = Column(DateTime, default=func.current_timestamp(), server_default=func.current_timestamp())
        
        __table_args__ = (
            Index('idx_oauth_audit_platform_timestamp', 'platform', 'timestamp'),
            Index('idx_oauth_audit_event_type', 'event_type'),
        )
else:
    OAuthToken = None
    OAuthAuditLog = None

class AdPlatform(Enum):
    """Supported ad platforms."""
    META = "meta"
    TIKTOK = "tiktok"
    GOOGLE_ADS = "google_ads"

class CampaignStatus(Enum):
    """Campaign status across platforms."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    DRAFT = "draft"

@dataclass
class CreativeAsset:
    """Creative asset for ad campaigns."""
    asset_id: str
    asset_type: str  # video, image, text
    asset_url: Optional[str] = None
    asset_data: Optional[bytes] = None
    duration_seconds: Optional[int] = None
    dimensions: Optional[Tuple[int, int]] = None
    file_size_bytes: Optional[int] = None
    template_id: Optional[str] = None
    quality_score: Optional[float] = None

@dataclass
class CampaignConfig:
    """Campaign configuration for ad platforms."""
    campaign_name: str
    objective: str
    budget_daily: float
    target_audience: Dict[str, Any]
    creative_assets: List[CreativeAsset]
    bid_strategy: Optional[str] = None
    schedule: Optional[Dict[str, Any]] = None
    placements: Optional[List[str]] = None

@dataclass
class AdPerformanceData:
    """Standardized ad performance data across platforms."""
    ad_id: str
    campaign_id: str
    platform: str
    timestamp: datetime
    
    # Core metrics
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    
    # Calculated metrics
    ctr: float = 0.0
    cpc: float = 0.0
    cpm: float = 0.0
    conversion_rate: float = 0.0
    roas: float = 0.0
    
    # Video-specific metrics
    video_views: int = 0
    video_completion_rate: float = 0.0
    engagement_rate: float = 0.0
    
    # Creative metadata
    creative_id: Optional[str] = None
    template_id: Optional[str] = None
    niche_type: Optional[str] = None

@dataclass
class OAuth2Config:
    "OAuth2 configuration for ad platforms."
    client_id: str
    client_secret: str
    redirect_uri: str
    authorization_url: str
    token_url: str
    scope: List[str]
    extra_params: Dict[str, Any] = field(default_factory=dict)

class OAuth2TokenManager:
    "Manages OAuth2 tokens with automatic refresh and secure storage."
    
    def __init__(self, platform: str):
        self.platform = platform
        self._rate_limiter = {}  # Simple in-memory rate limiter
    
    async def get_valid_token(self, account_id: str) -> Optional[str]:
        "Get valid access token, refreshing if necessary"
        if not DATABASE_AVAILABLE:
            logger.warning(f"Database not available for token storage: {self.platform}")
            return None
        
        session = get_db_session()
        if not session:
            return None
        
        try:
            # Get token from database
            oauth_token = session.query(OAuthToken).filter(
                OAuthToken.platform == self.platform,
                OAuthToken.account_id == account_id,
                OAuthToken.is_active == True
            ).first()
            
            if not oauth_token:
                logger.warning(f\"No OAuth token found for {self.platform} account {account_id}\")
                return None
            
            # Check if token is expired or expires soon
            if oauth_token.is_expired():
                logger.info(f\"Token expired for {self.platform} account {account_id}, attempting refresh\")
                success = await self._refresh_token(oauth_token, session)
                if not success:
                    return None
            elif oauth_token.expires_soon():
                logger.info(f\"Token expires soon for {self.platform} account {account_id}, refreshing proactively\")
                await self._refresh_token(oauth_token, session)
            
            # Update usage tracking
            oauth_token.last_used = func.current_timestamp()
            oauth_token.usage_count = oauth_token.usage_count + 1
            session.commit()
            
            return oauth_token.get_access_token()
            
        except Exception as e:
            logger.error(f\"Error getting valid token for {self.platform}: {e}\")
            session.rollback()
            return None
        finally:
            session.close()
    
    async def store_token(self, account_id: str, access_token: str, refresh_token: str = None, 
                         expires_in: int = None, scope: str = None) -> bool:
        \"\"\"Store OAuth token securely in database.\"\"\"                         
        if not DATABASE_AVAILABLE:
            logger.warning(f\"Database not available for token storage: {self.platform}\")
            return False
        
        session = get_db_session()
        if not session:
            return False
        
        try:
            # Check for existing token
            existing_token = session.query(OAuthToken).filter(
                OAuthToken.platform == self.platform,
                OAuthToken.account_id == account_id
            ).first()
            
            expires_at = None
            if expires_in:
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            
            if existing_token:
                # Update existing token
                existing_token.set_access_token(access_token)
                if refresh_token:
                    existing_token.set_refresh_token(refresh_token)
                existing_token.expires_at = expires_at
                existing_token.scope = scope
                existing_token.is_active = True
                existing_token.updated_at = func.current_timestamp()
                oauth_token = existing_token
            else:
                # Create new token
                oauth_token = OAuthToken(
                    platform=self.platform,
                    account_id=account_id,
                    expires_at=expires_at,
                    scope=scope
                )
                oauth_token.set_access_token(access_token)
                if refresh_token:
                    oauth_token.set_refresh_token(refresh_token)
                session.add(oauth_token)
            
            session.commit()
            
            # Log successful token storage
            await self._log_auth_event(\"token_stored\", account_id, 
                                     {\"expires_at\": expires_at.isoformat() if expires_at else None,
                                      \"has_refresh_token\": bool(refresh_token)})
            
            logger.info(f\"OAuth token stored successfully for {self.platform} account {account_id}\")
            return True
            
        except Exception as e:
            logger.error(f\"Error storing token for {self.platform}: {e}\")
            session.rollback()
            return False
        finally:
            session.close()
    
    async def _refresh_token(self, oauth_token: 'OAuthToken', session: Session) -> bool:
        \"\"\"Refresh OAuth token using refresh token.\"\"\"        
        refresh_token = oauth_token.get_refresh_token()
        if not refresh_token:
            logger.warning(f\"No refresh token available for {self.platform} account {oauth_token.account_id}\")
            return False
        
        # Check rate limiting
        if not self._check_rate_limit(oauth_token.account_id):
            logger.warning(f\"Rate limit exceeded for token refresh: {self.platform} {oauth_token.account_id}\")
            return False
        
        try:
            # Platform-specific token refresh
            token_data = await self._platform_refresh_token(oauth_token)
            if not token_data:
                return False
            
            # Update token in database
            oauth_token.set_access_token(token_data['access_token'])
            if 'refresh_token' in token_data:
                oauth_token.set_refresh_token(token_data['refresh_token'])
            
            if 'expires_in' in token_data:
                oauth_token.expires_at = datetime.utcnow() + timedelta(seconds=token_data['expires_in'])
            
            oauth_token.last_refresh = func.current_timestamp()
            session.commit()
            
            # Log successful refresh
            await self._log_auth_event(\"token_refreshed\", oauth_token.account_id, 
                                     {\"expires_at\": oauth_token.expires_at.isoformat() if oauth_token.expires_at else None})
            
            logger.info(f\"Token refreshed successfully for {self.platform} account {oauth_token.account_id}\")
            return True
            
        except Exception as e:
            logger.error(f\"Token refresh failed for {self.platform}: {e}\")
            await self._log_auth_event(\"token_refresh_failed\", oauth_token.account_id, {\"error\": str(e)})
            return False
    
    async def _platform_refresh_token(self, oauth_token: 'OAuthToken') -> Optional[Dict[str, Any]]:
        \"\"\"Platform-specific token refresh implementation.\"\"\"        
        # This will be implemented by platform-specific subclasses
        raise NotImplementedError(\"Platform-specific token refresh must be implemented\")
    
    def _check_rate_limit(self, account_id: str) -> bool:
        \"\"\"Simple rate limiting for token refresh requests.\"\"\"        
        now = time.time()
        key = f\"{self.platform}:{account_id}\"
        
        if key not in self._rate_limiter:
            self._rate_limiter[key] = {'count': 0, 'reset_time': now + 3600}
        
        rate_info = self._rate_limiter[key]
        
        # Reset counter if time window expired
        if now > rate_info['reset_time']:
            rate_info['count'] = 0
            rate_info['reset_time'] = now + 3600
        
        # Check rate limit (max 10 refreshes per hour per account)
        if rate_info['count'] >= 10:
            return False
        
        rate_info['count'] += 1
        return True
    
    async def _log_auth_event(self, event_type: str, account_id: str, details: Dict[str, Any] = None):
        \"\"\"Log authentication events for audit trail.\"\"\"        
        if not DATABASE_AVAILABLE:
            return
        
        session = get_db_session()
        if not session:
            return
        
        try:
            audit_log = OAuthAuditLog(
                platform=self.platform,
                account_id=account_id,
                event_type=event_type,
                event_details=details or {}
            )
            session.add(audit_log)
            session.commit()
            
        except Exception as e:
            logger.error(f\"Failed to log auth event: {e}\")
            session.rollback()
        finally:
            session.close()
    
    async def revoke_token(self, account_id: str) -> bool:
        \"\"\"Revoke and deactivate OAuth token.\"\"\"        
        if not DATABASE_AVAILABLE:
            return False
        
        session = get_db_session()
        if not session:
            return False
        
        try:
            oauth_token = session.query(OAuthToken).filter(
                OAuthToken.platform == self.platform,
                OAuthToken.account_id == account_id,
                OAuthToken.is_active == True
            ).first()
            
            if oauth_token:
                oauth_token.is_active = False
                oauth_token.updated_at = func.current_timestamp()
                session.commit()
                
                await self._log_auth_event(\"token_revoked\", account_id)
                logger.info(f\"Token revoked for {self.platform} account {account_id}\")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f\"Error revoking token for {self.platform}: {e}\")
            session.rollback()
            return False
        finally:
            session.close()

class AdPlatformClient(ABC):
    """Abstract base class for ad platform clients."""
    
    def __init__(self, platform: AdPlatform):
        self.platform = platform
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 100
        self.rate_limit_reset_time = time.time() + 3600
        self.token_manager = OAuth2TokenManager(platform.value)
        self.oauth_config: Optional[OAuth2Config] = None
        self.account_id: Optional[str] = None  # Set by platform-specific clients
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.ad_platforms.request_timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the platform using OAuth2."""
        pass
    
    async def get_oauth_authorization_url(self, state: str = None) -> str:
        """Generate OAuth2 authorization URL for user consent."""
        if not self.oauth_config:
            raise ConfigurationError(f"OAuth2 not configured for {self.platform.value}")
        
        if not state:
            state = oauth_security.generate_state_token()
        
        params = {
            'client_id': self.oauth_config.client_id,
            'redirect_uri': self.oauth_config.redirect_uri,
            'scope': ' '.join(self.oauth_config.scope),
            'response_type': 'code',
            'state': state,
            **self.oauth_config.extra_params
        }
        
        return f"{self.oauth_config.authorization_url}?{urlencode(params)}"
    
    async def handle_oauth_callback(self, authorization_code: str, state: str = None) -> bool:
        """Handle OAuth2 callback and exchange code for tokens."""
        if not self.oauth_config:
            raise ConfigurationError(f"OAuth2 not configured for {self.platform.value}")
        
        try:
            # Exchange authorization code for access token
            token_data = {
                'client_id': self.oauth_config.client_id,
                'client_secret': self.oauth_config.client_secret,
                'redirect_uri': self.oauth_config.redirect_uri,
                'code': authorization_code,
                'grant_type': 'authorization_code'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.oauth_config.token_url, data=token_data) as response:
                    if response.status == 200:
                        tokens = await response.json()
                        
                        # Store tokens securely
                        success = await self.token_manager.store_token(
                            account_id=self.account_id or "default",
                            access_token=tokens['access_token'],
                            refresh_token=tokens.get('refresh_token'),
                            expires_in=tokens.get('expires_in'),
                            scope=tokens.get('scope')
                        )
                        
                        if success:
                            await self.token_manager._log_auth_event("oauth_callback_success", 
                                                                   self.account_id or "default",
                                                                   {"state": state})
                            logger.info(f"OAuth2 tokens stored successfully for {self.platform.value}")
                            return True
                        else:
                            logger.error(f"Failed to store OAuth2 tokens for {self.platform.value}")
                            return False
                    else:
                        error_data = await response.text()
                        logger.error(f"OAuth2 token exchange failed for {self.platform.value}: {error_data}")
                        await self.token_manager._log_auth_event("oauth_callback_failed", 
                                                               self.account_id or "default",
                                                               {"error": error_data, "state": state})
                        return False
                        
        except Exception as e:
            logger.error(f"OAuth2 callback handling failed for {self.platform.value}: {e}")
            await self.token_manager._log_auth_event("oauth_callback_error", 
                                                   self.account_id or "default",
                                                   {"error": str(e), "state": state})
            return False
    
    async def get_valid_access_token(self) -> Optional[str]:
        """Get valid access token, refreshing if necessary."""
        if not self.account_id:
            logger.warning(f"Account ID not set for {self.platform.value}")
            return None
        
        return await self.token_manager.get_valid_token(self.account_id)
    
    async def revoke_access(self) -> bool:
        """Revoke OAuth2 access for this platform."""
        if not self.account_id:
            logger.warning(f"Account ID not set for {self.platform.value}")
            return False
        
        return await self.token_manager.revoke_token(self.account_id)

class MetaTokenManager(OAuth2TokenManager):
    \"\"\"Meta-specific OAuth2 token manager.\"\"\"
    
    def __init__(self):
        super().__init__(\"meta\")
    
    async def _platform_refresh_token(self, oauth_token: 'OAuthToken') -> Optional[Dict[str, Any]]:
        \"\"\"Refresh Meta access token using refresh token.\"\"\"
        refresh_token = oauth_token.get_refresh_token()
        if not refresh_token:
            return None
        
        try:
            token_url = \"https://graph.facebook.com/oauth/access_token\"
            
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': config.ad_platforms.meta_app_id,
                'client_secret': config.ad_platforms.meta_app_secret
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        return {
                            'access_token': token_data['access_token'],
                            'expires_in': token_data.get('expires_in', 3600)
                        }
                    else:
                        logger.error(f\"Meta token refresh failed: {response.status}\")
                        return None
                        
        except Exception as e:
            logger.error(f\"Meta token refresh error: {e}\")
            return None

class TikTokTokenManager(OAuth2TokenManager):
    \"\"\"TikTok-specific OAuth2 token manager.\"\"\"
    
    def __init__(self):
        super().__init__(\"tiktok\")
    
    async def _platform_refresh_token(self, oauth_token: 'OAuthToken') -> Optional[Dict[str, Any]]:
        \"\"\"Refresh TikTok access token using refresh token.\"\"\"
        refresh_token = oauth_token.get_refresh_token()
        if not refresh_token:
            return None
        
        try:
            token_url = \"https://business-api.tiktok.com/open_api/v1.3/oauth2/refresh_token/\"
            
            data = {
                'app_id': config.ad_platforms.tiktok_app_id,
                'secret': config.ad_platforms.tiktok_secret,
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, json=data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if response_data.get('code') == 0:
                            token_info = response_data['data']
                            return {
                                'access_token': token_info['access_token'],
                                'refresh_token': token_info.get('refresh_token'),
                                'expires_in': token_info.get('expires_in', 86400)
                            }
                        else:
                            logger.error(f\"TikTok token refresh failed: {response_data.get('message')}\")
                            return None
                    else:
                        logger.error(f\"TikTok token refresh failed: {response.status}\")
                        return None
                        
        except Exception as e:
            logger.error(f\"TikTok token refresh error: {e}\")
            return None

class GoogleAdsTokenManager(OAuth2TokenManager):
    \"\"\"Google Ads-specific OAuth2 token manager.\"\"\"
    
    def __init__(self):
        super().__init__(\"google_ads\")
    
    async def _platform_refresh_token(self, oauth_token: 'OAuthToken') -> Optional[Dict[str, Any]]:
        \"\"\"Refresh Google Ads access token using refresh token.\"\"\"
        refresh_token = oauth_token.get_refresh_token()
        if not refresh_token:
            return None
        
        try:
            token_url = \"https://oauth2.googleapis.com/token\"
            
            data = {
                'client_id': config.ad_platforms.google_ads_client_id,
                'client_secret': config.ad_platforms.google_ads_client_secret,
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        return {
                            'access_token': token_data['access_token'],
                            'expires_in': token_data.get('expires_in', 3600)
                        }
                    else:
                        logger.error(f\"Google Ads token refresh failed: {response.status}\")
                        return None
                        
        except Exception as e:
            logger.error(f\"Google Ads token refresh error: {e}\")
            return None
    
    @abstractmethod
    async def create_campaign(self, campaign_config: CampaignConfig) -> Dict[str, Any]:
        """Create a new campaign."""
        pass
    
    @abstractmethod
    async def upload_creative(self, creative: CreativeAsset) -> str:
        """Upload a creative asset and return asset ID."""
        pass
    
    @abstractmethod
    async def get_performance_data(self, campaign_id: str, start_date: datetime, end_date: datetime) -> List[AdPerformanceData]:
        """Get performance data for a campaign."""
        pass
    
    @abstractmethod
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause a campaign."""
        pass
    
    @abstractmethod
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume a paused campaign."""
        pass
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make rate-limited HTTP request."""
        try:
            # Check rate limit
            await self._check_rate_limit()
            
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")
            
            async with self.session.request(method, url, **kwargs) as response:
                # Update rate limit info from headers
                self._update_rate_limit(response.headers)
                
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"API request failed: {method} {url}", error=str(e), platform=self.platform.value)
            raise
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        
        if current_time < self.rate_limit_reset_time and self.rate_limit_remaining <= 0:
            sleep_time = self.rate_limit_reset_time - current_time
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds", platform=self.platform.value)
            await asyncio.sleep(sleep_time)
    
    def _update_rate_limit(self, headers: Dict[str, str]):
        """Update rate limit info from response headers."""
        # This would be platform-specific header parsing
        pass

class MetaAdClient(AdPlatformClient):
    """Meta (Facebook/Instagram) Ad Platform Client with OAuth2 support."""
    
    def __init__(self):
        super().__init__(AdPlatform.META)
        self.token_manager = MetaTokenManager()
        self.app_id = config.ad_platforms.meta_app_id
        self.app_secret = config.ad_platforms.meta_app_secret
        self.base_url = config.ad_platforms.meta_base_url
        self.api_version = config.ad_platforms.meta_api_version
        self.account_id = None  # Will be set after authentication
        
        # Configure OAuth2
        if self.app_id and self.app_secret:
            self.oauth_config = OAuth2Config(
                client_id=self.app_id,
                client_secret=self.app_secret,
                redirect_uri=config.ad_platforms.meta_redirect_uri if hasattr(config.ad_platforms, 'meta_redirect_uri') else 'https://localhost:8080/auth/meta/callback',
                authorization_url='https://www.facebook.com/dialog/oauth',
                token_url='https://graph.facebook.com/oauth/access_token',
                scope=['ads_management', 'ads_read', 'business_management'],
                extra_params={'response_type': 'code'}
            )
        else:
            logger.warning("Meta OAuth2 not fully configured - missing app_id or app_secret")
    
    async def authenticate(self) -> bool:
        """Authenticate with Meta API using OAuth2."""
        try:
            # Check if we have a valid token from OAuth2 flow
            access_token = await self.get_valid_access_token()
            
            if not access_token:
                # Fall back to configured token for backwards compatibility
                access_token = config.ad_platforms.meta_access_token
                if access_token and self.account_id:
                    # Store the configured token for future use
                    await self.token_manager.store_token(
                        account_id=self.account_id,
                        access_token=access_token,
                        expires_in=86400  # 24 hours default
                    )
            
            if not access_token:
                logger.error("No Meta access token available. Please complete OAuth2 flow or configure token.")
                return False
            
            # Test the token by making a request
            url = f"{self.base_url}/{self.api_version}/me"
            params = {'access_token': access_token, 'fields': 'id,name'}
            
            response = await self._make_request('GET', url, params=params)
            
            if response:
                # Set account ID from the response
                self.account_id = response.get('id', 'default')
                
                # Update token with account info if we got it from config
                if not await self.token_manager.get_valid_token(self.account_id):
                    await self.token_manager.store_token(
                        account_id=self.account_id,
                        access_token=access_token,
                        expires_in=86400
                    )
                
                logger.info(f"Meta authentication successful", user_id=self.account_id)
                await self.token_manager._log_auth_event("authentication_success", self.account_id, 
                                                       {"user_name": response.get('name')})
                return True
            else:
                logger.error("Meta authentication failed: No response from API")
                await self.token_manager._log_auth_event("authentication_failed", 
                                                       self.account_id or "unknown",
                                                       {"error": "No response from API"})
                return False
                
        except Exception as e:
            logger.error(f"Meta authentication failed: {e}")
            await self.token_manager._log_auth_event("authentication_error", 
                                                   self.account_id or "unknown",
                                                   {"error": str(e)})
            return False
    
    async def create_campaign(self, campaign_config: CampaignConfig) -> Dict[str, Any]:
        """Create a Meta ad campaign."""
        try:
            # Create campaign
            campaign_data = {
                'name': campaign_config.campaign_name,
                'objective': campaign_config.objective,
                'status': 'PAUSED',  # Start paused for review
                'daily_budget': int(campaign_config.budget_daily * 100),  # Convert to cents
                'access_token': self.access_token
            }
            
            # Use demo account for testing
            ad_account_id = "act_123456789"  # Would be from config
            url = f"{self.base_url}/{self.api_version}/{ad_account_id}/campaigns"
            
            campaign_response = await self._make_request('POST', url, data=campaign_data)
            campaign_id = campaign_response['id']
            
            logger.info(f"Meta campaign created", campaign_id=campaign_id, name=campaign_config.campaign_name)
            
            return {
                'campaign_id': campaign_id,
                'platform': 'meta',
                'status': 'created',
                'campaign_name': campaign_config.campaign_name
            }
            
        except Exception as e:
            logger.error(f"Failed to create Meta campaign: {e}")
            raise
    
    async def upload_creative(self, creative: CreativeAsset) -> str:
        """Upload creative to Meta."""
        try:
            if creative.asset_type == 'video':
                return await self._upload_video_creative(creative)
            else:
                return await self._upload_image_creative(creative)
                
        except Exception as e:
            logger.error(f"Failed to upload creative to Meta: {e}")
            raise
    
    async def _upload_video_creative(self, creative: CreativeAsset) -> str:
        """Upload video creative to Meta."""
        # Simplified video upload - would need actual file handling
        ad_account_id = "act_123456789"
        url = f"{self.base_url}/{self.api_version}/{ad_account_id}/advideos"
        
        data = {
            'access_token': self.access_token,
            'name': f'Video_{creative.asset_id}',
            'source': creative.asset_url  # Would be file upload in production
        }
        
        response = await self._make_request('POST', url, data=data)
        return response['id']
    
    async def _upload_image_creative(self, creative: CreativeAsset) -> str:
        """Upload image creative to Meta."""
        ad_account_id = "act_123456789"
        url = f"{self.base_url}/{self.api_version}/{ad_account_id}/adimages"
        
        data = {
            'access_token': self.access_token,
            'name': f'Image_{creative.asset_id}',
            'source': creative.asset_url
        }
        
        response = await self._make_request('POST', url, data=data)
        return response['images']['source']['hash']
    
    async def get_performance_data(self, campaign_id: str, start_date: datetime, end_date: datetime) -> List[AdPerformanceData]:
        """Get Meta campaign performance data."""
        try:
            url = f"{self.base_url}/{self.api_version}/{campaign_id}/insights"
            
            params = {
                'access_token': self.access_token,
                'time_range': json.dumps({
                    'since': start_date.strftime('%Y-%m-%d'),
                    'until': end_date.strftime('%Y-%m-%d')
                }),
                'fields': 'impressions,clicks,spend,conversions,ctr,cpc,cpm,video_avg_time_watched_actions',
                'level': 'ad',
                'limit': 100
            }
            
            response = await self._make_request('GET', url, params=params)
            
            performance_data = []
            for ad_data in response.get('data', []):
                performance = AdPerformanceData(
                    ad_id=ad_data.get('ad_id', ''),
                    campaign_id=campaign_id,
                    platform='meta',
                    timestamp=datetime.utcnow(),
                    impressions=int(ad_data.get('impressions', 0)),
                    clicks=int(ad_data.get('clicks', 0)),
                    conversions=int(ad_data.get('conversions', 0)),
                    spend=float(ad_data.get('spend', 0)),
                    ctr=float(ad_data.get('ctr', 0)),
                    cpc=float(ad_data.get('cpc', 0)),
                    cpm=float(ad_data.get('cpm', 0))
                )
                
                # Calculate additional metrics
                if performance.impressions > 0:
                    performance.ctr = performance.clicks / performance.impressions
                if performance.clicks > 0:
                    performance.conversion_rate = performance.conversions / performance.clicks
                if performance.spend > 0:
                    performance.roas = (performance.conversions * 50) / performance.spend  # Assuming $50 AOV
                
                performance_data.append(performance)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get Meta performance data: {e}")
            return []
    
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause Meta campaign."""
        try:
            url = f"{self.base_url}/{self.api_version}/{campaign_id}"
            data = {
                'access_token': self.access_token,
                'status': 'PAUSED'
            }
            
            await self._make_request('POST', url, data=data)
            logger.info(f"Meta campaign paused", campaign_id=campaign_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause Meta campaign: {e}")
            return False
    
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume Meta campaign."""
        try:
            url = f"{self.base_url}/{self.api_version}/{campaign_id}"
            data = {
                'access_token': self.access_token,
                'status': 'ACTIVE'
            }
            
            await self._make_request('POST', url, data=data)
            logger.info(f"Meta campaign resumed", campaign_id=campaign_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume Meta campaign: {e}")
            return False

class TikTokAdClient(AdPlatformClient):
    """TikTok for Business Ad Platform Client."""
    
    def __init__(self):
        super().__init__(AdPlatform.TIKTOK)
        self.access_token = config.ad_platforms.tiktok_access_token
        self.app_id = config.ad_platforms.tiktok_app_id
        self.secret = config.ad_platforms.tiktok_secret
        self.base_url = config.ad_platforms.tiktok_base_url
        self.api_version = config.ad_platforms.tiktok_api_version
        
        if not self.access_token:
            raise ConfigurationError("TikTok access token not configured")
    
    async def authenticate(self) -> bool:
        """Authenticate with TikTok API."""
        try:
            url = f"{self.base_url}/{self.api_version}/oauth2/advertiser/get/"
            headers = {
                'Access-Token': self.access_token,
                'Content-Type': 'application/json'
            }
            
            response = await self._make_request('GET', url, headers=headers)
            if response.get('code') == 0:
                logger.info("TikTok authentication successful")
                return True
            else:
                logger.error(f"TikTok authentication failed: {response.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"TikTok authentication failed: {e}")
            return False
    
    async def create_campaign(self, campaign_config: CampaignConfig) -> Dict[str, Any]:
        """Create a TikTok ad campaign."""
        try:
            advertiser_id = "123456789"  # Would be from config
            url = f"{self.base_url}/{self.api_version}/campaign/create/"
            
            headers = {
                'Access-Token': self.access_token,
                'Content-Type': 'application/json'
            }
            
            data = {
                'advertiser_id': advertiser_id,
                'campaign_name': campaign_config.campaign_name,
                'objective_type': campaign_config.objective,
                'budget_mode': 'BUDGET_MODE_DAY',
                'budget': campaign_config.budget_daily,
                'status': 'DISABLE'  # Start disabled for review
            }
            
            response = await self._make_request('POST', url, headers=headers, json=data)
            
            if response.get('code') == 0:
                campaign_id = response['data']['campaign_id']
                logger.info(f"TikTok campaign created", campaign_id=campaign_id)
                
                return {
                    'campaign_id': campaign_id,
                    'platform': 'tiktok',
                    'status': 'created',
                    'campaign_name': campaign_config.campaign_name
                }
            else:
                raise Exception(f"TikTok API error: {response.get('message')}")
                
        except Exception as e:
            logger.error(f"Failed to create TikTok campaign: {e}")
            raise
    
    async def upload_creative(self, creative: CreativeAsset) -> str:
        """Upload creative to TikTok."""
        try:
            advertiser_id = "123456789"
            url = f"{self.base_url}/{self.api_version}/file/video/ad/upload/"
            
            headers = {
                'Access-Token': self.access_token
            }
            
            # In production, would handle actual file upload
            data = {
                'advertiser_id': advertiser_id,
                'video_file': creative.asset_url,  # Would be file data
                'video_signature': hashlib.md5(creative.asset_url.encode()).hexdigest() if creative.asset_url else ''
            }
            
            response = await self._make_request('POST', url, headers=headers, data=data)
            
            if response.get('code') == 0:
                return response['data']['video_id']
            else:
                raise Exception(f"TikTok upload error: {response.get('message')}")
                
        except Exception as e:
            logger.error(f"Failed to upload creative to TikTok: {e}")
            raise
    
    async def get_performance_data(self, campaign_id: str, start_date: datetime, end_date: datetime) -> List[AdPerformanceData]:
        """Get TikTok campaign performance data."""
        try:
            advertiser_id = "123456789"
            url = f"{self.base_url}/{self.api_version}/report/integrated/get/"
            
            headers = {
                'Access-Token': self.access_token,
                'Content-Type': 'application/json'
            }
            
            data = {
                'advertiser_id': advertiser_id,
                'report_type': 'BASIC',
                'dimensions': ['campaign_id', 'ad_id'],
                'metrics': ['impressions', 'clicks', 'spend', 'conversions', 'ctr', 'cpc', 'cpm'],
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'filters': [{'field_name': 'campaign_id', 'filter_type': 'IN', 'filter_value': [campaign_id]}]
            }
            
            response = await self._make_request('POST', url, headers=headers, json=data)
            
            performance_data = []
            if response.get('code') == 0:
                for row in response.get('data', {}).get('list', []):
                    metrics = row.get('metrics', {})
                    dimensions = row.get('dimensions', {})
                    
                    performance = AdPerformanceData(
                        ad_id=dimensions.get('ad_id', ''),
                        campaign_id=campaign_id,
                        platform='tiktok',
                        timestamp=datetime.utcnow(),
                        impressions=int(metrics.get('impressions', 0)),
                        clicks=int(metrics.get('clicks', 0)),
                        conversions=int(metrics.get('conversions', 0)),
                        spend=float(metrics.get('spend', 0)),
                        ctr=float(metrics.get('ctr', 0)),
                        cpc=float(metrics.get('cpc', 0)),
                        cpm=float(metrics.get('cpm', 0))
                    )
                    
                    # TikTok typically has higher engagement rates
                    if performance.impressions > 0:
                        performance.engagement_rate = min(0.1, performance.clicks / performance.impressions * 2)
                    
                    performance_data.append(performance)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get TikTok performance data: {e}")
            return []
    
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause TikTok campaign."""
        try:
            url = f"{self.base_url}/{self.api_version}/campaign/update/status/"
            
            headers = {
                'Access-Token': self.access_token,
                'Content-Type': 'application/json'
            }
            
            data = {
                'advertiser_id': "123456789",
                'campaign_ids': [campaign_id],
                'status': 'DISABLE'
            }
            
            response = await self._make_request('POST', url, headers=headers, json=data)
            
            if response.get('code') == 0:
                logger.info(f"TikTok campaign paused", campaign_id=campaign_id)
                return True
            else:
                logger.error(f"Failed to pause TikTok campaign: {response.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to pause TikTok campaign: {e}")
            return False
    
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume TikTok campaign."""
        try:
            url = f"{self.base_url}/{self.api_version}/campaign/update/status/"
            
            headers = {
                'Access-Token': self.access_token,
                'Content-Type': 'application/json'
            }
            
            data = {
                'advertiser_id': "123456789",
                'campaign_ids': [campaign_id],
                'status': 'ENABLE'
            }
            
            response = await self._make_request('POST', url, headers=headers, json=data)
            
            if response.get('code') == 0:
                logger.info(f"TikTok campaign resumed", campaign_id=campaign_id)
                return True
            else:
                logger.error(f"Failed to resume TikTok campaign: {response.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resume TikTok campaign: {e}")
            return False

class GoogleAdsClient(AdPlatformClient):
    """Google Ads Platform Client."""
    
    def __init__(self):
        super().__init__(AdPlatform.GOOGLE_ADS)
        self.customer_id = config.ad_platforms.google_ads_customer_id
        self.developer_token = config.ad_platforms.google_ads_developer_token
        self.client_id = config.ad_platforms.google_ads_client_id
        self.client_secret = config.ad_platforms.google_ads_client_secret
        self.refresh_token = config.ad_platforms.google_ads_refresh_token
        self.base_url = config.ad_platforms.google_ads_base_url
        self.api_version = config.ad_platforms.google_ads_api_version
        self.access_token = None
        
        if not all([self.customer_id, self.developer_token, self.client_id]):
            raise ConfigurationError("Google Ads credentials not fully configured")
    
    async def authenticate(self) -> bool:
        """Authenticate with Google Ads API."""
        try:
            # OAuth2 token refresh
            if self.refresh_token:
                token_url = "https://oauth2.googleapis.com/token"
                data = {
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'refresh_token': self.refresh_token,
                    'grant_type': 'refresh_token'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(token_url, data=data) as response:
                        if response.status == 200:
                            token_data = await response.json()
                            self.access_token = token_data.get('access_token')
                            logger.info("Google Ads authentication successful")
                            return True
                        else:
                            logger.error(f"Google Ads token refresh failed: {response.status}")
                            return False
            
            return False
            
        except Exception as e:
            logger.error(f"Google Ads authentication failed: {e}")
            return False
    
    async def create_campaign(self, campaign_config: CampaignConfig) -> Dict[str, Any]:
        """Create a Google Ads campaign."""
        try:
            url = f"{self.base_url}/{self.api_version}/customers/{self.customer_id}/campaigns:mutate"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'developer-token': self.developer_token,
                'Content-Type': 'application/json'
            }
            
            # Simplified campaign creation
            operation = {
                'create': {
                    'name': campaign_config.campaign_name,
                    'advertising_channel_type': 'DISPLAY',  # or VIDEO for video campaigns
                    'status': 'PAUSED',
                    'manual_cpc': {},
                    'campaign_budget': f"customers/{self.customer_id}/campaignBudgets/budget_id"
                }
            }
            
            data = {
                'operations': [operation]
            }
            
            response = await self._make_request('POST', url, headers=headers, json=data)
            
            if 'results' in response:
                campaign_resource = response['results'][0]['resourceName']
                campaign_id = campaign_resource.split('/')[-1]
                
                logger.info(f"Google Ads campaign created", campaign_id=campaign_id)
                
                return {
                    'campaign_id': campaign_id,
                    'platform': 'google_ads',
                    'status': 'created',
                    'campaign_name': campaign_config.campaign_name
                }
            else:
                raise Exception(f"Google Ads campaign creation failed: {response}")
                
        except Exception as e:
            logger.error(f"Failed to create Google Ads campaign: {e}")
            raise
    
    async def upload_creative(self, creative: CreativeAsset) -> str:
        """Upload creative to Google Ads."""
        try:
            # For Google Ads, this would involve creating ad assets
            url = f"{self.base_url}/{self.api_version}/customers/{self.customer_id}/assets:mutate"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'developer-token': self.developer_token,
                'Content-Type': 'application/json'
            }
            
            asset_data = {
                'name': f'Asset_{creative.asset_id}',
                'type_': 'IMAGE' if creative.asset_type == 'image' else 'VIDEO',
                'image_asset': {'data': creative.asset_url} if creative.asset_type == 'image' else None,
                'youtube_video_asset': {'youtube_video_id': creative.asset_id} if creative.asset_type == 'video' else None
            }
            
            operation = {'create': asset_data}
            data = {'operations': [operation]}
            
            response = await self._make_request('POST', url, headers=headers, json=data)
            
            if 'results' in response:
                asset_resource = response['results'][0]['resourceName']
                return asset_resource.split('/')[-1]
            else:
                raise Exception(f"Google Ads asset upload failed: {response}")
                
        except Exception as e:
            logger.error(f"Failed to upload creative to Google Ads: {e}")
            raise
    
    async def get_performance_data(self, campaign_id: str, start_date: datetime, end_date: datetime) -> List[AdPerformanceData]:
        """Get Google Ads campaign performance data."""
        try:
            url = f"{self.base_url}/{self.api_version}/customers/{self.customer_id}/googleAds:search"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'developer-token': self.developer_token,
                'Content-Type': 'application/json'
            }
            
            query = f"""
                SELECT 
                    campaign.id,
                    ad_group.id,
                    ad_group_ad.ad.id,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.cost_micros,
                    metrics.ctr,
                    metrics.average_cpc,
                    metrics.average_cpm
                FROM ad_group_ad 
                WHERE campaign.id = {campaign_id}
                AND segments.date BETWEEN '{start_date.strftime('%Y-%m-%d')}' 
                AND '{end_date.strftime('%Y-%m-%d')}'
            """
            
            data = {'query': query}
            
            response = await self._make_request('POST', url, headers=headers, json=data)
            
            performance_data = []
            if 'results' in response:
                for result in response['results']:
                    metrics = result.get('metrics', {})
                    ad_info = result.get('adGroupAd', {}).get('ad', {})
                    
                    performance = AdPerformanceData(
                        ad_id=str(ad_info.get('id', '')),
                        campaign_id=campaign_id,
                        platform='google_ads',
                        timestamp=datetime.utcnow(),
                        impressions=int(metrics.get('impressions', 0)),
                        clicks=int(metrics.get('clicks', 0)),
                        conversions=int(metrics.get('conversions', 0)),
                        spend=float(metrics.get('costMicros', 0)) / 1000000,  # Convert micros to dollars
                        ctr=float(metrics.get('ctr', 0)),
                        cpc=float(metrics.get('averageCpc', 0)) / 1000000,
                        cpm=float(metrics.get('averageCpm', 0)) / 1000000
                    )
                    
                    performance_data.append(performance)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get Google Ads performance data: {e}")
            return []
    
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause Google Ads campaign."""
        try:
            url = f"{self.base_url}/{self.api_version}/customers/{self.customer_id}/campaigns:mutate"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'developer-token': self.developer_token,
                'Content-Type': 'application/json'
            }
            
            operation = {
                'update': {
                    'resourceName': f"customers/{self.customer_id}/campaigns/{campaign_id}",
                    'status': 'PAUSED'
                },
                'updateMask': 'status'
            }
            
            data = {'operations': [operation]}
            
            await self._make_request('POST', url, headers=headers, json=data)
            logger.info(f"Google Ads campaign paused", campaign_id=campaign_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause Google Ads campaign: {e}")
            return False
    
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume Google Ads campaign."""
        try:
            url = f"{self.base_url}/{self.api_version}/customers/{self.customer_id}/campaigns:mutate"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'developer-token': self.developer_token,
                'Content-Type': 'application/json'
            }
            
            operation = {
                'update': {
                    'resourceName': f"customers/{self.customer_id}/campaigns/{campaign_id}",
                    'status': 'ENABLED'
                },
                'updateMask': 'status'
            }
            
            data = {'operations': [operation]}
            
            await self._make_request('POST', url, headers=headers, json=data)
            logger.info(f"Google Ads campaign resumed", campaign_id=campaign_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume Google Ads campaign: {e}")
            return False

class AdPlatformManager:
    """Centralized manager for all ad platform integrations."""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available ad platform clients based on configuration."""
        try:
            if config.ad_platforms.enable_meta_integration:
                self.clients[AdPlatform.META] = MetaAdClient
                
            if config.ad_platforms.enable_tiktok_integration:
                self.clients[AdPlatform.TIKTOK] = TikTokAdClient
                
            if config.ad_platforms.enable_google_ads_integration:
                self.clients[AdPlatform.GOOGLE_ADS] = GoogleAdsClient
                
            logger.info(f"Ad platform manager initialized with {len(self.clients)} platforms")
            
        except Exception as e:
            logger.error(f"Failed to initialize ad platform clients: {e}")
    
    async def get_client(self, platform: AdPlatform) -> AdPlatformClient:
        """Get authenticated client for specified platform."""
        if platform not in self.clients:
            raise ValueError(f"Platform {platform.value} not configured or supported")
        
        client_class = self.clients[platform]
        client = client_class()
        
        # Authenticate the client
        async with client:
            if await client.authenticate():
                return client
            else:
                raise RuntimeError(f"Authentication failed for {platform.value}")
    
    async def sync_performance_data(self, hours_back: int = 24) -> Dict[str, List[AdPerformanceData]]:
        """Sync performance data from all configured platforms."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=hours_back)
            
            all_performance_data = {}
            
            for platform in self.clients.keys():
                try:
                    async with await self.get_client(platform) as client:
                        # In production, would get campaign IDs from database
                        campaign_ids = ["demo_campaign_123"]  
                        
                        platform_data = []
                        for campaign_id in campaign_ids:
                            campaign_data = await client.get_performance_data(campaign_id, start_date, end_date)
                            platform_data.extend(campaign_data)
                        
                        all_performance_data[platform.value] = platform_data
                        
                        # Record in monitoring system
                        for data in platform_data:
                            ad_metrics = AdPerformanceMetrics(
                                timestamp=data.timestamp,
                                ad_id=data.ad_id,
                                platform=data.platform,
                                campaign_id=data.campaign_id,
                                impressions=data.impressions,
                                clicks=data.clicks,
                                conversions=data.conversions,
                                spend=data.spend,
                                ctr=data.ctr,
                                cpc=data.cpc,
                                cpm=data.cpm,
                                conversion_rate=data.conversion_rate,
                                roas=data.roas,
                                video_views=data.video_views,
                                video_completion_rate=data.video_completion_rate,
                                engagement_rate=data.engagement_rate,
                                creative_id=data.creative_id,
                                template_id=data.template_id,
                                niche_type=data.niche_type
                            )
                            
                            record_ad_performance_metrics(ad_metrics)
                        
                        logger.info(f"Synced {len(platform_data)} performance records from {platform.value}")
                        
                except Exception as e:
                    logger.error(f"Failed to sync data from {platform.value}: {e}")
                    all_performance_data[platform.value] = []
            
            return all_performance_data
            
        except Exception as e:
            logger.error(f"Performance data sync failed: {e}")
            return {}
    
    async def create_multi_platform_campaign(self, campaign_config: CampaignConfig, platforms: List[AdPlatform]) -> Dict[str, Dict[str, Any]]:
        """Create campaign across multiple platforms."""
        try:
            results = {}
            
            for platform in platforms:
                if platform in self.clients:
                    try:
                        async with await self.get_client(platform) as client:
                            # Upload creatives first
                            uploaded_creatives = []
                            for creative in campaign_config.creative_assets:
                                creative_id = await client.upload_creative(creative)
                                uploaded_creatives.append(creative_id)
                            
                            # Create campaign
                            campaign_result = await client.create_campaign(campaign_config)
                            campaign_result['uploaded_creatives'] = uploaded_creatives
                            
                            results[platform.value] = campaign_result
                            
                    except Exception as e:
                        logger.error(f"Failed to create campaign on {platform.value}: {e}")
                        results[platform.value] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-platform campaign creation failed: {e}")
            return {}
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported and configured platforms."""
        return [platform.value for platform in self.clients.keys()]
    
    def get_platform_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all configured platforms."""
        status = {}
        
        for platform in AdPlatform:
            platform_config = {
                'configured': platform in self.clients,
                'enabled': False
            }
            
            if platform == AdPlatform.META:
                platform_config['enabled'] = config.ad_platforms.enable_meta_integration
                platform_config['has_credentials'] = bool(config.ad_platforms.meta_access_token)
            elif platform == AdPlatform.TIKTOK:
                platform_config['enabled'] = config.ad_platforms.enable_tiktok_integration
                platform_config['has_credentials'] = bool(config.ad_platforms.tiktok_access_token)
            elif platform == AdPlatform.GOOGLE_ADS:
                platform_config['enabled'] = config.ad_platforms.enable_google_ads_integration
                platform_config['has_credentials'] = bool(config.ad_platforms.google_ads_customer_id)
            
            status[platform.value] = platform_config
        
        return status

# Global ad platform manager instance
ad_platform_manager = AdPlatformManager()

# Convenience functions for easy access
async def sync_ad_performance_data(hours_back: int = 24) -> Dict[str, List[AdPerformanceData]]:
    """Sync performance data from all configured ad platforms."""
    return await ad_platform_manager.sync_performance_data(hours_back)

async def create_campaign_on_platforms(campaign_config: CampaignConfig, platforms: List[str]) -> Dict[str, Dict[str, Any]]:
    """Create campaign across specified platforms."""
    platform_enums = [AdPlatform(p) for p in platforms if p in [p.value for p in AdPlatform]]
    return await ad_platform_manager.create_multi_platform_campaign(campaign_config, platform_enums)

def get_ad_platform_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all ad platform integrations."""
    return ad_platform_manager.get_platform_status()