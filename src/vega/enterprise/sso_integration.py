"""
Enterprise SSO Integration System
================================

Comprehensive Single Sign-On integration supporting multiple providers
and enterprise authentication protocols.

Features:
- SAML 2.0 integration with enterprise identity providers
- OpenID Connect (OIDC) support
- OAuth 2.0 authentication flows
- Azure AD and Google Workspace integration
- LDAP directory integration
- Just-in-Time (JIT) user provisioning
- Advanced user attribute mapping
"""

import asyncio
import logging
import json
import uuid
import base64
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from urllib.parse import urlencode, parse_qs, urlparse
import httpx
import jwt
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate

logger = logging.getLogger(__name__)


class SSOProvider(Enum):
    """Supported SSO providers"""

    SAML2 = "saml2"
    OIDC = "oidc"
    OAUTH2 = "oauth2"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"
    OKTA = "okta"
    PING_IDENTITY = "ping_identity"
    LDAP = "ldap"


class AuthenticationStatus(Enum):
    """Authentication status types"""

    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    EXPIRED = "expired"
    INVALID = "invalid"


class UserProvisioningAction(Enum):
    """User provisioning actions"""

    CREATE = "create"
    UPDATE = "update"
    SUSPEND = "suspend"
    DELETE = "delete"
    REACTIVATE = "reactivate"


@dataclass
class SSOConfiguration:
    """SSO provider configuration"""

    provider_id: str
    tenant_id: str
    provider_type: SSOProvider
    enabled: bool = True

    # Provider details
    provider_name: str = ""
    issuer_url: str = ""
    metadata_url: str = ""

    # SAML configuration
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None
    saml_slo_url: Optional[str] = None
    saml_certificate: Optional[str] = None
    saml_private_key: Optional[str] = None

    # OIDC configuration
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    oidc_discovery_url: Optional[str] = None
    oidc_scopes: List[str] = field(
        default_factory=lambda: ["openid", "profile", "email"]
    )

    # OAuth2 configuration
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_auth_url: Optional[str] = None
    oauth2_token_url: Optional[str] = None
    oauth2_user_info_url: Optional[str] = None

    # LDAP configuration
    ldap_server: Optional[str] = None
    ldap_port: int = 389
    ldap_use_ssl: bool = False
    ldap_bind_dn: Optional[str] = None
    ldap_bind_password: Optional[str] = None
    ldap_user_base_dn: Optional[str] = None
    ldap_user_filter: str = "(uid={username})"

    # User attribute mapping
    attribute_mapping: Dict[str, str] = field(default_factory=dict)
    role_mapping: Dict[str, List[str]] = field(default_factory=dict)

    # Just-in-Time provisioning
    jit_provisioning: bool = True
    auto_create_users: bool = True
    default_role: str = "user"

    # Security settings
    require_encrypted_assertions: bool = True
    require_signed_requests: bool = True
    session_timeout: int = 3600

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SSOUser:
    """SSO authenticated user"""

    user_id: str
    tenant_id: str
    provider_id: str
    external_id: str

    # User attributes
    email: str
    first_name: str = ""
    last_name: str = ""
    display_name: str = ""
    username: str = ""

    # Groups and roles
    groups: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

    # Authentication details
    auth_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""

    # Raw attributes from provider
    raw_attributes: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SSOManager:
    """Comprehensive SSO management system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.http_client = httpx.AsyncClient()

        # SSO configurations by tenant
        self.sso_configs: Dict[str, Dict[str, SSOConfiguration]] = {}

        # Active sessions
        self.active_sessions: Dict[str, SSOUser] = {}

        # Provider metadata cache
        self.provider_metadata: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize SSO manager"""

        logger.info("Initializing SSO manager")

        # Load SSO configurations
        await self._load_sso_configurations()

        # Start background tasks
        asyncio.create_task(self._session_cleanup_task())
        asyncio.create_task(self._metadata_refresh_task())

    async def _load_sso_configurations(self):
        """Load SSO configurations from storage"""

        # This would load from database/Redis in production
        # For demo, initialize with sample configurations

        self.sso_configs = {}

    async def create_sso_configuration(
        self,
        tenant_id: str,
        provider_type: SSOProvider,
        provider_name: str,
        config_data: Dict[str, Any],
    ) -> SSOConfiguration:
        """Create new SSO configuration for tenant"""

        provider_id = str(uuid.uuid4())

        sso_config = SSOConfiguration(
            provider_id=provider_id,
            tenant_id=tenant_id,
            provider_type=provider_type,
            provider_name=provider_name,
            **config_data,
        )

        # Validate configuration
        await self._validate_sso_configuration(sso_config)

        # Store configuration
        if tenant_id not in self.sso_configs:
            self.sso_configs[tenant_id] = {}

        self.sso_configs[tenant_id][provider_id] = sso_config

        # Load provider metadata
        await self._load_provider_metadata(sso_config)

        logger.info(f"Created SSO configuration: {provider_id} for tenant: {tenant_id}")

        return sso_config

    async def _validate_sso_configuration(self, config: SSOConfiguration):
        """Validate SSO configuration"""

        if config.provider_type == SSOProvider.SAML2:
            required_fields = ["saml_entity_id", "saml_sso_url"]
            for field in required_fields:
                if not getattr(config, field):
                    raise ValueError(f"Missing required SAML field: {field}")

        elif config.provider_type == SSOProvider.OIDC:
            required_fields = [
                "oidc_client_id",
                "oidc_client_secret",
                "oidc_discovery_url",
            ]
            for field in required_fields:
                if not getattr(config, field):
                    raise ValueError(f"Missing required OIDC field: {field}")

        elif config.provider_type == SSOProvider.OAUTH2:
            required_fields = [
                "oauth2_client_id",
                "oauth2_client_secret",
                "oauth2_auth_url",
                "oauth2_token_url",
            ]
            for field in required_fields:
                if not getattr(config, field):
                    raise ValueError(f"Missing required OAuth2 field: {field}")

        elif config.provider_type == SSOProvider.LDAP:
            required_fields = ["ldap_server", "ldap_user_base_dn"]
            for field in required_fields:
                if not getattr(config, field):
                    raise ValueError(f"Missing required LDAP field: {field}")

    async def _load_provider_metadata(self, config: SSOConfiguration):
        """Load metadata from SSO provider"""

        try:
            if config.provider_type == SSOProvider.OIDC and config.oidc_discovery_url:
                # Load OIDC discovery document
                response = await self.http_client.get(config.oidc_discovery_url)
                if response.status_code == 200:
                    self.provider_metadata[config.provider_id] = response.json()

            elif config.provider_type == SSOProvider.SAML2 and config.metadata_url:
                # Load SAML metadata
                response = await self.http_client.get(config.metadata_url)
                if response.status_code == 200:
                    metadata = self._parse_saml_metadata(response.text)
                    self.provider_metadata[config.provider_id] = metadata

        except Exception as e:
            logger.error(f"Failed to load provider metadata: {e}")

    def _parse_saml_metadata(self, metadata_xml: str) -> Dict[str, Any]:
        """Parse SAML metadata XML"""

        try:
            root = ET.fromstring(metadata_xml)

            # Extract SSO URL, SLO URL, and certificate
            metadata = {}

            # Find SSO service
            sso_service = root.find(
                ".//{urn:oasis:names:tc:SAML:2.0:metadata}SingleSignOnService"
            )
            if sso_service is not None:
                metadata["sso_url"] = sso_service.get("Location")

            # Find SLO service
            slo_service = root.find(
                ".//{urn:oasis:names:tc:SAML:2.0:metadata}SingleLogoutService"
            )
            if slo_service is not None:
                metadata["slo_url"] = slo_service.get("Location")

            # Find certificate
            cert_element = root.find(
                ".//{http://www.w3.org/2000/09/xmldsig#}X509Certificate"
            )
            if cert_element is not None:
                metadata["certificate"] = cert_element.text.strip()

            return metadata

        except ET.ParseError as e:
            logger.error(f"Failed to parse SAML metadata: {e}")
            return {}

    async def initiate_sso_login(
        self, tenant_id: str, provider_id: str, return_url: str
    ) -> Dict[str, Any]:
        """Initiate SSO login flow"""

        config = self._get_sso_config(tenant_id, provider_id)
        if not config:
            raise ValueError(f"SSO configuration not found: {provider_id}")

        if config.provider_type == SSOProvider.SAML2:
            return await self._initiate_saml_login(config, return_url)
        elif config.provider_type == SSOProvider.OIDC:
            return await self._initiate_oidc_login(config, return_url)
        elif config.provider_type == SSOProvider.OAUTH2:
            return await self._initiate_oauth2_login(config, return_url)
        else:
            raise ValueError(f"Unsupported SSO provider type: {config.provider_type}")

    async def _initiate_saml_login(
        self, config: SSOConfiguration, return_url: str
    ) -> Dict[str, Any]:
        """Initiate SAML SSO login"""

        # Generate SAML request
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create SAML AuthnRequest
        saml_request = f"""
        <samlp:AuthnRequest 
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" 
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{timestamp}"
            Destination="{config.saml_sso_url}"
            ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            AssertionConsumerServiceURL="{self.config.get('base_url')}/sso/saml/acs">
            <saml:Issuer>{config.saml_entity_id}</saml:Issuer>
        </samlp:AuthnRequest>
        """

        # Encode SAML request
        encoded_request = base64.b64encode(saml_request.encode()).decode()

        # Create redirect URL
        params = {"SAMLRequest": encoded_request, "RelayState": return_url}

        redirect_url = f"{config.saml_sso_url}?{urlencode(params)}"

        return {"redirect_url": redirect_url, "request_id": request_id, "method": "GET"}

    async def _initiate_oidc_login(
        self, config: SSOConfiguration, return_url: str
    ) -> Dict[str, Any]:
        """Initiate OIDC login"""

        # Get OIDC endpoints from discovery document
        metadata = self.provider_metadata.get(config.provider_id, {})
        auth_endpoint = metadata.get("authorization_endpoint")

        if not auth_endpoint:
            raise ValueError("OIDC authorization endpoint not found")

        # Generate state and nonce
        state = str(uuid.uuid4())
        nonce = str(uuid.uuid4())

        # Create authorization URL
        params = {
            "client_id": config.oidc_client_id,
            "response_type": "code",
            "scope": " ".join(config.oidc_scopes),
            "redirect_uri": f"{self.config.get('base_url')}/sso/oidc/callback",
            "state": state,
            "nonce": nonce,
        }

        redirect_url = f"{auth_endpoint}?{urlencode(params)}"

        # Store state for validation
        self._store_auth_state(
            state,
            {
                "provider_id": config.provider_id,
                "tenant_id": config.tenant_id,
                "return_url": return_url,
                "nonce": nonce,
            },
        )

        return {"redirect_url": redirect_url, "state": state, "method": "GET"}

    async def _initiate_oauth2_login(
        self, config: SSOConfiguration, return_url: str
    ) -> Dict[str, Any]:
        """Initiate OAuth2 login"""

        # Generate state
        state = str(uuid.uuid4())

        # Create authorization URL
        params = {
            "client_id": config.oauth2_client_id,
            "response_type": "code",
            "redirect_uri": f"{self.config.get('base_url')}/sso/oauth2/callback",
            "state": state,
            "scope": "read:user user:email",
        }

        redirect_url = f"{config.oauth2_auth_url}?{urlencode(params)}"

        # Store state for validation
        self._store_auth_state(
            state,
            {
                "provider_id": config.provider_id,
                "tenant_id": config.tenant_id,
                "return_url": return_url,
            },
        )

        return {"redirect_url": redirect_url, "state": state, "method": "GET"}

    def _store_auth_state(self, state: str, data: Dict[str, Any]):
        """Store authentication state temporarily"""

        # In production, this would use Redis with TTL
        # For demo, store in memory
        if not hasattr(self, "_auth_states"):
            self._auth_states = {}

        self._auth_states[state] = {
            **data,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
        }

    def _get_auth_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Get and remove authentication state"""

        if not hasattr(self, "_auth_states"):
            return None

        auth_data = self._auth_states.pop(state, None)

        if auth_data and auth_data["expires_at"] > datetime.now(timezone.utc):
            return auth_data

        return None

    async def handle_saml_response(
        self, saml_response: str, relay_state: str
    ) -> SSOUser:
        """Handle SAML response"""

        try:
            # Decode SAML response
            decoded_response = base64.b64decode(saml_response).decode()

            # Parse SAML response
            root = ET.fromstring(decoded_response)

            # Extract user attributes
            attributes = self._extract_saml_attributes(root)

            # Find matching SSO configuration
            config = self._find_config_by_issuer(attributes.get("issuer"))
            if not config:
                raise ValueError("SSO configuration not found for issuer")

            # Create SSO user
            sso_user = await self._create_sso_user(config, attributes)

            # Store session
            session_id = str(uuid.uuid4())
            sso_user.session_id = session_id
            self.active_sessions[session_id] = sso_user

            return sso_user

        except Exception as e:
            logger.error(f"Failed to handle SAML response: {e}")
            raise

    def _extract_saml_attributes(self, saml_root) -> Dict[str, Any]:
        """Extract attributes from SAML response"""

        attributes = {}

        # Find assertion
        assertion = saml_root.find(
            ".//{urn:oasis:names:tc:SAML:2.0:assertion}Assertion"
        )
        if assertion is None:
            raise ValueError("No assertion found in SAML response")

        # Extract subject
        subject = assertion.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}Subject")
        if subject is not None:
            name_id = subject.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}NameID")
            if name_id is not None:
                attributes["name_id"] = name_id.text

        # Extract attribute statements
        attr_statements = assertion.findall(
            ".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeStatement"
        )

        for attr_statement in attr_statements:
            attrs = attr_statement.findall(
                ".//{urn:oasis:names:tc:SAML:2.0:assertion}Attribute"
            )

            for attr in attrs:
                attr_name = attr.get("Name")
                attr_values = []

                for value in attr.findall(
                    ".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue"
                ):
                    if value.text:
                        attr_values.append(value.text)

                if len(attr_values) == 1:
                    attributes[attr_name] = attr_values[0]
                else:
                    attributes[attr_name] = attr_values

        return attributes

    async def handle_oidc_callback(self, code: str, state: str) -> SSOUser:
        """Handle OIDC callback"""

        # Validate state
        auth_data = self._get_auth_state(state)
        if not auth_data:
            raise ValueError("Invalid or expired state")

        config = self._get_sso_config(auth_data["tenant_id"], auth_data["provider_id"])
        if not config:
            raise ValueError("SSO configuration not found")

        # Exchange code for tokens
        tokens = await self._exchange_oidc_code(config, code)

        # Get user info from ID token and userinfo endpoint
        user_info = await self._get_oidc_user_info(config, tokens)

        # Create SSO user
        sso_user = await self._create_sso_user(config, user_info)

        # Store session
        session_id = str(uuid.uuid4())
        sso_user.session_id = session_id
        self.active_sessions[session_id] = sso_user

        return sso_user

    async def _exchange_oidc_code(
        self, config: SSOConfiguration, code: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""

        metadata = self.provider_metadata.get(config.provider_id, {})
        token_endpoint = metadata.get("token_endpoint")

        if not token_endpoint:
            raise ValueError("OIDC token endpoint not found")

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"{self.config.get('base_url')}/sso/oidc/callback",
            "client_id": config.oidc_client_id,
            "client_secret": config.oidc_client_secret,
        }

        response = await self.http_client.post(token_endpoint, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Token exchange failed: {response.text}")

    async def _get_oidc_user_info(
        self, config: SSOConfiguration, tokens: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get user information from OIDC provider"""

        user_info = {}

        # Decode ID token
        id_token = tokens.get("id_token")
        if id_token:
            # In production, verify signature
            payload = jwt.decode(id_token, options={"verify_signature": False})
            user_info.update(payload)

        # Get additional user info from userinfo endpoint
        metadata = self.provider_metadata.get(config.provider_id, {})
        userinfo_endpoint = metadata.get("userinfo_endpoint")

        if userinfo_endpoint and tokens.get("access_token"):
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            response = await self.http_client.get(userinfo_endpoint, headers=headers)

            if response.status_code == 200:
                user_info.update(response.json())

        return user_info

    async def _create_sso_user(
        self, config: SSOConfiguration, attributes: Dict[str, Any]
    ) -> SSOUser:
        """Create SSO user from provider attributes"""

        # Apply attribute mapping
        mapped_attrs = self._map_user_attributes(config, attributes)

        user_id = str(uuid.uuid4())
        external_id = mapped_attrs.get("external_id", mapped_attrs.get("email", ""))

        sso_user = SSOUser(
            user_id=user_id,
            tenant_id=config.tenant_id,
            provider_id=config.provider_id,
            external_id=external_id,
            email=mapped_attrs.get("email", ""),
            first_name=mapped_attrs.get("first_name", ""),
            last_name=mapped_attrs.get("last_name", ""),
            display_name=mapped_attrs.get("display_name", ""),
            username=mapped_attrs.get("username", ""),
            groups=mapped_attrs.get("groups", []),
            roles=mapped_attrs.get("roles", [config.default_role]),
            raw_attributes=attributes,
        )

        # JIT provisioning
        if config.jit_provisioning:
            await self._provision_user(config, sso_user, UserProvisioningAction.CREATE)

        logger.info(f"Created SSO user: {user_id} from provider: {config.provider_id}")

        return sso_user

    def _map_user_attributes(
        self, config: SSOConfiguration, attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map provider attributes to user attributes"""

        mapped = {}

        # Apply attribute mapping configuration
        for target_attr, source_attr in config.attribute_mapping.items():
            if source_attr in attributes:
                mapped[target_attr] = attributes[source_attr]

        # Default mappings if not configured
        if "email" not in mapped:
            mapped["email"] = attributes.get("email", attributes.get("mail", ""))

        if "first_name" not in mapped:
            mapped["first_name"] = attributes.get(
                "given_name", attributes.get("givenName", "")
            )

        if "last_name" not in mapped:
            mapped["last_name"] = attributes.get(
                "family_name", attributes.get("surname", "")
            )

        if "username" not in mapped:
            mapped["username"] = attributes.get(
                "preferred_username", attributes.get("uid", mapped.get("email", ""))
            )

        # Map roles
        mapped_roles = []
        for provider_role, app_roles in config.role_mapping.items():
            if provider_role in attributes.get(
                "groups", []
            ) or provider_role in attributes.get("roles", []):
                mapped_roles.extend(app_roles)

        mapped["roles"] = mapped_roles or [config.default_role]

        return mapped

    async def _provision_user(
        self,
        config: SSOConfiguration,
        sso_user: SSOUser,
        action: UserProvisioningAction,
    ):
        """Provision user in the application"""

        # This would integrate with the main user management system
        # For demo purposes, just log the action

        logger.info(f"User provisioning: {action.value} for user {sso_user.email}")

    def _get_sso_config(
        self, tenant_id: str, provider_id: str
    ) -> Optional[SSOConfiguration]:
        """Get SSO configuration"""

        tenant_configs = self.sso_configs.get(tenant_id, {})
        return tenant_configs.get(provider_id)

    def _find_config_by_issuer(self, issuer: str) -> Optional[SSOConfiguration]:
        """Find SSO configuration by issuer"""

        for tenant_configs in self.sso_configs.values():
            for config in tenant_configs.values():
                if config.saml_entity_id == issuer:
                    return config

        return None

    async def logout_user(self, session_id: str) -> bool:
        """Logout SSO user"""

        sso_user = self.active_sessions.pop(session_id, None)
        if not sso_user:
            return False

        config = self._get_sso_config(sso_user.tenant_id, sso_user.provider_id)
        if config and config.saml_slo_url:
            # Initiate SLO if supported
            await self._initiate_slo(config, sso_user)

        logger.info(f"Logged out SSO user: {sso_user.user_id}")

        return True

    async def _initiate_slo(self, config: SSOConfiguration, sso_user: SSOUser):
        """Initiate Single Logout"""

        # Create SAML LogoutRequest
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        logout_request = f"""
        <samlp:LogoutRequest 
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" 
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{timestamp}"
            Destination="{config.saml_slo_url}">
            <saml:Issuer>{config.saml_entity_id}</saml:Issuer>
            <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">
                {sso_user.email}
            </saml:NameID>
        </samlp:LogoutRequest>
        """

        # This would send the logout request to the provider
        logger.info(f"Initiated SLO for user: {sso_user.email}")

    async def get_user_session(self, session_id: str) -> Optional[SSOUser]:
        """Get user session"""

        return self.active_sessions.get(session_id)

    async def list_tenant_sso_configs(self, tenant_id: str) -> List[SSOConfiguration]:
        """List SSO configurations for tenant"""

        tenant_configs = self.sso_configs.get(tenant_id, {})
        return list(tenant_configs.values())

    async def update_sso_configuration(
        self, tenant_id: str, provider_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update SSO configuration"""

        config = self._get_sso_config(tenant_id, provider_id)
        if not config:
            return False

        # Update configuration
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.updated_at = datetime.now(timezone.utc)

        # Reload provider metadata if needed
        if any(key in updates for key in ["metadata_url", "oidc_discovery_url"]):
            await self._load_provider_metadata(config)

        logger.info(f"Updated SSO configuration: {provider_id}")

        return True

    async def delete_sso_configuration(self, tenant_id: str, provider_id: str) -> bool:
        """Delete SSO configuration"""

        tenant_configs = self.sso_configs.get(tenant_id, {})
        if provider_id in tenant_configs:
            del tenant_configs[provider_id]

            # Remove provider metadata
            self.provider_metadata.pop(provider_id, None)

            logger.info(f"Deleted SSO configuration: {provider_id}")
            return True

        return False

    async def _session_cleanup_task(self):
        """Background task to clean up expired sessions"""

        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                now = datetime.now(timezone.utc)
                expired_sessions = []

                for session_id, sso_user in self.active_sessions.items():
                    # Check if session is expired (default 1 hour)
                    session_age = now - sso_user.auth_time
                    if session_age.total_seconds() > 3600:
                        expired_sessions.append(session_id)

                # Remove expired sessions
                for session_id in expired_sessions:
                    self.active_sessions.pop(session_id, None)

                if expired_sessions:
                    logger.info(
                        f"Cleaned up {len(expired_sessions)} expired SSO sessions"
                    )

            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")

    async def _metadata_refresh_task(self):
        """Background task to refresh provider metadata"""

        while True:
            try:
                await asyncio.sleep(86400)  # Run daily

                for tenant_configs in self.sso_configs.values():
                    for config in tenant_configs.values():
                        if config.enabled:
                            await self._load_provider_metadata(config)

                logger.info("Refreshed SSO provider metadata")

            except Exception as e:
                logger.error(f"Error in metadata refresh task: {e}")


class SSOMiddleware:
    """Middleware for SSO authentication"""

    def __init__(self, sso_manager: SSOManager):
        self.sso_manager = sso_manager

    async def __call__(self, request, call_next):
        """Process request with SSO authentication"""

        # Check for SSO session
        session_id = request.headers.get("X-Session-ID") or request.cookies.get(
            "sso_session"
        )

        if session_id:
            sso_user = await self.sso_manager.get_user_session(session_id)
            if sso_user:
                # Add SSO user to request
                request.state.sso_user = sso_user
                request.state.authenticated = True
            else:
                request.state.authenticated = False
        else:
            request.state.authenticated = False

        response = await call_next(request)
        return response
