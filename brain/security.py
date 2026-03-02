"""
engine.security — Security & Multi-User Controls (Gap E)
==========================================================
Minimal RBAC layer + secrets vault interface + access logs.

Roles:
  ADMIN:    full access, can modify config, manage users
  ANALYST:  run pipelines, submit data, view all outputs
  REVIEWER: read-only access to decision packages
  AUDITOR:  read-only access to replay logs + ledger
  AGENT:    system role for AI agents (limited to assigned tools)

Access log: every action is logged with user, role, timestamp, action.
Secrets vault: interface for API keys, never exposed in outputs/logs.
"""

from __future__ import annotations
import hashlib
import hmac
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class Role(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    AUDITOR = "auditor"
    AGENT = "agent"


# Permission matrix: role → allowed actions
PERMISSIONS: Dict[str, Set[str]] = {
    "admin": {"run_pipeline", "submit_data", "submit_assumption",
              "view_package", "view_replay", "view_ledger",
              "manage_users", "manage_config", "manage_secrets",
              "export_data", "delete_run"},
    "analyst": {"run_pipeline", "submit_data", "submit_assumption",
                "view_package", "view_replay", "export_data"},
    "reviewer": {"view_package", "view_replay"},
    "auditor": {"view_package", "view_replay", "view_ledger", "export_data"},
    "agent": {"call_tool", "submit_output"},  # Agents have minimal permissions
}


@dataclass
class User:
    """Authenticated user."""
    user_id: str
    name: str
    role: Role
    password_hash: str = ""        # bcrypt/sha256 hash
    api_key_hash: str = ""
    created_at: float = 0.0
    active: bool = True


@dataclass
class AccessLogEntry:
    """Single entry in the access log."""
    timestamp: float
    user_id: str
    role: str
    action: str
    resource: str = ""
    success: bool = True
    reason: str = ""
    ip_address: str = ""


class AccessDenied(Exception):
    """Raised when a user lacks permission for an action."""
    def __init__(self, user_id: str, action: str, role: str):
        self.user_id = user_id
        self.action = action
        super().__init__(f"Access denied: {user_id} ({role}) cannot {action}")


class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._access_log: List[AccessLogEntry] = []
        # Create default system user
        self._users["system"] = User(
            user_id="system", name="System",
            role=Role.ADMIN, created_at=time.time())

    def add_user(self, user_id: str, name: str, role: Role,
                 password: str = "") -> User:
        pw_hash = hashlib.sha256(password.encode()).hexdigest() if password else ""
        user = User(user_id=user_id, name=name, role=role,
                    password_hash=pw_hash, created_at=time.time())
        self._users[user_id] = user
        self._log(user_id, "user_created", f"role={role.value}")
        return user

    def authenticate(self, user_id: str, password: str) -> Optional[User]:
        user = self._users.get(user_id)
        if not user or not user.active:
            self._log(user_id, "auth_failed", reason="user not found")
            return None
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        if not hmac.compare_digest(pw_hash, user.password_hash):
            self._log(user_id, "auth_failed", reason="bad password")
            return None
        self._log(user_id, "auth_success")
        return user

    def check_permission(self, user_id: str, action: str,
                         resource: str = "") -> bool:
        """Check if user has permission. Logs the attempt."""
        user = self._users.get(user_id)
        if not user or not user.active:
            self._log(user_id, action, resource, success=False,
                      reason="user not found or inactive")
            return False
        allowed = action in PERMISSIONS.get(user.role.value, set())
        self._log(user_id, action, resource, success=allowed,
                  reason="" if allowed else "insufficient permissions")
        return allowed

    def require_permission(self, user_id: str, action: str,
                           resource: str = ""):
        """Check permission or raise AccessDenied."""
        if not self.check_permission(user_id, action, resource):
            user = self._users.get(user_id)
            role = user.role.value if user else "unknown"
            raise AccessDenied(user_id, action, role)

    def _log(self, user_id: str, action: str, resource: str = "",
             success: bool = True, reason: str = ""):
        self._access_log.append(AccessLogEntry(
            timestamp=time.time(), user_id=user_id,
            role=self._users.get(user_id, User("?","?",Role.AGENT)).role.value,
            action=action, resource=resource,
            success=success, reason=reason))

    def access_log(self, limit: int = 100) -> List[Dict]:
        entries = self._access_log[-limit:]
        return [{"ts": e.timestamp, "user": e.user_id, "role": e.role,
                 "action": e.action, "resource": e.resource,
                 "ok": e.success, "reason": e.reason}
                for e in entries]

    def users(self) -> List[Dict]:
        return [{"id": u.user_id, "name": u.name, "role": u.role.value,
                 "active": u.active} for u in self._users.values()]


class SecretsVault:
    """Interface for storing API keys and credentials.

    Keys are stored in memory with hashed names.
    NEVER exposed in outputs, logs, or decision packages.
    In production: replace with AWS Secrets Manager / Vault / etc.
    """

    def __init__(self):
        self._secrets: Dict[str, str] = {}  # name_hash → encrypted_value
        self._access_log: List[Dict] = []

    def store(self, name: str, value: str, stored_by: str = "system"):
        """Store a secret. Value is stored with a simple obfuscation
        (in production, use proper encryption at rest)."""
        name_hash = hashlib.sha256(name.encode()).hexdigest()[:16]
        # Simple XOR obfuscation (NOT real encryption — use Fernet/AES in prod)
        obfuscated = self._obfuscate(value, name)
        self._secrets[name_hash] = obfuscated
        self._access_log.append({
            "ts": time.time(), "action": "store",
            "name_hash": name_hash, "by": stored_by})

    def retrieve(self, name: str, requested_by: str = "system") -> Optional[str]:
        """Retrieve a secret."""
        name_hash = hashlib.sha256(name.encode()).hexdigest()[:16]
        obfuscated = self._secrets.get(name_hash)
        self._access_log.append({
            "ts": time.time(), "action": "retrieve",
            "name_hash": name_hash, "by": requested_by,
            "found": obfuscated is not None})
        if obfuscated is None:
            return None
        return self._deobfuscate(obfuscated, name)

    def delete(self, name: str, deleted_by: str = "system") -> bool:
        name_hash = hashlib.sha256(name.encode()).hexdigest()[:16]
        found = name_hash in self._secrets
        self._secrets.pop(name_hash, None)
        self._access_log.append({
            "ts": time.time(), "action": "delete",
            "name_hash": name_hash, "by": deleted_by, "found": found})
        return found

    def list_keys(self) -> List[str]:
        """List stored key hashes (never the actual names)."""
        return list(self._secrets.keys())

    def _obfuscate(self, value: str, key: str) -> str:
        key_bytes = hashlib.sha256(key.encode()).digest()
        return ''.join(chr(ord(c) ^ key_bytes[i % len(key_bytes)])
                       for i, c in enumerate(value))

    def _deobfuscate(self, value: str, key: str) -> str:
        return self._obfuscate(value, key)  # XOR is symmetric
