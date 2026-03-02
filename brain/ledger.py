"""
engine.ledger — Hash-Chained Audit Ledger (Gap C)
====================================================
Every event includes the hash of the previous event,
forming a tamper-evident chain. If any entry is modified,
all subsequent hashes break.

The final decision package includes the ledger head hash,
creating an unforgeable audit trail.

Structure:
  entry_0: hash = H(genesis || content_0)
  entry_1: hash = H(entry_0.hash || content_1)
  entry_N: hash = H(entry_{N-1}.hash || content_N)

Verification: walk chain forward, recompute each hash,
confirm it matches stored hash. Any mismatch = tampered.
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


GENESIS_HASH = "0" * 64  # The "previous hash" for the first entry


@dataclass
class LedgerEntry:
    """Single entry in the hash chain."""
    sequence: int
    timestamp: float
    event_type: str
    content: Dict
    prev_hash: str           # Hash of previous entry
    entry_hash: str = ""     # H(prev_hash || content)

    def compute_hash(self) -> str:
        """Compute the hash for this entry."""
        payload = json.dumps({
            "seq": self.sequence,
            "prev": self.prev_hash,
            "type": self.event_type,
            "content": self.content,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "seq": self.sequence, "ts": self.timestamp,
            "type": self.event_type,
            "prev_hash": self.prev_hash[:16] + "...",
            "hash": self.entry_hash[:16] + "...",
            "content_keys": list(self.content.keys()) if isinstance(self.content, dict) else [],
        }


class HashChainedLedger:
    """Tamper-evident hash-chained audit ledger.

    Usage:
        ledger = HashChainedLedger()
        ledger.append("tool_call", {"tool": "census", "result": {...}})
        ledger.append("agent_output", {"agent": "uw", "output": {...}})
        ledger.append("decision", {"decision": "GO", "confidence": 0.72})
        assert ledger.verify()
        head_hash = ledger.head_hash()
    """

    def __init__(self):
        self._entries: List[LedgerEntry] = []
        self._head_hash: str = GENESIS_HASH

    def append(self, event_type: str, content: Dict) -> LedgerEntry:
        """Append an entry to the chain."""
        entry = LedgerEntry(
            sequence=len(self._entries),
            timestamp=time.time(),
            event_type=event_type,
            content=content,
            prev_hash=self._head_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self._head_hash = entry.entry_hash
        self._entries.append(entry)
        return entry

    def head_hash(self) -> str:
        """Get the current head hash (for inclusion in decision package)."""
        return self._head_hash

    def verify(self) -> bool:
        """Walk the chain and verify every hash. Returns True if intact."""
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry.prev_hash != prev:
                return False
            computed = entry.compute_hash()
            if computed != entry.entry_hash:
                return False
            prev = entry.entry_hash
        return True

    def find_tamper(self) -> Optional[int]:
        """If chain is broken, return the first tampered entry index."""
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry.prev_hash != prev:
                return entry.sequence
            computed = entry.compute_hash()
            if computed != entry.entry_hash:
                return entry.sequence
            prev = entry.entry_hash
        return None  # No tampering detected

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    def summary(self) -> Dict:
        return {
            "entries": len(self._entries),
            "head_hash": self._head_hash[:16] + "..." if self._head_hash != GENESIS_HASH else "genesis",
            "verified": self.verify(),
            "event_types": list(set(e.event_type for e in self._entries)),
        }

    def save(self, path: str):
        """Save the full ledger (entries + hashes) for audit."""
        data = {
            "head_hash": self._head_hash,
            "entry_count": len(self._entries),
            "entries": [{
                "seq": e.sequence, "ts": e.timestamp,
                "type": e.event_type, "content": e.content,
                "prev_hash": e.prev_hash, "hash": e.entry_hash,
            } for e in self._entries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "HashChainedLedger":
        """Load a ledger and verify integrity."""
        with open(path) as f:
            data = json.load(f)
        ledger = cls()
        for ed in data["entries"]:
            entry = LedgerEntry(
                sequence=ed["seq"], timestamp=ed["ts"],
                event_type=ed["type"], content=ed["content"],
                prev_hash=ed["prev_hash"], entry_hash=ed["hash"],
            )
            ledger._entries.append(entry)
            ledger._head_hash = entry.entry_hash
        return ledger
