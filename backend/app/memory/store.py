"""External Memory Store: independent subsystem for managing, retrieving and injecting contextual memories."""

import os
import re
import json
import time
import math
from typing import List, Dict, Any, Optional
from app.core.types import MemoryItem, MemoryType, MemoryProvenance


# Retrieval used to score raw token overlap, so a word like "our" matched every
# memory (all seeds start with "Our ...") and irrelevant facts were injected into
# the prompt. Stopwords are dropped and domain terms are expanded to a canonical
# token, so "3-person startup" matches a memory about "teams under 5 engineers".
STOPWORDS = {
    "the", "and", "for", "you", "your", "yours", "our", "ours", "are", "was", "were",
    "should", "would", "could", "can", "will", "shall", "may", "might", "must",
    "this", "that", "these", "those", "there", "their", "them", "they", "with",
    "what", "when", "where", "which", "who", "whom", "why", "how", "about",
    "have", "has", "had", "been", "being", "does", "did", "done", "doing",
    "from", "into", "than", "then", "but", "not", "also", "just", "like", "some",
    "such", "only", "very", "much", "many", "more", "most", "any", "all", "each",
    "use", "using", "used", "get", "got", "make", "made", "need", "want", "please",
    "tell", "give", "help", "now", "new", "one", "two", "its", "it", "we", "us",
}

DOMAIN_SYNONYMS = {
    "team": {"startup", "startups", "team", "teams", "person", "people", "engineer",
             "engineers", "headcount", "hire", "hiring", "founder", "founders", "small"},
    "orchestration": {"kubernetes", "k8s", "orchestration", "container", "containers",
                      "docker", "microservice", "microservices", "cluster", "clusters"},
    "database": {"database", "databases", "db", "postgres", "postgresql", "rds", "sql"},
    "backend": {"backend", "python", "fastapi", "pydantic", "api", "apis"},
    "architecture": {"architecture", "architectural", "monolith", "monoliths", "design",
                     "infrastructure", "deployment", "deploy"},
}

_WORD_TO_CANON: Dict[str, str] = {}
for _canon, _terms in DOMAIN_SYNONYMS.items():
    for _term in _terms:
        _WORD_TO_CANON[_term] = _canon


def _canonicalize(words: set) -> set:
    """Map recognised domain terms to a canonical token; drop stopwords."""
    out = set()
    for w in words:
        if w in STOPWORDS:
            continue
        canon = _WORD_TO_CANON.get(w)
        out.add(canon if canon else w)
    return out


INITIAL_SEED_MEMORIES = [
    {
        "content": "Our primary production database is PostgreSQL 16 on AWS RDS with strict junction tables for many-to-many relationships.",
        "type": MemoryType.FACT,
        "provenance": MemoryProvenance.USER,
        "tags": ["database", "postgres", "infrastructure", "schema"]
    },
    {
        "content": "User prefers architectural feedback to prioritize modular monoliths over premature microservices for teams under 5 engineers.",
        "type": MemoryType.PREFERENCE,
        "provenance": MemoryProvenance.USER,
        "tags": ["architecture", "microservices", "team-size", "monolith"]
    },
    {
        "content": "Our backend codebase uses Python 3.12 and FastAPI with strictly typed Pydantic models.",
        "type": MemoryType.FACT,
        "provenance": MemoryProvenance.USER,
        "tags": ["backend", "python", "fastapi", "pydantic"]
    }
]


class ExternalMemoryStore:
    """Manages user-approved facts, preferences, and episodic summaries outside transformer weights."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.memory_dir = os.path.join(artifacts_dir, "memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.store_file = os.path.join(self.memory_dir, "store.json")
        self.master_enabled: bool = True
        self.memories: Dict[str, MemoryItem] = {}
        self._load_store()

    def _load_store(self):
        """Load persisted memories or seed with defaults on first run."""
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.master_enabled = data.get("master_enabled", True)
                    items = data.get("items", [])
                    for it in items:
                        item = MemoryItem(**it)
                        self.memories[item.memory_id] = item
                return
            except Exception as e:
                print(f"[ExternalMemoryStore] Error loading memory store: {e}")

        # Seed initial memories
        for seed in INITIAL_SEED_MEMORIES:
            mem_id = f"mem-{int(time.time())}-{len(self.memories)+1}"
            item = MemoryItem(
                memory_id=mem_id,
                content=seed["content"],
                type=seed["type"],
                provenance=seed["provenance"],
                tags=seed["tags"],
                enabled=True
            )
            self.memories[mem_id] = item
        self._save_store()

    def _save_store(self):
        """Persist memory store to JSON."""
        data = {
            "master_enabled": self.master_enabled,
            "updated_at": time.time(),
            "items": [m.model_dump() for m in self.memories.values()]
        }
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def set_enabled(self, enabled: bool):
        self.master_enabled = enabled
        self._save_store()

    def add_memory(
        self,
        content: str,
        type: MemoryType = MemoryType.FACT,
        provenance: MemoryProvenance = MemoryProvenance.USER,
        tags: Optional[List[str]] = None,
        owner_id: str = "default-user"
    ) -> MemoryItem:
        mem_id = f"mem-{int(time.time() * 1000) % 100000000}"
        item = MemoryItem(
            memory_id=mem_id,
            owner_id=owner_id,
            content=content.strip(),
            type=type,
            provenance=provenance,
            tags=tags or [],
            enabled=True,
            created_at=time.time(),
            updated_at=time.time()
        )
        self.memories[mem_id] = item
        self._save_store()
        return item

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        enabled: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[MemoryItem]:
        item = self.memories.get(memory_id)
        if not item:
            return None
        if content is not None:
            item.content = content.strip()
        if enabled is not None:
            item.enabled = enabled
        if tags is not None:
            item.tags = tags
        item.updated_at = time.time()
        self._save_store()
        return item

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            del self.memories[memory_id]
            self._save_store()
            return True
        return False

    def list_memories(self, type_filter: Optional[str] = None) -> List[MemoryItem]:
        items = list(self.memories.values())
        if type_filter:
            items = [m for m in items if m.type.value == type_filter]
        return sorted(items, key=lambda m: m.updated_at, reverse=True)

    def retrieve_relevant(self, query: str, limit: int = 3, threshold: float = 0.08) -> List[MemoryItem]:
        """
        Token-overlap & keyword relevance retrieval.
        Finds only contextually relevant memories for the incoming user query.
        """
        if not self.master_enabled:
            return []

        # Tokenize query into words (lower, stripped), drop stopwords, expand domain terms
        query_words = _canonicalize(set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower())))
        if not query_words:
            return []

        scored_items = []
        for item in self.memories.values():
            if not item.enabled:
                continue

            content_words = _canonicalize(
                set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", item.content.lower()))
            )
            tag_words = _canonicalize(set(t.lower() for t in item.tags))
            all_mem_words = content_words.union(tag_words)

            # Jaccard + Tag boost relevance
            overlap = query_words.intersection(all_mem_words)
            if not overlap:
                continue

            tag_overlap = query_words.intersection(tag_words)
            score = (len(overlap) / (len(query_words) + 1)) + (len(tag_overlap) * 0.25)

            if score >= threshold:
                item_copy = item.model_copy()
                item_copy.score = round(score, 3)
                scored_items.append(item_copy)

        scored_items.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_items[:limit]

    def format_context_block(self, memories: List[MemoryItem]) -> str:
        """Format retrieved memories into a structured context block for the model."""
        if not memories:
            return ""

        lines = ["[EXTERNAL USER CONTEXT]"]
        for m in memories:
            tag_str = f" (#{', #'.join(m.tags)})" if m.tags else ""
            lines.append(f"• [{m.type.value.upper()}] ({m.provenance.value}) {m.content}{tag_str}")
        lines.append("[END CONTEXT]\n")
        return "\n".join(lines)
