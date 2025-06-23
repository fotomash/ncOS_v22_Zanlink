"""
VectorMemoryBoot Agent Implementation
Handles vector database initialization and persistence with single-session priority
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np


@dataclass
class VectorEntry:
    """Represents a single vector entry in memory"""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'vector': self.vector,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class VectorMemoryBoot:
    """
    Primary initialization component for vector memory subsystem.
    Implements lightweight, single-session vector storage.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_store: Dict[str, VectorEntry] = {}
        self.dimension = config.get('vector_dimension', 768)
        self.max_entries = config.get('max_entries', 10000)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the vector memory system"""
        try:
            # Clear existing memory for single-session mode
            self.memory_store.clear()

            # Set up indexing structures
            self.index_by_metadata = {}
            self.index_by_timestamp = []

            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize VectorMemoryBoot: {e}")
            return False

    def generate_id(self, content: str) -> str:
        """Generate unique ID for vector entry"""
        return hashlib.sha256(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> str:
        """Add a vector to memory"""
        if not self.initialized:
            raise RuntimeError("VectorMemoryBoot not initialized")

        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")

        if len(self.memory_store) >= self.max_entries:
            # Remove oldest entry in FIFO manner
            oldest_id = self.index_by_timestamp[0]
            self.remove_vector(oldest_id)

        entry_id = self.generate_id(str(metadata))
        entry = VectorEntry(
            id=entry_id,
            vector=vector,
            metadata=metadata,
            timestamp=datetime.now()
        )

        self.memory_store[entry_id] = entry
        self.index_by_timestamp.append(entry_id)

        # Update metadata index
        for key, value in metadata.items():
            if key not in self.index_by_metadata:
                self.index_by_metadata[key] = []
            self.index_by_metadata[key].append(entry_id)

        return entry_id

    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from memory"""
        if vector_id not in self.memory_store:
            return False

        entry = self.memory_store[vector_id]

        # Remove from indexes
        self.index_by_timestamp.remove(vector_id)
        for key, value in entry.metadata.items():
            if key in self.index_by_metadata:
                self.index_by_metadata[key].remove(vector_id)

        del self.memory_store[vector_id]
        return True

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors"""
        if not self.initialized:
            raise RuntimeError("VectorMemoryBoot not initialized")

        similarities = []

        for entry_id, entry in self.memory_store.items():
            similarity = self.cosine_similarity(query_vector, entry.vector)
            if similarity >= self.similarity_threshold:
                similarities.append((entry_id, similarity, entry.metadata))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def search_by_metadata(self, metadata_filters: Dict[str, Any]) -> List[VectorEntry]:
        """Search vectors by metadata"""
        matching_ids = set()

        for key, value in metadata_filters.items():
            if key in self.index_by_metadata:
                entry_ids = self.index_by_metadata[key]
                for entry_id in entry_ids:
                    if entry_id in self.memory_store:
                        entry = self.memory_store[entry_id]
                        if entry.metadata.get(key) == value:
                            matching_ids.add(entry_id)

        return [self.memory_store[eid] for eid in matching_ids]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            'total_entries': len(self.memory_store),
            'max_entries': self.max_entries,
            'dimension': self.dimension,
            'initialized': self.initialized,
            'memory_usage_percentage': (len(self.memory_store) / self.max_entries) * 100
        }

    def clear_memory(self) -> bool:
        """Clear all vector memory (for single-session mode)"""
        self.memory_store.clear()
        self.index_by_metadata.clear()
        self.index_by_timestamp.clear()
        return True

    def export_session(self) -> Dict[str, Any]:
        """Export current session data"""
        return {
            'vectors': {k: v.to_dict() for k, v in self.memory_store.items()},
            'metadata_index': self.index_by_metadata,
            'timestamp_index': self.index_by_timestamp,
            'stats': self.get_memory_stats()
        }
