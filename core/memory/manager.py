#!/usr/bin/env python3
"""
NCOS v24 - Unified Memory Manager
This module provides a centralized memory system for all agents, combining
a high-speed cache with a persistent vector database for long-term storage
and context-aware retrieval.
"""
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
from ncos.core.base import BaseComponent, DataModel

class VectorStoreInterface(ABC):
    """
    Abstract base class for vector store implementations. This ensures that any
    vector database (Chroma, FAISS, Pinecone, etc.) can be plugged into the
    memory manager by creating a compatible adapter.
    """
    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def add_vectors(self, collection: str, ids: List[str], vectors: List[np.ndarray], metadatas: List[Dict]):
        pass

    @abstractmethod
    async def search(self, collection: str, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        pass

    @abstractmethod
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        pass

class MockVectorStore(VectorStoreInterface):
    """
    A mock, in-memory vector store for testing and development.
    It simulates the behavior of a real vector database.
    """
    def __init__(self):
        self._collections: Dict[str, Dict[str, Any]] = {}
        logger.warning("Initializing with MockVectorStore. For production, use a persistent vector DB.")

    async def initialize(self):
        logger.info("MockVectorStore initialized.")

    async def add_vectors(self, collection: str, ids: List[str], vectors: List[np.ndarray], metadatas: List[Dict]):
        if collection not in self._collections:
            self._collections[collection] = {"ids": [], "vectors": [], "metadatas": []}
        
        for i, vector_id in enumerate(ids):
            if vector_id in self._collections[collection]["ids"]:
                # Update existing vector
                idx = self._collections[collection]["ids"].index(vector_id)
                self._collections[collection]["vectors"][idx] = vectors[i]
                self._collections[collection]["metadatas"][idx] = metadatas[i]
            else:
                self._collections[collection]["ids"].append(vector_id)
                self._collections[collection]["vectors"].append(vectors[i])
                self._collections[collection]["metadatas"].append(metadatas[i])
        logger.debug(f"Added/updated {len(ids)} vectors in collection '{collection}'.")

    async def search(self, collection: str, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        if collection not in self._collections or not self._collections[collection]["vectors"]:
            return []

        # Simple cosine similarity search
        collection_vectors = np.array(self._collections[collection]["vectors"])
        dot_product = np.dot(collection_vectors, query_vector)
        norms = np.linalg.norm(collection_vectors, axis=1) * np.linalg.norm(query_vector)
        
        # Avoid division by zero
        norms[norms == 0] = 1e-9
        
        similarities = dot_product / norms
        
        # Get top_k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            sim = similarities[i]
            # Convert similarity to distance (0=identical, 1=different)
            distance = 1 - sim
            results.append((self._collections[collection]["ids"][i], distance, self._collections[collection]["metadatas"][i]))
        
        return results

    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        if collection not in self._collections:
            return {"vector_count": 0, "dimension": "N/A"}
        
        vectors = self._collections[collection]["vectors"]
        return {
            "vector_count": len(vectors),
            "dimension": vectors[0].shape[0] if vectors else "N/A"
        }

class MemoryManager(BaseComponent):
    """
    Manages all forms of system memory, including agent state, shared context,
    and the long-term vector store.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vector_store: VectorStoreInterface = None
        self._cache: Dict[str, Any] = {} # High-speed key-value cache
        self._state_store: Dict[str, Any] = {} # For persistent agent/system state
        self.vector_store_provider = config.get("vector_store_provider", "mock")

    async def initialize(self) -> bool:
        """Initializes the vector store and loads any persistent state."""
        self.logger.info(f"Initializing memory with provider: {self.vector_store_provider}...")
        
        if self.vector_store_provider == "mock":
            self.vector_store = MockVectorStore()
        # elif self.vector_store_provider == "chroma":
        #     # self.vector_store = ChromaDBAdapter(...)
        #     pass
        else:
            raise ValueError(f"Unsupported vector store provider: {self.vector_store_provider}")

        await self.vector_store.initialize()
        
        # In a real system, you would load self._state_store from a persistent
        # database like Redis or PostgreSQL here.
        # e.g., self._state_store = await load_from_redis()
        
        self.is_initialized = True
        self.logger.success("Memory Manager is operational.")
        return True

    async def process(self, data: Dict[str, Any]) -> Any:
        """
        Main processing method for memory operations.
        Routes memory requests to the appropriate handler.
        
        Example `data` payload:
        {
            "operation": "vector_search",
            "collection": "market_events",
            "query_vector": [...],
            "top_k": 5
        }
        """
        operation = data.get("operation")
        if not operation:
            raise ValueError("Memory operation not specified.")

        handlers = {
            "vector_search": self.search_vectors,
            "vector_add": self.add_to_vector_store,
            "set_state": self.set_state,
            "get_state": self.get_state,
        }

        handler = handlers.get(operation)
        if handler:
            return await handler(data)
        else:
            raise NotImplementedError(f"Memory operation '{operation}' is not implemented.")

    async def add_to_vector_store(self, payload: Dict[str, Any]):
        """Adds a batch of vectors to a specified collection."""
        collection = payload.get("collection")
        vectors_data = payload.get("vectors", []) # Expects list of dicts
        
        if not all([collection, vectors_data]):
            raise ValueError("Missing 'collection' or 'vectors' in payload.")
            
        ids = [item["id"] for item in vectors_data]
        vectors = [np.array(item["vector"]) for item in vectors_data]
        metadatas = [item.get("metadata", {}) for item in vectors_data]
        
        await self.vector_store.add_vectors(collection, ids, vectors, metadatas)
        self.logger.info(f"Added {len(vectors)} vectors to collection '{collection}'.")
        return {"status": "success", "added": len(vectors)}

    async def search_vectors(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Performs a similarity search in the vector store."""
        collection = payload.get("collection")
        query_vector = np.array(payload.get("query_vector", []))
        top_k = payload.get("top_k", 10)

        if not all([collection, query_vector.any()]):
            raise ValueError("Missing 'collection' or 'query_vector' in payload.")

        results = await self.vector_store.search(collection, query_vector, top_k)
        
        return [
            {"id": _id, "distance": dist, "metadata": meta}
            for _id, dist, meta in results
        ]

    async def set_state(self, payload: Dict[str, Any]):
        """Saves a persistent state value (e.g., for an agent)."""
        key = payload.get("key")
        value = payload.get("value")
        if not key:
            raise ValueError("Missing 'key' for set_state operation.")
        
        self._state_store[key] = value
        # In a real system, this would also write to a persistent DB.
        # await self.persistent_db.set(key, value)
        return {"status": "success", "key": key}

    async def get_state(self, payload: Dict[str, Any]) -> Any:
        """Retrieves a persistent state value."""
        key = payload.get("key")
        if not key:
            raise ValueError("Missing 'key' for get_state operation.")
        
        return self._state_store.get(key)

    async def cleanup(self):
        """Saves final state and closes connections."""
        # In a real system, you would save the entire self._state_store
        # to a persistent database before shutting down.
        self.logger.info("Memory Manager cleanup complete.")
        await super().cleanup()
