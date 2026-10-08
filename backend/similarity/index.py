from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging
import uuid

logger = logging.getLogger(__name__)

import os

class SimilarityIndex:
    def __init__(self, host="localhost", port=6333, collection_name=None):
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "seismic_events_v2")
        try:
            self.client = QdrantClient(host=host, port=port)
            self._ensure_collection()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE),
            )

    def add(self, event_id: uuid.UUID, embedding: list, metadata: dict = None):
        """
        Add a single event to the index.
        event_id should be the UUID from the Postgres events table.
        """
        if not self.client:
            logger.error("Qdrant client not initialized.")
            return False
            
        point = PointStruct(
            id=str(event_id),
            vector=embedding,
            payload=metadata or {}
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        return True
        
    def search(self, query_embedding: list, limit: int = 5, source_type: str = "catalog"):
        """
        Search for similar events.
        """
        if not self.client:
            logger.error("Qdrant client not initialized.")
            return []

        query_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="source_type",
                    match=rest.MatchValue(value=source_type)
                )
            ]
        ) if source_type else None
            
        if hasattr(self.client, 'query_points'):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                query_filter=query_filter
            )
            return response.points
        elif hasattr(self.client, 'search'):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter
            )
        else:
            logger.error("No search or query_points method found on Qdrant client.")
            return []
