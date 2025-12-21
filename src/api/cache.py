"""
Redis cache layer for sentiment predictions
Reduces model inference time for repeated queries
"""

import redis
import json
import hashlib
from typing import Optional, Dict, Any
from loguru import logger
import os


class RedisCache:
    """Redis-based caching for predictions"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,  # 1 hour default TTL
        enabled: bool = True
    ):
        self.enabled = enabled
        self.ttl = ttl
        self.client = None
        
        if not self.enabled:
            logger.info("Cache disabled")
            return
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"✓ Connected to Redis at {host}:{port}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis unavailable: {e}. Running without cache.")
            self.enabled = False
            self.client = None
    
    def _generate_key(self, text: str, model_type: str) -> str:
        """Generate cache key from text and model type"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"prediction:{model_type}:{text_hash}"
    
    def get(self, text: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Get prediction from cache"""
        if not self.enabled or not self.client:
            return None
        
        try:
            key = self._generate_key(text, model_type)
            cached = self.client.get(key)
            
            if cached:
                logger.debug(f"Cache HIT for key: {key}")
                return json.loads(cached)
            
            logger.debug(f"Cache MISS for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        text: str,
        model_type: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Store prediction in cache"""
        if not self.enabled or not self.client:
            return
        
        try:
            key = self._generate_key(text, model_type)
            value = json.dumps(prediction)
            ttl = ttl or self.ttl
            
            self.client.setex(key, ttl, value)
            logger.debug(f"Cached prediction for key: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled or not self.client:
            return {"enabled": False}
        
        try:
            info = self.client.info("stats")
            return {
                "enabled": True,
                "total_connections": info.get("total_connections_received", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)
    
    def clear(self):
        """Clear all cached predictions"""
        if not self.enabled or not self.client:
            return
        
        try:
            # Delete all prediction keys
            keys = self.client.keys("prediction:*")
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached predictions")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


# Global cache instance
_cache_instance = None


def get_cache() -> RedisCache:
    """Get or create global cache instance"""
    global _cache_instance
    
    if _cache_instance is None:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        cache_ttl = int(os.getenv("CACHE_TTL", 3600))
        
        _cache_instance = RedisCache(
            host=redis_host,
            port=redis_port,
            ttl=cache_ttl,
            enabled=cache_enabled
        )
    
    return _cache_instance
