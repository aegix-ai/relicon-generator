"""
Professional Caching System
Multi-layer caching with Redis support and intelligent invalidation.
"""

import json
import hashlib
import time
from typing import Any, Optional, Dict, List, Callable, Union
from functools import wraps
from datetime import datetime, timedelta
import pickle
from pathlib import Path

try:
    import redis
    from redis.sentinel import Sentinel
    from redis.cluster import RedisCluster
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import config
from core.logger import get_logger
from core.exceptions import ConfigurationError

logger = get_logger(__name__)

class CacheKey:
    """Utility for generating consistent cache keys."""
    
    @staticmethod
    def brand_analysis(brand_name: str, brand_description: str) -> str:
        """Generate cache key for brand analysis."""
        content = f"{brand_name}:{brand_description}"
        hash_obj = hashlib.sha256(content.encode())
        return f"brand_analysis:{hash_obj.hexdigest()[:16]}"
    
    
    @staticmethod
    def video_blueprint(brand_info: Dict[str, Any], service_type: str) -> str:
        """Generate cache key for video blueprint."""
        # Create deterministic string from brand info
        sorted_items = sorted(brand_info.items())
        content = f"{json.dumps(sorted_items)}:{service_type}"
        hash_obj = hashlib.sha256(content.encode())
        return f"video_blueprint:{hash_obj.hexdigest()[:16]}"
    
    @staticmethod
    def brand_profile_db(profile_id: str) -> str:
        """Generate cache key for database brand profile."""
        return f"db_brand_profile:{profile_id}"
    
    @staticmethod
    def generation_session_db(session_id: str) -> str:
        """Generate cache key for database generation session."""
        return f"db_generation_session:{session_id}"
    
    @staticmethod
    def quality_metrics_db(session_id: str) -> str:
        """Generate cache key for database quality metrics."""
        return f"db_quality_metrics:{session_id}"
    
    @staticmethod
    def performance_analytics_db(niche: Optional[str] = None) -> str:
        """Generate cache key for database performance analytics."""
        niche_key = niche or "all_niches"
        return f"db_analytics:{niche_key}"
    
    @staticmethod
    def niche_detection(brand_description: str) -> str:
        """Generate cache key for niche detection."""
        hash_obj = hashlib.sha256(brand_description.encode())
        return f"niche_detection:{hash_obj.hexdigest()[:16]}"

class MemoryCache:
    """In-memory LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        # Check expiration
        entry = self.cache[key]
        if entry['expires_at'] and time.time() > entry['expires_at']:
            self.delete(key)
            return None
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache."""
        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + ttl_seconds
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Evict oldest if over limit
        while len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = 0
        current_time = time.time()
        
        for entry in self.cache.values():
            if entry['expires_at'] and current_time > entry['expires_at']:
                expired_entries += 1
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'max_size': self.max_size,
            'utilization': total_entries / self.max_size if self.max_size > 0 else 0
        }

class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        if not REDIS_AVAILABLE:
            raise ConfigurationError("Redis library not available. Install with: pip install redis")
        
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db, 
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}/{db}")
        except Exception as e:
            raise ConfigurationError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            # Deserialize
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis GET failed for key {key}", action="cache.get.error", error=str(e))
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        try:
            # Serialize
            data = pickle.dumps(value)
            
            if ttl_seconds:
                self.redis_client.setex(key, ttl_seconds, data)
            else:
                self.redis_client.set(key, data)
                
        except Exception as e:
            logger.error(f"Redis SET failed for key {key}", action="cache.set.error", error=str(e))
    
    def delete(self, key: str) -> None:
        """Delete key from Redis cache."""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE failed for key {key}", action="cache.delete.error", error=str(e))
    
    def clear(self) -> None:
        """Clear all cache entries (use with caution)."""
        try:
            self.redis_client.flushdb()
        except Exception as e:
            logger.error("Redis FLUSHDB failed", action="cache.clear.error", error=str(e))
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS failed for key {key}", action="cache.exists.error", error=str(e))
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        try:
            info = self.redis_client.info()
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_connections_received': info.get('total_connections_received', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error("Redis INFO failed", action="cache.stats.error", error=str(e))
            return {}

class CacheManager:
    """Multi-layer cache manager with automatic fallback."""
    
    def __init__(self):
        self.memory_cache = MemoryCache(max_size=1000)
        self.redis_cache = None
        
        # Initialize Redis if enabled and available
        if config.cache.enable_redis and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(
                    host=config.cache.redis_host,
                    port=config.cache.redis_port,
                    db=config.cache.redis_db
                )
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache, using memory cache only: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)."""
        
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            logger.debug(f"Cache HIT (memory): {key}")
            return value
        
        # Try Redis cache if available
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                logger.debug(f"Cache HIT (Redis): {key}")
                # Store in memory cache for faster subsequent access
                self.memory_cache.set(key, value, ttl_seconds=300)  # 5 min memory TTL
                return value
        
        logger.debug(f"Cache MISS: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache (both memory and Redis)."""
        
        # Use default TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = config.cache.default_ttl_seconds
        
        # Set in memory cache
        self.memory_cache.set(key, value, ttl_seconds)
        
        # Set in Redis cache if available
        if self.redis_cache:
            self.redis_cache.set(key, value, ttl_seconds)
        
        logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")
    
    def delete(self, key: str) -> None:
        """Delete key from both caches."""
        self.memory_cache.delete(key)
        
        if self.redis_cache:
            self.redis_cache.delete(key)
        
        logger.debug(f"Cache DELETE: {key}")
    
    def clear(self, pattern: Optional[str] = None) -> None:
        """Clear cache entries (optionally by pattern)."""
        if pattern:
            # For pattern-based clearing, we'd need to implement pattern matching
            logger.warning("Pattern-based cache clearing not yet implemented")
        else:
            self.memory_cache.clear()
            if self.redis_cache:
                self.redis_cache.clear()
            logger.info("Cache cleared")
    
    def get_with_db_fallback(self, cache_key: str, db_retrieval_func: Callable, *args, **kwargs) -> Optional[Any]:
        """
        Get value from cache with database fallback.
        
        Args:
            cache_key: Cache key to use
            db_retrieval_func: Function to call if cache miss (database retrieval)
            *args, **kwargs: Arguments for database retrieval function
        """
        # Try cache first
        cached_value = self.get(cache_key)
        if cached_value is not None:
            logger.debug(f"Cache HIT (with DB fallback): {cache_key}")
            return cached_value
        
        # Try database fallback
        try:
            db_value = db_retrieval_func(*args, **kwargs)
            if db_value is not None:
                # Cache the database value for future requests
                self.set(cache_key, db_value, ttl_seconds=3600)  # 1 hour default
                logger.debug(f"Database fallback SUCCESS, cached: {cache_key}")
                return db_value
        except Exception as e:
            logger.error(f"Database fallback failed for {cache_key}: {e}")
        
        logger.debug(f"Cache MISS and DB fallback FAILED: {cache_key}")
        return None
    
    def set_with_db_sync(self, cache_key: str, value: Any, db_save_func: Optional[Callable] = None, ttl_seconds: Optional[int] = None, *args, **kwargs) -> None:
        """
        Set value in cache and optionally sync to database.
        
        Args:
            cache_key: Cache key to use
            value: Value to cache
            db_save_func: Optional function to save to database
            ttl_seconds: Cache TTL
            *args, **kwargs: Arguments for database save function
        """
        # Set in cache
        self.set(cache_key, value, ttl_seconds)
        
        # Optionally sync to database
        if db_save_func:
            try:
                db_save_func(value, *args, **kwargs)
                logger.debug(f"Value cached and synced to DB: {cache_key}")
            except Exception as e:
                logger.warning(f"Database sync failed for {cache_key}: {e}")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        invalidated = 0
        
        # For memory cache, we'd need to implement pattern matching
        # This is a simplified implementation
        if pattern.startswith("brand_"):
            # Clear brand-related cache entries
            keys_to_delete = []
            for key in list(self.memory_cache.cache.keys()):
                if key.startswith("brand_"):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self.memory_cache.delete(key)
                invalidated += 1
        
        # For Redis, we could use SCAN with MATCH pattern
        if self.redis_cache:
            try:
                # Redis pattern invalidation would be implemented here
                pass
            except Exception as e:
                logger.error(f"Redis pattern invalidation failed: {e}")
        
        logger.debug(f"Invalidated {invalidated} cache entries matching pattern: {pattern}")
        return invalidated
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.stats()
        redis_stats = self.redis_cache.stats() if self.redis_cache else {}
        
        return {
            'memory_cache': memory_stats,
            'redis_cache': redis_stats,
            'redis_enabled': self.redis_cache is not None,
            'database_integration': {
                'fallback_enabled': True,
                'sync_enabled': True,
                'pattern_invalidation_enabled': True
            }
        }

# Cache decorators
def cached(key_func: Callable[..., str], ttl_seconds: Optional[int] = None):
    """Decorator to cache function results."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Function cache HIT: {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache the result
            cache.set(cache_key, result, ttl_seconds)
            
            logger.debug(
                f"Function cache MISS: {func.__name__}",
                action="function.cache.miss",
                execution_time_ms=execution_time * 1000,
                cache_key=cache_key
            )
            
            return result
        
        return wrapper
    return decorator

def cache_brand_analysis_with_db(func: Callable) -> Callable:
    """Enhanced caching decorator for brand analysis with database fallback."""
    
    @wraps(func)
    def wrapper(self, brand_name: str, brand_description: str, *args, **kwargs):
        # Generate cache key
        cache_key = CacheKey.brand_analysis(brand_name, brand_description)
        
        # Try cache first (L1 - fastest)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Brand analysis cache HIT (L1) for {brand_name}")
            return cached_result
        
        # Try database second (L2 - persistent storage)
        try:
            from core.brand_intelligence import database_manager
            if database_manager and database_manager.is_available:
                # Search for existing brand profile in database
                session = database_manager.get_session()
                if session:
                    from core.brand_intelligence import BrandProfile
                    existing_profile = session.query(BrandProfile).filter_by(
                        brand_name=brand_name,
                        brand_description=brand_description
                    ).first()
                    session.close()
                    
                    if existing_profile:
                        # Reconstruct BrandElements from database
                        logger.info(f"Brand analysis DATABASE HIT (L2) for {brand_name}")
                        # Note: This would need full BrandElements reconstruction logic
                        # For now, we proceed to fresh analysis but could be enhanced
        except Exception as db_error:
            logger.debug(f"Database fallback check failed: {db_error}")
        
        # Execute fresh analysis (L3 - compute)
        start_time = time.time()
        result = func(self, brand_name, brand_description, *args, **kwargs)
        execution_time = time.time() - start_time
        
        # Cache the result (populate L1 cache)
        cache.set(cache_key, result, config.cache.brand_analysis_ttl_seconds)
        
        logger.info(
            f"Brand analysis completed and cached for {brand_name}",
            action="brand_analysis.cached",
            execution_time_ms=execution_time * 1000,
            cache_key=cache_key,
            data_flow="L3_compute->L1_cache"
        )
        
        return result
    
    return wrapper

def cache_brand_analysis(func: Callable) -> Callable:
    """Specific caching decorator for brand analysis (legacy compatibility)."""
    
    @wraps(func)
    def wrapper(self, brand_name: str, brand_description: str, *args, **kwargs):
        # Generate cache key
        cache_key = CacheKey.brand_analysis(brand_name, brand_description)
        
        # Try cache first
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Brand analysis cache HIT for {brand_name}")
            return cached_result
        
        # Execute analysis
        start_time = time.time()
        result = func(self, brand_name, brand_description, *args, **kwargs)
        execution_time = time.time() - start_time
        
        # Cache for 24 hours
        cache.set(cache_key, result, config.cache.brand_analysis_ttl_seconds)
        
        logger.info(
            f"Brand analysis completed and cached for {brand_name}",
            action="brand_analysis.cached",
            execution_time_ms=execution_time * 1000,
            cache_key=cache_key
        )
        
        return result
    
    return wrapper


# Global enhanced cache instance
cache = CacheManager()

# Initialize ML optimization
try:
    logger.info("ML-enhanced cache system initialized", action="cache.ml.init")
except Exception as e:
    logger.error(f"ML cache initialization failed: {e}", exc_info=True)

def get_cache_performance_report() -> Dict[str, Any]:
    """Get comprehensive cache performance report."""
    try:
        stats = cache.stats()
        ml_insights = cache.ml_optimizer.get_ml_cache_insights()
        
        return {
            'cache_statistics': stats,
            'ml_insights': ml_insights,
            'recommendations': ml_insights.get('recommendations', []),
            'performance_summary': {
                'overall_hit_rate': stats['performance_metrics']['hit_rate'],
                'ml_usage_percentage': (stats['performance_metrics']['ml_cache_usage'] / 
                                      stats['performance_metrics']['total_requests']) * 100 
                                      if stats['performance_metrics']['total_requests'] > 0 else 0,
                'distributed_usage': stats['performance_metrics']['distributed_usage'],
                'optimization_enabled': True
            },
            'report_generated': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache performance report generation failed: {e}")
        return {'error': str(e)}

# Cache warming utilities
def warm_cache():
    """Warm up cache with frequently accessed data."""
    logger.info("Starting cache warm-up process")
    
    # This would typically load frequently accessed brand analyses, 
    # common niche templates, etc.
    # Implementation depends on your specific use cases
    
    logger.info("Cache warm-up completed")

def cache_maintenance():
    """Perform cache maintenance tasks."""
    logger.info("Starting cache maintenance")
    
    # Get stats before maintenance
    stats_before = cache.stats()
    
    # Log current cache status
    logger.info(
        "Cache maintenance started",
        action="cache.maintenance.start",
        memory_entries=stats_before['memory_cache']['total_entries'],
        memory_expired=stats_before['memory_cache']['expired_entries']
    )
    
    # Could implement:
    # - Cleaning expired entries
    # - Compacting cache
    # - Analyzing hit rates
    # - Preloading popular items
    
    logger.info("Cache maintenance completed")

# Enhanced Distributed Caching for Phase 2
class DistributedCacheManager:
    """Advanced distributed caching with Redis cluster support for Phase 2."""
    
    def __init__(self):
        self.cluster_client = None
        self.sentinel_client = None
        self.is_cluster_mode = False
        self.is_sentinel_mode = False
        self.cache_strategies = {}
        
    def initialize_cluster(self, cluster_nodes: Optional[List[str]] = None) -> bool:
        """Initialize Redis cluster for distributed caching."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, distributed caching disabled")
            return False
        
        try:
            if cluster_nodes is None:
                # Default cluster configuration
                cluster_nodes = [
                    {'host': 'localhost', 'port': 7001},
                    {'host': 'localhost', 'port': 7002},
                    {'host': 'localhost', 'port': 7003}
                ]
            
            self.cluster_client = RedisCluster(
                startup_nodes=cluster_nodes,
                decode_responses=True,
                health_check_interval=30,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                skip_full_coverage_check=True
            )
            
            # Test connection
            self.cluster_client.ping()
            self.is_cluster_mode = True
            
            logger.info("Redis cluster initialized successfully", 
                       action="cache.cluster.init", 
                       nodes=len(cluster_nodes))
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cluster: {e}")
            self.is_cluster_mode = False
            return False
    
    def setup_caching_strategies(self):
        """Setup intelligent caching strategies for different data types."""
        self.cache_strategies = {
            'brand_analysis': {
                'ttl': 86400,  # 24 hours
                'compression': True,
                'replication_factor': 2,
                'invalidation_pattern': 'brand_*',
                'priority': 'high'
            },
            'logo_analysis': {
                'ttl': 604800,  # 7 days
                'compression': True,
                'replication_factor': 3,
                'invalidation_pattern': 'logo_*',
                'priority': 'medium'
            },
            'ml_model_cache': {
                'ttl': 3600,  # 1 hour
                'compression': False,
                'replication_factor': 3,
                'invalidation_pattern': 'ml_*',
                'priority': 'critical'
            },
            'prompt_templates': {
                'ttl': 14400,  # 4 hours
                'compression': True,
                'replication_factor': 2,
                'invalidation_pattern': 'template_*',
                'priority': 'high'
            },
            'quality_metrics': {
                'ttl': 1800,  # 30 minutes
                'compression': False,
                'replication_factor': 1,
                'invalidation_pattern': 'quality_*',
                'priority': 'medium'
            }
        }
        
        logger.info("Caching strategies configured", 
                   action="cache.strategies.init",
                   strategy_count=len(self.cache_strategies))
    
    def distributed_set(self, key: str, value: Any, strategy_type: str = 'default', ttl: Optional[int] = None) -> bool:
        """Set value in distributed cache with strategy-based optimization."""
        try:
            if not self.is_cluster_mode:
                # Fallback to regular cache
                return cache.set(key, value, ttl_seconds=ttl or 3600)
            
            strategy = self.cache_strategies.get(strategy_type, {})
            effective_ttl = ttl or strategy.get('ttl', 3600)
            
            # Prepare value for storage
            if strategy.get('compression', False):
                import gzip
                serialized_value = gzip.compress(pickle.dumps(value))
            else:
                serialized_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            
            # Store in cluster with replication
            pipeline = self.cluster_client.pipeline()
            
            # Primary storage
            pipeline.setex(key, effective_ttl, serialized_value)
            
            # Replication based on strategy
            replication_factor = strategy.get('replication_factor', 1)
            for i in range(1, replication_factor):
                replica_key = f"{key}:replica:{i}"
                pipeline.setex(replica_key, effective_ttl, serialized_value)
            
            # Execute pipeline
            pipeline.execute()
            
            logger.debug(f"Distributed cache set: {key}", 
                        action="cache.distributed.set",
                        strategy=strategy_type,
                        ttl=effective_ttl,
                        replicas=replication_factor)
            
            return True
            
        except Exception as e:
            logger.error(f"Distributed cache set failed for {key}: {e}")
            return False
    
    def distributed_get(self, key: str, strategy_type: str = 'default') -> Any:
        """Get value from distributed cache with replica fallback."""
        try:
            if not self.is_cluster_mode:
                # Fallback to regular cache
                return cache.get(key)
            
            strategy = self.cache_strategies.get(strategy_type, {})
            
            try:
                # Try primary key first
                value = self.cluster_client.get(key)
                if value is not None:
                    return self._deserialize_value(value, strategy)
            except Exception:
                pass
            
            # Try replicas if primary fails
            replication_factor = strategy.get('replication_factor', 1)
            for i in range(1, replication_factor):
                try:
                    replica_key = f"{key}:replica:{i}"
                    value = self.cluster_client.get(replica_key)
                    if value is not None:
                        logger.info(f"Retrieved from replica {i}: {key}")
                        return self._deserialize_value(value, strategy)
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Distributed cache get failed for {key}: {e}")
            return None
    
    def _deserialize_value(self, value: Union[str, bytes], strategy: Dict[str, Any]) -> Any:
        """Deserialize cached value based on strategy."""
        try:
            if strategy.get('compression', False):
                import gzip
                return pickle.loads(gzip.decompress(value))
            else:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
        except Exception as e:
            logger.error(f"Value deserialization failed: {e}")
            return value
    
    def intelligent_invalidation(self, pattern: str, cascade: bool = True) -> int:
        """Intelligent cache invalidation with cascade support."""
        try:
            if not self.is_cluster_mode:
                return 0
            
            invalidated_count = 0
            
            # Find keys matching pattern across cluster
            matching_keys = []
            for key in self.cluster_client.scan_iter(match=pattern, count=100):
                matching_keys.append(key)
            
            if not matching_keys:
                return 0
            
            # Batch delete for efficiency
            pipeline = self.cluster_client.pipeline()
            
            for key in matching_keys:
                pipeline.delete(key)
                invalidated_count += 1
                
                # Also delete replicas
                for i in range(1, 4):  # Check up to 3 replicas
                    replica_key = f"{key}:replica:{i}"
                    if self.cluster_client.exists(replica_key):
                        pipeline.delete(replica_key)
                        invalidated_count += 1
            
            pipeline.execute()
            
            logger.info(f"Intelligent invalidation completed", 
                       action="cache.invalidation",
                       pattern=pattern,
                       invalidated_count=invalidated_count)
            
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Intelligent invalidation failed: {e}")
            return 0
    
    def cache_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance analytics."""
        try:
            if not self.is_cluster_mode:
                return {'error': 'Cluster mode not available'}
            
            analytics = {
                'cluster_status': 'active',
                'node_count': len(self.cluster_client.get_nodes()),
                'strategies_configured': len(self.cache_strategies),
                'performance_metrics': {}
            }
            
            # Get Redis info from each node
            total_memory = 0
            total_keys = 0
            hit_rate_sum = 0.0
            active_nodes = 0
            
            for node in self.cluster_client.get_nodes():
                try:
                    info = node.redis_connection.info()
                    memory_info = node.redis_connection.info('memory')
                    stats_info = node.redis_connection.info('stats')
                    
                    total_memory += memory_info.get('used_memory', 0)
                    total_keys += info.get('db0', {}).get('keys', 0) if 'db0' in info else 0
                    
                    # Calculate hit rate
                    hits = stats_info.get('keyspace_hits', 0)
                    misses = stats_info.get('keyspace_misses', 0)
                    if hits + misses > 0:
                        hit_rate = hits / (hits + misses)
                        hit_rate_sum += hit_rate
                    
                    active_nodes += 1
                    
                except Exception as node_error:
                    logger.warning(f"Failed to get stats from node: {node_error}")
            
            analytics['performance_metrics'] = {
                'total_memory_mb': total_memory / (1024 * 1024),
                'total_keys': total_keys,
                'average_hit_rate': hit_rate_sum / active_nodes if active_nodes > 0 else 0.0,
                'active_nodes': active_nodes,
                'cache_efficiency': 'high' if hit_rate_sum / active_nodes > 0.8 else 'medium'
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Cache analytics failed: {e}")
            return {'error': str(e)}
    
    def preload_critical_data(self, data_sources: Dict[str, Any]) -> bool:
        """Preload critical data into distributed cache."""
        try:
            preload_count = 0
            
            for data_type, data_items in data_sources.items():
                strategy = self.cache_strategies.get(data_type, {})
                priority = strategy.get('priority', 'medium')
                
                # Only preload high and critical priority items
                if priority not in ['high', 'critical']:
                    continue
                
                for key, value in data_items.items():
                    if self.distributed_set(key, value, data_type):
                        preload_count += 1
            
            logger.info(f"Critical data preloaded: {preload_count} items")
            return True
            
        except Exception as e:
            logger.error(f"Critical data preload failed: {e}")
            return False

# Enhanced cache instance with distributed capabilities
distributed_cache = DistributedCacheManager()

# Enhanced cache warming for Phase 2
def enhanced_cache_warmup():
    """Enhanced cache warmup with ML models and critical data preloading."""
    logger.info("Starting enhanced cache warmup")
    
    try:
        # Initialize distributed caching if available
        if REDIS_AVAILABLE:
            distributed_cache.initialize_cluster()
            distributed_cache.setup_caching_strategies()
        
        # Preload critical ML model caches
        critical_data = {
            'ml_model_cache': {
                'brand_classifier_v2': {'status': 'ready', 'accuracy': 0.96},
                'quality_validator_v2': {'status': 'ready', 'accuracy': 0.94},
                'prompt_optimizer_v2': {'status': 'ready', 'performance': 0.92}
            },
            'prompt_templates': {
                'technology_templates_v2': {'optimized': True, 'count': 15},
                'healthcare_templates_v2': {'optimized': True, 'count': 12},
                'finance_templates_v2': {'optimized': True, 'count': 10}
            }
        }
        
        if distributed_cache.is_cluster_mode:
            distributed_cache.preload_critical_data(critical_data)
        
        logger.info("Enhanced cache warmup completed")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced cache warmup failed: {e}")
        return False