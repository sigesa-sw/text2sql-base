"""
This module provides caching functionality for the Flask application.

The module defines two classes:
- Cache: An abstract base class that defines the interface
 for caching implementations
- MemoryCache: A concrete implementation that provides simple in-memory caching

Example usage:
    cache = MemoryCache()
    cache_id = cache.generate_id()
    cache.set(cache_id, "field1", "value1")
    value = cache.get(cache_id, "field1")
"""

from abc import ABC, abstractmethod
import uuid


class Cache(ABC):
    """
    Cache is an abstract base class for caching data.
    """

    @abstractmethod
    def generate_id(self):
        """
        Generate a unique identifier for the cache.
        """

    @abstractmethod
    def get(self, cache_id, field):
        """
        Get a value from the cache.
        """

    @abstractmethod
    def get_all(self, field_list) -> list:
        """
        Get all values from the cache.
        """

    @abstractmethod
    def set(self, cache_id, field, value):
        """
        Set a value in the cache.
        """

    @abstractmethod
    def delete(self, cache_id):
        """
        Delete a value from the cache.
        """


class MemoryCache(Cache):
    """
    MemoryCache is a simple in-memory cache.
    """

    def __init__(self):
        self.cache = {}

    def generate_id(self):
        return str(uuid.uuid4())

    def set(self, cache_id, field, value):
        if cache_id not in self.cache:
            self.cache[cache_id] = {}

        self.cache[cache_id][field] = value

    def get(self, cache_id, field):
        if cache_id not in self.cache:
            return None
        if field not in self.cache[cache_id]:
            return None
        return self.cache[cache_id][field]

    def get_all(self, field_list) -> list:
        return [
            {
                "id": cache_id,
                **{
                    field: self.get(cache_id=cache_id, field=field)
                    for field in field_list
                },
            }
            for cache_id in self.cache
        ]

    def delete(self, cache_id):
        if cache_id in self.cache:
            del self.cache[cache_id]
