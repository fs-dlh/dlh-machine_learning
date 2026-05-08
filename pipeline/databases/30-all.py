#!/usr/bin/env python3
"""List all documents in a MongoDB collection."""

def list_all(mongo_collection):
    """
    List all documents in a MongoDB collection.

    Args:
        mongo_collection: pymongo collection object

    Returns:
        list: List of all documents in the collection
    """
    return list(mongo_collection.find())
