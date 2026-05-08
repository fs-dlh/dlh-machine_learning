#!/usr/bin/env python3
"""Find schools by topic."""

def schools_by_topic(mongo_collection, topic):
    """
    Return list of schools having a specific topic.

    Args:
        mongo_collection: pymongo collection object
        topic (str): topic to search for

    Returns:
        list: list of school documents that contain the topic
    """
    return list(mongo_collection.find({"topics": topic}))
