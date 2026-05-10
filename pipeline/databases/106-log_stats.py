#!/usr/bin/env python3
"""Provide stats about Nginx logs stored in MongoDB including top IPs."""

from pymongo import MongoClient
import sys

def log_stats():
    """Display statistics from the logs.nginx collection."""
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    total = collection.count_documents({})
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    lines = []
    lines.append(f"{total} logs")
    lines.append("Methods:")
    for method in methods:
        count = collection.count_documents({"method": method})
        lines.append(f"\tmethod {method}: {count}")

    status_count = collection.count_documents({"method": "GET", "path": "/status"})
    lines.append(f"{status_count} status check")
    lines.append("IPs:")

    pipeline = [
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    top_ips = collection.aggregate(pipeline)
    for ip_doc in top_ips:
        lines.append(f"{ip_doc['_id']}: {ip_doc['count']}")

    sys.stdout.write("\n".join(lines))

if __name__ == "__main__":
    log_stats()
