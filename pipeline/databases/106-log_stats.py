#!/usr/bin/env python3
"""Provide stats about Nginx logs stored in MongoDB including top IPs."""

from pymongo import MongoClient

def log_stats():
    """Display statistics from the logs.nginx collection."""
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client.logs
    collection = db.nginx

    total = collection.count_documents({})
    print(f"{total} logs")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_count = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_count} status check")

    # Top 10 most present IPs
    print("IPs:")
    pipeline = [
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    top_ips = collection.aggregate(pipeline)
    for ip_doc in top_ips:
        print(f"    {ip_doc['_id']}: {ip_doc['count']}\n")

if __name__ == "__main__":
    log_stats()
    