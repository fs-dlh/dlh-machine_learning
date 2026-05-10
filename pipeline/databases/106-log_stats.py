#!/usr/bin/env python3
"""Provide stats about Nginx logs stored in MongoDB including top IPs."""

from pymongo import MongoClient
import sys

def log_stats():
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    out_lines = []
    total = collection.count_documents({})
    out_lines.append(f"{total} logs")
    out_lines.append("Methods:")
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        count = collection.count_documents({"method": method})
        out_lines.append(f"\tmethod {method}: {count}")
    status_count = collection.count_documents({"method": "GET", "path": "/status"})
    out_lines.append(f"{status_count} status check")
    out_lines.append("IPs:")
    pipeline = [
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    top_ips = collection.aggregate(pipeline)
    for ip_doc in top_ips:
        out_lines.append(f"{ip_doc['_id']}: {ip_doc['count']}")
    # Join with newline and add a final newline
    sys.stdout.write("\n".join(out_lines) + "\n")

if __name__ == "__main__":
    log_stats()
    
