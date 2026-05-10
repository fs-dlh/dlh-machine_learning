#!/usr/bin/env python3
"""Provide stats about Nginx logs stored in MongoDB including top IPs."""

from pymongo import MongoClient

def log_stats():
    """Display statistics from the logs.nginx collection."""
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx
    total = collection.count_documents({})
    print(f"{total} logs")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")
    status_count = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_count} status check")

    all_docs = collection.find()
    ip_count = {}
    for doc in all_docs:
        ip = doc.get("ip")
        if ip:
            ip_count[ip] = ip_count.get(ip, 0) + 1
    ip_list = [(ip, count) for ip, count in ip_count.items()]
    ip_list.sort(key=lambda x: x[1], reverse=True)
    print("IPs:")
    top_10 = ip_list[:10]
    for ip, count in top_10:
        print(f"\t{ip}: {count}")

if __name__ == "__main__":
    log_stats()
    