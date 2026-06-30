#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单调试脚本：请求 /new/queryIndicatorsByCid 并打印响应摘要
"""
import requests, json

BASE_URL = "https://data.stats.gov.cn/dg/website/publicrelease/web/external"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python requests",
    "Content-Type": "application/json;charset=UTF-8",
}

CIDS = [
    "70551119a5104e13a5a2bbb3f36410f9",
]

for cid in CIDS:
    url = f"{BASE_URL}/new/queryIndicatorsByCid?cid={cid}"
    print("Requesting:", url)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        print("Status:", r.status_code)
        print("Content-Type:", r.headers.get('Content-Type'))
        text = r.text
        print("Response snippet (first 4000 chars):\n", text[:4000])
        try:
            j = r.json()
            print("Parsed JSON (type):", type(j))
            if isinstance(j, dict):
                print("Top keys:", list(j.keys()))
                print("JSON snippet:", json.dumps(j, ensure_ascii=False)[:4000])
        except Exception as e:
            print("JSON parse error:", e)
    except Exception as e:
        print("Request error:", e)
