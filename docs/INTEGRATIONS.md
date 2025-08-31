# Integrations

[Back to Docs Hub](README.md)

This document describes the built-in integrations: search, fetch, OSINT, and Slack.

## Search (DuckDuckGo)

Module: `integrations/search.py`

- `web_search(query, max_results=5, safesearch='moderate') -> List[Dict]`
- `image_search(query, max_results=5, safesearch='moderate') -> List[Dict]`

If the `duckduckgo-search` package is missing or network is unavailable, these functions return empty lists.

API endpoints:

- POST `/search/web`
- POST `/search/images`

CLI:

- `python -m cli search web "query"`
- `python -m cli search images "query"`

## Fetch (HTML -> text)

Module: `integrations/fetch.py`

- `fetch_text(url, timeout=8.0, max_chars=8000) -> Optional[str]`

Uses httpx + BeautifulSoup(lxml). Returns None on errors or if optional deps are missing.

## OSINT & Networking

Module: `integrations/osint.py`

- `dns_lookup(hostname) -> DNSResult`
- `reverse_dns(ip) -> List[str]`
- `http_headers(url, timeout=5.0) -> HTTPHeaders`
- `ssl_cert_info(host, port=443, timeout=5.0) -> Optional[SSLCertInfo]`
- `robots_txt(url, timeout=5.0) -> str`
- `whois_lookup(domain) -> Dict` (requires python-whois)
- `tcp_scan(host, ports, timeout=0.6, concurrency=200) -> List[(port, state)]`
- `username_search(username, include_nsfw=False, sites=None, timeout=5.0) -> List[Dict]`

API endpoints mirror these utilities; all require X-API-Key and are intended for local use.

CLI examples:

```bash
python -m cli osint dns example.com
python -m cli osint rdns 1.1.1.1
python -m cli osint headers https://example.com
python -m cli osint ssl example.com --port 443
python -m cli osint robots https://example.com
python -m cli osint whois example.com
python -m cli osint username someuser --sites github,reddit
python -m cli net scan 127.0.0.1 22,80,443,8000-8010
```

## Slack

Module: `integrations/slack_connector.py`

- `send_slack_message(webhook_url, text) -> bool`

Configure `SLACK_WEBHOOK_URL` in `.env` to use the `/integrations/test` endpoint or the CLI command:

```bash
python -m cli integrations test
```

