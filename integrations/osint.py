"""
osint.py - Local OSINT and network utilities for Vega2.0

Notes:
- All functions operate locally without external APIs (except whois which may depend on python-whois if installed; otherwise gracefully degrades).
- Network scanning is intentionally conservative and single-host limited.
- Always validate/limit inputs at the API/CLI layer.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, cast

import http.client
from urllib.parse import urlparse, quote
from contextlib import closing


@dataclass
class DNSResult:
    hostname: str
    addresses: List[str]


def dns_lookup(hostname: str) -> DNSResult:
    infos = socket.getaddrinfo(hostname, None)
    addrs: List[str] = []
    for info in infos:
        addr = str(info[4][0])
        if addr not in addrs:
            addrs.append(addr)
    return DNSResult(hostname=hostname, addresses=addrs)


def reverse_dns(ip: str) -> List[str]:
    try:
        host, aliases, _ = socket.gethostbyaddr(ip)
        names = [host] + list(aliases)
        # dedupe
        out = []
        for n in names:
            if n not in out:
                out.append(n)
        return out
    except Exception:
        return []


@dataclass
class HTTPHeaders:
    url: str
    status: int
    headers: Dict[str, str]


def http_headers(url: str, timeout: float = 5.0) -> HTTPHeaders:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    host = parsed.netloc or parsed.path
    path = parsed.path or "/"
    if scheme == "https":
        conn = http.client.HTTPSConnection(host, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, timeout=timeout)
    try:
        conn.request("HEAD", path)
        resp = conn.getresponse()
        hdrs = {k: v for k, v in resp.getheaders()}
        return HTTPHeaders(url=url, status=resp.status, headers=hdrs)
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class SSLCertInfo:
    host: str
    subject: str
    issuer: str
    not_before: str
    not_after: str


def ssl_cert_info(
    host: str, port: int = 443, timeout: float = 5.0
) -> Optional[SSLCertInfo]:
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert() or {}
                subject = ", ".join(
                    "{}={}".format(k, v)
                    for t in cast(list, cert.get("subject", []))
                    for k, v in t
                )
                issuer = ", ".join(
                    "{}={}".format(k, v)
                    for t in cast(list, cert.get("issuer", []))
                    for k, v in t
                )
                nb_str = str(cert.get("notBefore", ""))
                na_str = str(cert.get("notAfter", ""))
                return SSLCertInfo(
                    host=host,
                    subject=str(subject),
                    issuer=str(issuer),
                    not_before=nb_str,
                    not_after=na_str,
                )
    except Exception:
        return None


def robots_txt(url: str, timeout: float = 5.0) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    host = parsed.netloc or parsed.path
    robots_url = f"{scheme}://{host}/robots.txt"
    conn = (
        http.client.HTTPSConnection(host, timeout=timeout)
        if scheme == "https"
        else http.client.HTTPConnection(host, timeout=timeout)
    )
    try:
        conn.request("GET", "/robots.txt")
        resp = conn.getresponse()
        if resp.status >= 400:
            return ""
        body = resp.read(65536).decode("utf-8", errors="ignore")
        return body
    except Exception:
        return ""
    finally:
        try:
            conn.close()
        except Exception:
            pass


def whois_lookup(domain: str) -> Dict:
    # Lazy import to prevent "Unable to import 'whois'" at module import time
    try:
        import whois as _whois  # type: ignore
    except Exception:
        return {"error": "python-whois not installed"}
    try:
        w = _whois.whois(domain)
        # Convert to JSON-serializable dict
        out = {}
        for k, v in w.items():
            try:
                if hasattr(v, "isoformat"):
                    out[k] = v.isoformat()
                elif isinstance(v, (list, tuple)):
                    out[k] = [str(x) for x in v]
                elif v is None:
                    out[k] = None
                else:
                    out[k] = str(v)
            except Exception:
                out[k] = str(v)
        return out
    except Exception as exc:
        return {"error": str(exc)}


async def tcp_scan(
    host: str, ports: List[int], timeout: float = 0.6, concurrency: int = 200
) -> List[Tuple[int, str]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Tuple[int, str]] = []

    async def check(p: int):
        try:
            async with sem:
                conn = asyncio.open_connection(host, p)
                reader, writer = await asyncio.wait_for(conn, timeout=timeout)
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                results.append((p, "open"))
        except Exception:
            # closed/filtered
            pass

    tasks = [asyncio.create_task(check(p)) for p in ports]
    for t in tasks:
        try:
            await t
        except Exception:
            pass
    results.sort(key=lambda x: x[0])
    return results


def username_search(
    username: str,
    include_nsfw: bool = False,
    sites: Optional[List[str]] = None,
    timeout: float = 5.0,
) -> List[Dict]:
    """
    Check if a username exists on a curated list of sites by probing profile URLs.

    Uses stdlib http.client. HEAD first (if allowed), then GET fallback with small read.
    Returns list of {site, url, exists, status, note}.
    """
    if not username or len(username) > 64:
        return []

    # Curated targets; broad coverage of popular platforms (dev, social, creative, blogs).
    # Only include URLs that plausibly map to username-based profile pages (avoid numeric-id-only sites).
    # Each tuple: (name, url_template, is_nsfw)
    all_sites: List[Tuple[str, str, bool]] = [
        # Developer and package ecosystems
        ("github", "https://github.com/{u}", False),
        ("gitlab", "https://gitlab.com/{u}", False),
        ("bitbucket", "https://bitbucket.org/{u}", False),
        ("sourceforge", "https://sourceforge.net/u/{u}/profile", False),
        ("npm", "https://www.npmjs.com/~{u}", False),
        ("pypi", "https://pypi.org/user/{u}/", False),
        ("crates", "https://crates.io/users/{u}", False),
        ("nuget", "https://www.nuget.org/profiles/{u}", False),
        ("packagist", "https://packagist.org/users/{u}", False),
        ("rubygems", "https://rubygems.org/profiles/{u}", False),
        ("dockerhub", "https://hub.docker.com/u/{u}", False),
        ("anaconda", "https://anaconda.org/{u}", False),
        ("hexpm", "https://hex.pm/users/{u}", False),
        ("hackage", "https://hackage.haskell.org/user/{u}", False),
        ("metacpan", "https://metacpan.org/author/{u}", False),
        # Coding communities and sandboxes
        ("devto", "https://dev.to/{u}", False),
        ("hashnode", "https://hashnode.com/@{u}", False),
        ("codepen", "https://codepen.io/{u}", False),
        ("codesandbox", "https://codesandbox.io/u/{u}", False),
        ("replit", "https://replit.com/@{u}", False),
        ("glitch", "https://glitch.com/@{u}", False),
        ("sourcehut", "https://sr.ht/~{u}", False),
        ("launchpad", "https://launchpad.net/~{u}", False),
        ("exercism", "https://exercism.org/profiles/{u}", False),
        ("codeforces", "https://codeforces.com/profile/{u}", False),
        ("codewars", "https://www.codewars.com/users/{u}", False),
        ("leetcode", "https://leetcode.com/{u}/", False),
        ("hackerrank", "https://www.hackerrank.com/{u}", False),
        ("tryhackme", "https://tryhackme.com/p/{u}", False),
        # AI/ML communities
        ("huggingface", "https://huggingface.co/{u}", False),
        ("civitai", "https://civitai.com/user/{u}", False),
        ("replicate", "https://replicate.com/{u}", False),
        # Social/media
        ("x", "https://x.com/{u}", False),
        ("twitter", "https://twitter.com/{u}", False),
        ("reddit", "https://www.reddit.com/user/{u}", False),
        ("instagram", "https://www.instagram.com/{u}/", False),
        ("tiktok", "https://www.tiktok.com/@{u}", False),
        ("youtube", "https://www.youtube.com/@{u}", False),
        ("twitch", "https://www.twitch.tv/{u}", False),
        ("vimeo", "https://vimeo.com/{u}", False),
        ("snapchat", "https://www.snapchat.com/add/{u}", False),
        ("telegram", "https://t.me/{u}", False),
        ("vk", "https://vk.com/{u}", False),
        ("quora", "https://www.quora.com/profile/{u}", False),
        ("keybase", "https://keybase.io/{u}", False),
        ("gravatar", "https://en.gravatar.com/{u}", False),
        ("producthunt", "https://www.producthunt.com/@{u}", False),
        ("tumblr", "https://{u}.tumblr.com", False),
        ("wordpress", "https://{u}.wordpress.com", False),
        ("blogger", "https://{u}.blogspot.com", False),
        ("livejournal", "https://{u}.livejournal.com", False),
        ("substack", "https://{u}.substack.com", False),
        ("wikipedia", "https://en.wikipedia.org/wiki/User:{u}", False),
        ("hn", "https://news.ycombinator.com/user?id={u}", False),
        ("lobsters", "https://lobste.rs/u/{u}", False),
        ("disqus", "https://disqus.com/by/{u}/", False),
        # Creative/portfolio
        ("behance", "https://www.behance.net/{u}", False),
        ("dribbble", "https://dribbble.com/{u}", False),
        ("artstation", "https://www.artstation.com/{u}", False),
        ("deviantart", "https://www.deviantart.com/{u}", False),
        ("500px", "https://500px.com/{u}", False),
        ("flickr", "https://www.flickr.com/people/{u}/", False),
        ("pinterest", "https://www.pinterest.com/{u}/", False),
        ("soundcloud", "https://soundcloud.com/{u}", False),
        ("bandcamp", "https://bandcamp.com/{u}", False),
        # Gaming and misc
        ("steam", "https://steamcommunity.com/id/{u}", False),
        ("chesscom", "https://www.chess.com/member/{u}", False),
        ("lichess", "https://lichess.org/@/{u}", False),
        # Pro and freelancing
        ("linkedin", "https://www.linkedin.com/in/{u}/", False),
        ("patreon", "https://www.patreon.com/{u}", False),
        ("kofi", "https://ko-fi.com/{u}", False),
        ("buymeacoffee", "https://www.buymeacoffee.com/{u}", False),
        ("gumroad", "https://gumroad.com/{u}", False),
        ("fiverr", "https://www.fiverr.com/{u}", False),
        ("freelancer", "https://www.freelancer.com/u/{u}", False),
    ]

    nsfw_sites: List[Tuple[str, str, bool]] = [
        ("onlyfans", "https://onlyfans.com/{u}", True),
        ("fansly", "https://fansly.com/{u}", True),
        ("pornhub", "https://www.pornhub.com/users/{u}", True),
        ("xvideos", "https://www.xvideos.com/profiles/{u}", True),
        ("xhamster", "https://xhamster.com/users/{u}", True),
        ("chaturbate", "https://chaturbate.com/{u}", True),
        ("myfreecams", "https://profiles.myfreecams.com/{u}", True),
        ("cam4", "https://www.cam4.com/{u}", True),
        ("redgifs", "https://www.redgifs.com/users/{u}", True),
        ("fapello", "https://fapello.com/{u}/", True),
    ]

    targets3 = all_sites + (nsfw_sites if include_nsfw else [])
    # Honor explicit site filter if provided
    if sites:
        sset = {s.lower() for s in sites}
        targets3 = [t for t in targets3 if t[0] in sset]

    results: List[Dict] = []

    def _probe(url: str) -> Tuple[bool, int, str]:
        p = urlparse(url)
        scheme = p.scheme or "https"
        host = p.netloc
        # Rebuild full path including query string if present
        path = p.path or "/"
        if p.query:
            path = f"{path}?{p.query}"
        conn_cls = (
            http.client.HTTPSConnection
            if scheme == "https"
            else http.client.HTTPConnection
        )
        try:
            with closing(conn_cls(host, timeout=timeout)) as conn:
                # Prefer HEAD; many sites allow and it's faster
                try:
                    conn.request(
                        "HEAD",
                        path,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Vega2.0 OSINT; +https://local)",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        },
                    )
                    resp = conn.getresponse()
                    status = resp.status
                    resp.read(0)
                except Exception:
                    # Fallback to GET, but limit body read
                    try:
                        conn.request(
                            "GET",
                            path,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Vega2.0 OSINT; +https://local)",
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            },
                        )
                        resp = conn.getresponse()
                        status = resp.status
                        _ = resp.read(2048)
                    except Exception:
                        return False, 0, "connection error"
                # Heuristic: 200/301/302 often indicate existence; 404 -> not exists
                if status in (200, 301, 302, 303):
                    return True, status, "ok"
                if status in (401, 403):
                    # Many sites protect profiles; treat as inconclusive but likely exists
                    return True, status, "protected"
                if status == 404:
                    return False, status, "not found"
                return False, status, "unknown"
        except Exception:
            return False, 0, "connection error"

    fmt_u = quote(username, safe="-_.~")
    for name, tpl, _is_nsfw in targets3:
        url = tpl.format(u=fmt_u)
        exists, status, note = _probe(url)
        results.append(
            {
                "site": name,
                "url": url,
                "exists": bool(exists),
                "status": int(status),
                "note": note,
                "nsfw": bool(_is_nsfw),
            }
        )
    return results
