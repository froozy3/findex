from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}
SKIPPED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip", ".gz", ".css", ".js", ".ico", ".mp4")


def canonicalize(url: str) -> str:
    """Lowercase scheme/host, drop fragment and default port, sort the query, strip a trailing slash."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    port = parts.port
    netloc = host if port is None or DEFAULT_PORTS.get(parts.scheme) == port else f"{host}:{port}"
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def resolve_link(base_url: str, href: str) -> str | None:
    """Absolute canonical URL for a link found on `base_url`, or None if it isn't crawlable."""
    absolute = urljoin(base_url, href.strip())
    parts = urlsplit(absolute)
    if parts.scheme not in DEFAULT_PORTS or not parts.hostname or parts.path.lower().endswith(SKIPPED_EXTENSIONS):
        return None
    return canonicalize(absolute)


def host_of(url: str) -> str:
    return urlsplit(url).netloc


def is_domain_allowed(url: str, allowed_domains: frozenset[str]) -> bool:
    hostname = urlsplit(url).hostname or ""
    return not allowed_domains or any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains
    )
