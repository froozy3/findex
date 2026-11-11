from dataclasses import dataclass

from selectolax.lexbor import LexborHTMLParser

from findex.crawler.urls import resolve_link

IGNORED_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "form", "svg")


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    title: str
    text: str
    links: list[str]


def extract_page(html: bytes, base_url: str) -> ExtractedPage:
    tree = LexborHTMLParser(html)
    links = [
        link for node in tree.css("a[href]") if (link := resolve_link(base_url, node.attributes.get("href") or ""))
    ]
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    tree.strip_tags(list(IGNORED_TAGS))
    body = tree.body
    text = " ".join(body.text(separator=" ", strip=True).split()) if body else ""
    return ExtractedPage(title or base_url, text, links)
