from __future__ import annotations

import ipaddress
import re
import socket
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool
from langfuse import observe

from config import settings
from retriever import get_retriever


def debug_print(*args) -> None:
    if not settings.debug:
        return

    text = " ".join(str(arg) for arg in args)
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = (
            text.replace("🔧", "TOOL")
            .replace("📎", "RESULT")
            .replace("⏸️", "PAUSE")
        )
        print(fallback)


def preview_text(value: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


AUTO_SAVE_NOTE_MARKERS = (
    "best-effort draft",
    "maximum revision limit",
    "ліміт ревіз",
    "максимальн",
)


GENERIC_SECTION_TITLES = {
    "executive summary",
    "summary",
    "overview",
    "key findings",
    "findings",
    "notes",
    "sources",
    "source",
    "conclusion",
}


def _normalize_title_line(line: str) -> str:
    text = re.sub(r"\s+", " ", str(line)).strip().strip("#").strip()
    text = re.sub(r"[\._-]+", " ", text)
    return text[:120].strip()


def _slugify_filename(value: str) -> str:
    text = _normalize_title_line(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:80].strip("_")


def _extract_title_from_markdown(content: str) -> str | None:
    for line in str(content).splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        title = _normalize_title_line(stripped)
        if not title:
            continue

        if title.lower() in GENERIC_SECTION_TITLES:
            continue

        if stripped.startswith("# "):
            return title

    return None


def _ensure_markdown_title(content: str, fallback_title: str = "Research Report") -> str:
    text = str(content or "").lstrip()
    title = _extract_title_from_markdown(text) or _normalize_title_line(fallback_title) or "Research Report"

    if text.startswith("# "):
        first_line, *rest = text.splitlines()
        current_title = _normalize_title_line(first_line)
        normalized_first = f"# {title}"
        if current_title and current_title.lower() not in GENERIC_SECTION_TITLES:
            return text if first_line.strip() == normalized_first else "\n".join([normalized_first, *rest])
        return "\n".join([normalized_first, *rest])

    return f"# {title}\n\n{text}" if text else f"# {title}\n"


def _sanitize_filename(filename: str, content: str = "") -> str:
    original_name = Path(str(filename or "")).name
    stem = Path(original_name).stem

    candidate = _slugify_filename(stem)
    if not candidate:
        candidate = _slugify_filename(_extract_title_from_markdown(content or "") or "report")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{candidate}_{timestamp}.md"


def _is_auto_save_note(feedback: str) -> bool:
    text = str(feedback or "").strip().lower()
    return any(marker in text for marker in AUTO_SAVE_NOTE_MARKERS)


def _format_web_results(results: list[dict]) -> str:
    if not results:
        return "No web results found."

    lines = [f"[{len(results)} results found]"]
    for i, item in enumerate(results, 1):
        title = item.get("title") or "No title"
        url = item.get("href") or item.get("url") or ""
        snippet = item.get("body") or item.get("snippet") or ""
        lines.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {snippet}")
    return "\n\n".join(lines)


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("Invalid URL: hostname is missing.")

    forbidden_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    if parsed.hostname.lower() in forbidden_hosts:
        raise ValueError("Local addresses are not allowed.")

    try:
        ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("Private or local network addresses are not allowed.")
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve host: {exc}") from exc


def _format_knowledge_results(results: list[dict]) -> str:
    if not results:
        return "No results found in the local knowledge base."

    lines = [f"[{len(results)} documents found]"]
    for i, item in enumerate(results, 1):
        score = item.get("rerank_score", item.get("hybrid_score", 0.0))
        snippet = re.sub(r"\s+", " ", item.get("text", "").strip())[:350]

        lines.append(
            f"{i}. Source: {item.get('filename', 'unknown')}, page {item.get('page', '?')}"
        )
        lines.append(f"   Relevance: {score:.4f}")
        if snippet:
            suffix = "..." if len(snippet) == 350 else ""
            lines.append(f"   Text: {snippet}{suffix}")

    return "\n".join(lines)


@tool
@observe(name="web_search_tool")
def web_search(query: str) -> str:
    """Search the public web with DuckDuckGo for current information."""
    if not query or not isinstance(query, str):
        return "Invalid web search query."

    q = query.strip()
    debug_print(f'  🔧 web_search("{preview_text(q)}")')

    if len(q) < 2 or len(q) > 500:
        return "Query length must be between 2 and 500 characters."

    try:
        results = list(
            DDGS(timeout=settings.url_fetch_timeout_sec).text(
                q,
                max_results=settings.max_search_results,
            )
        )
        debug_print(f"  📎 [{len(results)} results found]")
        return _format_web_results(results)
    except Exception as exc:
        return f"web_search error: {exc}"


@tool
@observe(name="read_url_tool")
def read_url(url: str) -> str:
    """Read and extract the main text from a specific webpage."""
    if not url or not isinstance(url, str):
        return "Invalid URL."

    debug_print(f'  🔧 read_url("{preview_text(url, limit=120)}")')

    try:
        _validate_remote_url(url)
        downloaded = trafilatura.fetch_url(
            url,
            timeout=settings.url_fetch_timeout_sec,
        )
        if not downloaded:
            return "Unable to download the page."

        text = trafilatura.extract(downloaded)
        if not text:
            return "Unable to extract useful text from the page."

        debug_print(f"  📎 [{len(text)} chars]")
        return text[: settings.max_url_content_length]
    except Exception as exc:
        return f"read_url error: {exc}"


@tool
@observe(name="knowledge_search_tool")
def knowledge_search(query: str) -> str:
    """Search the local RAG knowledge base built from the course materials."""
    if not query or not isinstance(query, str):
        return "Invalid knowledge search query."

    q = query.strip()
    debug_print(f'  🔧 knowledge_search("{preview_text(q)}")')

    if len(q) < 2:
        return "Query is too short."

    try:
        retriever = get_retriever()
        results = retriever.search(q)
        debug_print(f"  📎 [{len(results)} documents found]")
        return _format_knowledge_results(results)
    except FileNotFoundError:
        return "Knowledge index not found. Run `python ingest.py` first."
    except Exception as exc:
        return f"knowledge_search error: {exc}"


@tool
@observe(name="save_report_tool")
def save_report(filename: str, content: str, feedback: str = "") -> str:
    """Save the final markdown report to the output folder. This tool is HITL-gated."""
    feedback_text = str(feedback or "").strip()
    debug_print(
        f'  🔧 save_report(filename="{filename}", feedback="{preview_text(feedback_text, limit=80)}")'
    )

    if feedback_text:
        if _is_auto_save_note(feedback_text):
            content = f"> **Note:** {feedback_text}\n\n{content}"
        else:
            return (
                "REPORT NOT SAVED.\n"
                "Reviewer requested changes before saving.\n"
                f"Feedback: {feedback_text}\n"
                "Do not tell the user the report was saved. Revise the draft using this feedback, "
                "then call save_report again with updated content."
            )

    if not isinstance(content, str) or not content.strip():
        return "Report content is empty."

    normalized_content = _ensure_markdown_title(
        content,
        fallback_title=filename or "Research Report",
    )
    safe_name = _sanitize_filename(filename, normalized_content)
    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = (output_dir / safe_name).resolve()
    if path.parent != output_dir:
        return "Invalid save path."

    path.write_text(normalized_content, encoding="utf-8")
    debug_print(f"  📎 saved to: {path}")

    # Return a compact excerpt so the Supervisor can include key findings in its
    # final reply without re-generating the full content (saves output tokens).
    excerpt = normalized_content.strip()[:2000]
    if len(normalized_content.strip()) > 2000:
        excerpt += "\n\n[…full report saved to file…]"
    return (
        f"Report saved to {path}\n\n"
        f"--- REPORT EXCERPT (include key findings from this in your reply) ---\n"
        f"{excerpt}"
    )
