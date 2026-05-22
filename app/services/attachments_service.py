"""Local text extraction for user-uploaded attachments (Path B).

Why local extraction:
    - Provider-independent: not coupled to OpenAI's or Anthropic's Files API.
    - Prepares the ground for RAG chunking in Module 3 — once the text is
      in our hands, we can chunk, embed, and store it.
    - Deterministic: extraction never depends on the LLM. The same file
      always produces the same text, so the exact-match cache keeps working.

Supported formats: PDF (``pypdf``), DOCX (``python-docx``), plain text
(``.txt`` / ``.md``). Anything else raises ``AttachmentExtractionError``.
"""

from __future__ import annotations

from io import BytesIO

import structlog

log = structlog.get_logger()


class AttachmentExtractionError(Exception):
    """Raised when a file cannot be parsed (unsupported format or corrupt)."""


def extract_text(filename: str, content: bytes) -> str:
    """Return the textual content of a single attachment."""
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            text = _extract_pdf(content)
        elif name.endswith(".docx"):
            text = _extract_docx(content)
        elif name.endswith((".txt", ".md")):
            text = content.decode("utf-8", errors="replace")
        else:
            raise AttachmentExtractionError(
                f"Unsupported attachment format: {filename!r}. "
                "Accepted: .pdf, .docx, .txt, .md."
            )
    except AttachmentExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 — wrap any parser-specific error
        log.warning(
            "attachment_extraction_failed",
            filename=filename,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        raise AttachmentExtractionError(
            f"Failed to extract text from {filename!r}: {exc}"
        ) from exc

    log.info(
        "attachment_extracted",
        filename=filename,
        chars=len(text),
        bytes=len(content),
    )
    return text.strip()


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text)
