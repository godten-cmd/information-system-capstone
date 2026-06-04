"""DocumentAssembler: Jinja2 rendering, section/table parsing, and EnterpriseDocument construction."""

from __future__ import annotations

import re
from pathlib import Path

import jinja2

from .schemas import DocumentMetadata, DocumentSection, DocumentTable, EnterpriseDocument

_SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE)
_MAX_HEADING_WORDS = 5  # structural headings are ≤5-word noun phrases; longer lines are list items

_TABLE_BLOCK_RE = re.compile(
    r"(?P<block>(?:\|[^\n]+\|\n)+)",
    re.MULTILINE,
)
_TABLE_ROW_RE = re.compile(r"\|([^|]+)")
_SEP_ROW_RE = re.compile(r"^\s*[-:]+\s*$")


def _cells(line: str) -> list[str]:
    return [m.group(1).strip() for m in _TABLE_ROW_RE.finditer(line)]


class DocumentAssembler:
    """Render Jinja2 templates and parse the resulting body into structured schema objects."""

    def __init__(self, template_dir: Path) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=False,
            undefined=jinja2.StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, template_name: str, context: dict) -> str:
        tmpl = self._env.get_template(template_name)
        return tmpl.render(**context)

    # ── Section parsing ───────────────────────────────────────────────────────

    def parse_sections(self, body: str) -> list[DocumentSection]:
        structural_matches: list[re.Match] = []
        expected_num = 1

        for m in _SECTION_RE.finditer(body):
            num = int(m.group(1))
            heading_text = m.group(2).strip()

            # Filter 1: structural headings never end with a period
            # (catches procedure steps like "Submit a leave request through HYPortal.")
            if heading_text.endswith("."):
                continue

            # Filter 2: structural headings are short noun phrases (≤5 words)
            # (catches agenda items like "Prioritise backlog items for the sprint")
            if len(heading_text.split()) > _MAX_HEADING_WORDS:
                continue

            # Filter 3: section numbers must be strictly sequential (1, 2, 3, …)
            # (catches short agenda items appearing before the actual heading with that number)
            if num != expected_num:
                continue

            # Filter 4: structural headings must be preceded by a blank line (\n\n).
            # Agenda items inside a section body are separated by single \n, not \n\n,
            # so this cleanly rejects "2. Review ADRs..." inside Meeting Overview
            # while accepting "2. Attendance" which follows a blank line.
            pos = m.start()
            if pos >= 2 and body[pos - 2 : pos] != "\n\n":
                continue

            structural_matches.append(m)
            expected_num += 1

        sections: list[DocumentSection] = []
        for i, m in enumerate(structural_matches):
            heading_text = m.group(2).strip()
            char_start = m.start()
            char_end = (
                structural_matches[i + 1].start()
                if i + 1 < len(structural_matches)
                else len(body)
            )
            if char_end <= char_start:
                char_end = char_start + 1
            content = body[char_start:char_end]
            sections.append(
                DocumentSection(
                    heading=heading_text,
                    level=1,
                    path=[f"{m.group(1)}. {heading_text}"],
                    content=content,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        return sections

    # ── Table parsing ─────────────────────────────────────────────────────────

    def parse_tables(
        self,
        body: str,
        sections: list[DocumentSection],
    ) -> list[DocumentTable]:
        tables: list[DocumentTable] = []
        for m in _TABLE_BLOCK_RE.finditer(body):
            block = m.group("block")
            char_start = m.start()
            char_end = m.end()

            raw_rows = [line for line in block.splitlines() if line.strip().startswith("|")]
            if len(raw_rows) < 2:
                continue

            headers = _cells(raw_rows[0])
            if not headers:
                continue

            data_rows: list[list[str]] = []
            for row_line in raw_rows[1:]:
                cells = _cells(row_line)
                if all(_SEP_ROW_RE.match(c) for c in cells):
                    continue
                if len(cells) != len(headers):
                    cells = (cells + [""] * len(headers))[: len(headers)]
                data_rows.append(cells)

            section_path = self._section_path_at(char_start, sections)

            tables.append(
                DocumentTable(
                    section_path=section_path,
                    headers=headers,
                    rows=data_rows,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        return tables

    def _section_path_at(self, offset: int, sections: list[DocumentSection]) -> list[str]:
        """Return the path of the innermost section containing *offset*."""
        containing = [s for s in sections if s.char_start <= offset < s.char_end]
        if containing:
            return containing[-1].path
        return ["Document"]

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_document(
        self,
        metadata: DocumentMetadata,
        body: str,
        references: list[str] | None = None,
    ) -> EnterpriseDocument:
        sections = self.parse_sections(body)
        tables = self.parse_tables(body, sections)
        return EnterpriseDocument(
            metadata=metadata,
            body=body,
            sections=sections,
            tables=tables,
            references=references or [],
        )
