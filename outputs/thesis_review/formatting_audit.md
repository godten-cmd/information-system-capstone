# Thesis Formatting Audit Report
## outputs/thesis_latex/ — Phase 11

**Audit Date**: 2026-06-03  
**Compiler**: pdflatex (TeX Live 2025)  
**Document Class**: report, 12pt, a4paper, twoside  
**Final Page Count**: 58 pages

---

## I. Compilation Warnings Summary

| Warning | Severity | Status |
|---|---|---|
| `Package microtype Warning: Unable to apply patch 'footnote'` | Minor | Cosmetic only; microtype cannot patch footnotes in report class. Harmless. |
| `Package hyperref Warning: Token not allowed in a PDF string (Unicode)` (×2) | Minor | Occurs in chapter title with math tokens in TOC bookmarks. Use `\texorpdfstring{}{}` to fix. |
| `Package natbib Warning: Citation may have changed` | Minor | Normal on multi-pass compile; resolves on final pass. |
| `LaTeX Warning: Label(s) may have changed. Rerun` | Minor | Normal; resolved by final rerun. |

No critical errors remain in the final compiled PDF.

---

## II. Blank Pages and Page Break Issues

### Finding 2.1 — Unnecessary Blank Pages from `\cleardoublepage`
- **Location**: thesis.tex, lines 154, 157, 162, 167 (between Abstract, TOC, LoF, LoT, Acronyms)
- **Issue**: `\cleardoublepage` in a twoside `report` class forces odd-page starts, producing blank even pages between front-matter sections. In a 58-page document, this produces 3–4 blank pages that interrupt reading flow.
- **Status**: Applied fix — replaced `\cleardoublepage` between front-matter sections with `\clearpage`.
- **Exception**: `\cleardoublepage` before main chapters is kept for proper chapter-start convention.

### Finding 2.2 — Acronym Table Not Floated
- **Location**: thesis.tex, Notation section (after LoT, before main matter)
- **Issue**: The Notation table uses bare `\begin{tabular}` inside a `\chapter*` without a float wrapper. If the table overflows onto a second page, it will have no caption and no reference.
- **Status**: Applied fix — wrapped in `\begin{table}[H]` with a null caption.

### Finding 2.3 — Abstract `\vfill` Creates Excessive Whitespace
- **Location**: abstract.tex (end of abstract environment)
- **Issue**: The abstract environment ends with `\vfill`, which pushes the keyword line to the bottom of the page. On short abstracts this creates ~10 cm of whitespace. The abstract is approximately 350 words and fits comfortably on one page without pushing to the bottom.
- **Status**: Applied fix — removed `\vfill` from abstract environment definition. Keywords now appear immediately after abstract body text.

---

## III. Figure Placement Issues

### Finding 3.1 — Figures May Float Far from Text Reference
- **Location**: results.tex — all five figures use `[htbp]`
- **Issue**: With a large amount of surrounding table content, the `[htbp]` placement algorithm may push figures several pages past their reference point. Figure 3 (heatmap) and Figure 4 (category bar) are particularly at risk given the surrounding large tables.
- **Recommendation**: Add `\FloatBarrier` from the `placeins` package after each figure-table cluster in results.tex to prevent figures from floating past their section boundary.
- **Status**: Applied fix — added `\usepackage{placeins}` and `\FloatBarrier` calls.

### Finding 3.2 — Figure Path Relative Reference
- **Location**: thesis.tex `\graphicspath{{../final_report/figures/}}`
- **Issue**: The figure path is relative and will break if the thesis is compiled from a different working directory. All five figures are resolved correctly when compiled from `outputs/thesis_latex/`, but any automated build system that changes directory will fail.
- **Recommendation**: Note this in a compilation README. Alternatively, copy figures to `outputs/thesis_latex/figures/` and update the path.
- **Status**: Not auto-applied (would require copying files). Documented.

---

## IV. Table Placement Issues

### Finding 4.1 — Wide Tables in Two-Column-Spanning Environment
- **Location**: results.tex — Tables `retrieval_performance`, `reasoning_type`, `category` use `table*` (span two columns)
- **Issue**: In the `report` class with single-column layout, `table*` has no effect and is redundant. The tables compile correctly, but the `*` environment may interact unexpectedly with `float` and `placeins`.
- **Status**: Applied fix — changed `table*` to `table` in results.tex (single-column document has no need for column-spanning tables).

### Finding 4.2 — Category Table Column Count Mismatch
- **Location**: results.tex, Table `tab:category`
- **Issue**: The category table header declares 11 columns (`lrrrrrrrrrr`) but the data contains 9 data columns plus 1 label column = 10 columns. This causes pdflatex to issue an "Extra alignment tab has been changed to \cr" warning and may produce misaligned columns.
- **Status**: Applied fix — corrected column specification to `lrrrrrrrrr` (10 columns) matching the actual data.

---

## V. Cross-Reference and Citation Issues

### Finding 5.1 — `\ref{sec:method_eval}` in results.tex
- **Location**: results.tex, Section 4.6 last paragraph: "Using the effect size classification from Section~\ref{sec:method_eval}"
- **Issue**: The label `\sec:method_eval` is defined in methodology.tex. This cross-reference is valid, but if methodology.tex is ever separated from the main compilation, it will produce an undefined reference.
- **Status**: Verified correct — cross-reference resolves in full compilation.

### Finding 5.2 — `\ref{fig:heatmap}` in results.tex
- **Location**: results.tex, Section 4.1.4
- **Issue**: `\ref{fig:heatmap}` references Figure 3 (reasoning type heatmap). This label is defined later in results.tex. Forward references to figures are valid in LaTeX but produce "Label may have changed" warnings on the first compile pass.
- **Status**: Resolves correctly on multi-pass compile. No action needed.

### Finding 5.3 — `luan2021sparse_dense` citation produces "may have changed" warning
- **Location**: discussion.tex — uses `\citep{luan2021sparse_dense}` in Implication 1 (footnote/inline)
- **Issue**: On first bibtex pass this citation was undefined; resolves on second pass. The BibTeX entry exists in references.bib and resolves correctly.
- **Status**: Resolved in final compile. No action needed.

---

## VI. Bibliography Issues

### Finding 6.1 — Empty booktitle in `nakano2022webgpt`
- **Location**: references.bib
- **Issue**: The original entry used `@inproceedings` with no `booktitle` field. BibTeX warned "empty booktitle". 
- **Status**: Fixed (changed to `@article` — the paper is an arXiv preprint with no conference proceedings).

### Finding 6.2 — `openai2024gpt4` uses `@techreport` without standard fields
- **Location**: references.bib
- **Issue**: The OpenAI GPT-4 technical report entry uses `@techreport` with `note = {arXiv:2303.08774}`. While correct, some bibliography styles display this inconsistently.
- **Status**: Acceptable. No change needed.

### Finding 6.3 — `su2023hybrid` appears unused
- **Location**: references.bib
- **Issue**: The entry `su2023hybrid` is defined but never cited in any chapter file.
- **Status**: Recommend removing to avoid unused bibliography warnings.

### Finding 6.4 — `salton1983vector` appears unused
- **Location**: references.bib
- **Issue**: Defined but not cited in the thesis text.
- **Status**: Recommend removing.

---

## VII. Orphan/Widow Paragraph Issues

### Finding 7.1 — Potential widow lines in Results chapter
- **Location**: results.tex — dense metric discussions
- **Issue**: Several paragraphs ending sections (e.g., end of Section 4.1.2) may produce widow lines (single line at top of new page). The one-and-a-half line spacing increases this risk.
- **Status**: Applied global fix — added `\clubpenalty=10000` and `\widowpenalty=10000` to thesis.tex preamble.

---

## VIII. Other Formatting Issues

### Finding 8.1 — Title Page Has Placeholder
- **Location**: thesis.tex, line 145
- **Issue**: `Supervisor: [Supervisor Name]` is a literal placeholder.
- **Status**: CRITICAL — must be replaced before submission.

### Finding 8.2 — Notation Table Uses `\toprule`/`\midrule`/`\bottomrule` Without `booktabs` in Chapter*
- **Location**: thesis.tex, Notation chapter
- **Issue**: The notation table correctly uses booktabs, but it is not a float (no `\begin{table}`), which means it cannot be referenced by number and will not appear in the List of Tables.
- **Status**: Applied fix — wrapped in `table[H]` float with no number (uses `\caption*{}`).

### Finding 8.3 — Chapter Numbering Restart After Appendix
- **Location**: thesis.tex appendix section
- **Issue**: `\appendix` resets chapter counter to A, B, C... This is correct behavior. However, labels in appendix chapters (e.g., `\label{app:sekd}`) use alphabetic chapter IDs which display correctly in cross-references.
- **Status**: Correct — no action needed.

### Finding 8.4 — `setspace` + `microtype` Interaction
- **Location**: thesis.tex preamble
- **Issue**: `\onehalfspacing` from `setspace` interacts with microtype's `\footnote` patch, causing the "Unable to apply patch 'footnote'" warning. This is a known limitation of the two packages.
- **Status**: Harmless cosmetic warning. Can be suppressed with `\microtypesetup{activate={true,nocompatibility}}`.
