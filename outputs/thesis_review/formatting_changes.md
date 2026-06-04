# Formatting Changes Applied to English Thesis
## outputs/thesis_latex/ — Phase 11

**Applied**: 2026-06-03  
**Page count change**: 58 → 55 pages (3 blank pages removed)

---

## Changes Applied

### 1. Removed `\vfill` from Abstract Environment
- **File**: `thesis.tex`
- **Change**: Removed trailing `\vfill` from `abstract` environment definition
- **Effect**: Eliminated ~10 cm of forced whitespace at bottom of abstract page. Keywords now appear immediately after abstract body.

### 2. Replaced `\cleardoublepage` with `\clearpage` in Front Matter
- **File**: `thesis.tex`
- **Change**: All `\cleardoublepage` calls between TOC, LoF, LoT, and Notation sections changed to `\clearpage`
- **Effect**: Eliminated 3 unnecessary blank even-pages in the front matter. The `\cleardoublepage` before main chapter input calls was retained to preserve proper chapter-start convention for twoside printing.

### 3. Added Float Barrier Package and Widow/Orphan Penalties
- **File**: `thesis.tex` (preamble)
- **Added**:
  ```latex
  \usepackage{placeins}
  \clubpenalty=10000
  \widowpenalty=10000
  \microtypesetup{activate={true,nocompatibility}}
  ```
- **Effect**: Prevents figures from drifting past section boundaries; suppresses orphan/widow lines; suppresses microtype/setspace compatibility warning.

### 4. Added `\FloatBarrier` Before Major Sections in Results
- **File**: `results.tex`
- **Change**: Added `\FloatBarrier` before sections 4.2, 4.3, 4.5
- **Effect**: Prevents figures/tables from floating past their parent section boundary.

### 5. Changed `table*` to `table` in Results Chapter
- **File**: `results.tex`
- **Change**: All `\begin{table*}` / `\end{table*}` environments changed to `\begin{table}` / `\end{table}`
- **Effect**: Eliminates unnecessary two-column spanning (single-column document), removes potential float conflicts.

### 6. Fixed Category Table Column Specification
- **File**: `results.tex`
- **Change**: `{lrrrrrrrrrr}` (11 cols) → `{lrcccccccc}` (10 cols)
- **Effect**: Eliminates "Extra alignment tab" LaTeX warning; ensures column header alignment matches data.

### 7. Fixed `nakano2022webgpt` BibTeX Entry Type
- **File**: `references.bib`
- **Change**: `@inproceedings` → `@article` (paper is an arXiv preprint with no conference)
- **Effect**: Eliminates "empty booktitle" BibTeX warning.

---

## Changes NOT Applied (Require Manual Action)

| Issue | Reason Not Auto-Applied |
|---|---|
| Replace `[Supervisor Name]` placeholder | Requires user input — actual supervisor name unknown |
| Copy figures to `thesis_latex/figures/` | Would duplicate large PNG files |
| Remove unused BibTeX entries (`su2023hybrid`, `salton1983vector`) | Low risk — unused entries do not cause errors |
| Add bootstrap confidence intervals | Requires running Python analysis — no experiment rerun needed but code execution required |
| Add `\texorpdfstring{}{}` for math in chapter titles | Minor cosmetic; bookmarks work correctly without it |

---

## Compilation Status After Changes

| Pass | Result |
|---|---|
| `pdflatex` pass 1 | 0 errors, minor warnings |
| `bibtex` | 0 errors (1 warning — fixed) |
| `pdflatex` pass 2 | 0 errors |
| `pdflatex` pass 3 (final) | **55 pages, 0 errors** |
