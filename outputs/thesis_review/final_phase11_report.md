# Phase 11 Final Report
## Thesis Review, Improvement, and Korean Thesis Generation

**Completed**: 2026-06-03  
**Scope**: English thesis review, formatting audit and fixes, Korean thesis generation, compilation verification

---

## I. Thesis Artifact Inventory

### English Thesis (outputs/thesis_latex/)

| Artifact | Status | Notes |
|---|---|---|
| `thesis.pdf` | ✅ Compiled | 55 pages, 1.4 MB |
| `thesis.tex` | ✅ Fixed | Formatting improvements applied |
| `abstract.tex` | ✅ Complete | ~350 words |
| `introduction.tex` | ✅ Complete | RQ1–RQ7 stated |
| `literature_review.tex` | ✅ Complete | 33 citations |
| `methodology.tex` | ✅ Complete | BM25, Dense, Hybrid, Rule, GPT-5 |
| `results.tex` | ✅ Fixed | Tables/figures embedded, column spec fixed |
| `discussion.tex` | ✅ Complete | Threats to validity included |
| `conclusion.tex` | ✅ Fixed | Unicode minus fixed |
| `references.bib` | ✅ Fixed | nakano2022webgpt entry type corrected |

### Korean Thesis (outputs/thesis_latex_korean/)

| Artifact | Status | Notes |
|---|---|---|
| `thesis_ko.pdf` | ✅ Compiled | 51 pages, 1.4 MB |
| `thesis_ko.tex` | ✅ Complete | XeLaTeX, AppleMyungjo/AppleGothic fonts |
| `abstract_ko.tex` | ✅ Complete | 647 Korean characters (within 600–800 target) |
| `introduction_ko.tex` | ✅ Complete | Korean academic style |
| `literature_review_ko.tex` | ✅ Complete | All citations preserved |
| `methodology_ko.tex` | ✅ Complete | SEKD description, all systems |
| `results_ko.tex` | ✅ Complete | All tables and figures |
| `discussion_ko.tex` | ✅ Complete | Threats to validity in Korean |
| `conclusion_ko.tex` | ✅ Complete | RQ1–RQ7 answers, future work |

### Review Artifacts (outputs/thesis_review/)

| Artifact | Status | Content |
|---|---|---|
| `thesis_review_report.md` | ✅ Complete | 6 scores, 20 issues, 10 Q&A |
| `formatting_audit.md` | ✅ Complete | 13 formatting findings |
| `formatting_changes.md` | ✅ Complete | 7 changes applied, 5 deferred |

---

## II. Quantitative Summary

| Property | English Thesis | Korean Thesis |
|---|---|---|
| **Page count** | **55 pages** | **51 pages** |
| Document class | report, 12pt, a4paper | report, 12pt, a4paper |
| Compiler | pdflatex | xelatex |
| File size | 1.4 MB | 1.4 MB |
| Chapters | 6 main + 3 appendices | 6 main + 2 appendices |
| **Figures** | **5 figures** | 5 figures (same assets) |
| **Tables** | **9 tables** (6 main + 3 appendix) | 9 tables |
| **References** | **33 citations** (BibTeX entries) | 33 citations (shared .bib) |
| Compilation status | Clean (0 errors) | Clean (0 errors, 2 font warnings†) |

† Font warnings: AppleMyungjo has no bold/italic variants; LaTeX falls back to regular weight gracefully.

---

## III. Review Scores

| Criterion | Score | Assessment |
|---|---|---|
| **Originality** | 6/10 | Enterprise-focused controlled comparison is a meaningful undergraduate contribution, though adaptive retrieval planning is a known concept |
| **Technical Depth** | 7/10 | Five-system comparison with correct implementation; lacks ColBERT/SPLADE comparison and statistical significance testing |
| **Experimental Rigor** | 4/10 | Critical issue: oracle label circularity with rule planner; n=5 temporal query pool; no significance tests |
| **Writing Quality** | 7/10 | Clear structure, academic prose, good abstract; some Discussion–Results redundancy |
| **Presentation Quality** | 7/10 | Good booktabs tables, 5 informative figures; placeholder in title page |
| **Overall Thesis Quality** | **6/10** | Competent undergraduate capstone; honest reporting of limitations; falls short of publication quality due to methodological issues |

---

## IV. Top 5 Major Weaknesses

1. **Oracle label circularity** (Critical): The SSA oracle was co-designed with the rule planner's logic (4/7 rules directly encode oracle labels), making the rule planner's SSA comparison self-referential rather than independently validated.

2. **No statistical significance testing** (Critical): Performance differences of 0.5–1.1 pp between systems are reported as findings without bootstrap CIs or permutation tests. At n=380, differences < 1 pp are very likely not statistically significant.

3. **Temporal query pool n=5** (Critical): All conclusions about temporal query failure rest on 5 queries. The "Dense fails completely on temporal queries" finding — used as a principal finding — requires much larger sample (n≥50) for statistical validity.

4. **Single embedding model** (Major): All dense retrieval conclusions are specific to BGE-small-en-v1.5 (384-dim). Larger models may substantially change dense vs. hybrid comparisons.

5. **Synthetic corpus only** (Major): No validation on real enterprise documents; findings may not generalize due to controlled linguistic properties of SEKD.

---

## V. Submission Readiness Assessment

### English Thesis
| Requirement | Status |
|---|---|
| Compiles without errors | ✅ Yes (pdflatex, clean) |
| All cross-references resolve | ✅ Yes |
| All figures render | ✅ Yes (5 figures) |
| All tables format correctly | ✅ Yes (column spec fixed) |
| Bibliography complete | ✅ Yes (33 entries) |
| RQ1–RQ7 systematically addressed | ✅ Yes |
| Threats to validity included | ✅ Yes |
| Supervisor name filled | ❌ **Placeholder — must fix** |
| Statistical significance | ⚠️ Not tested |
| Temporal conclusions caveated | ⚠️ Requires disclosure addition |

**English thesis submission readiness**: **Conditionally ready**. One blocker (supervisor name placeholder). Two disclosures recommended before defense.

### Korean Thesis
| Requirement | Status |
|---|---|
| Compiles with xelatex | ✅ Yes (51 pages) |
| Korean rendering correct | ✅ Yes (AppleMyungjo) |
| All figures render | ✅ Yes |
| All tables format correctly | ✅ Yes |
| Citations preserved | ✅ Yes (shared references.bib) |
| RQ1–RQ7 in Korean | ✅ Yes |
| Natural Korean academic style | ✅ Yes |
| Technical terms handled consistently | ✅ Yes (English+Korean dual notation) |
| Supervisor name filled | ❌ **Placeholder — must fix** |

**Korean thesis submission readiness**: **Conditionally ready** (same supervisor name blocker).

---

## VI. Remaining Manual Work Before Graduation Submission

### Must-Do (< 2 hours total)

| Task | Est. Time | Details |
|---|---|---|
| Fill supervisor name | 5 min | Replace `[Supervisor Name]` / `[지도교수명]` in both thesis.tex and thesis_ko.tex |
| Add oracle circularity disclosure | 30 min | Expand Section 3.2 (Oracle Labels) in both theses to explicitly acknowledge rule planner SSA circularity |
| Add temporal n=5 caveat | 20 min | Add footnote/sentence to all temporal conclusions: "based on n=5 queries; preliminary" |
| Recompile both PDFs | 5 min | After above changes |

### Recommended (4–6 hours)

| Task | Est. Time | Details |
|---|---|---|
| Run bootstrap confidence intervals | 2 hrs | Python post-hoc analysis of evaluation outputs; add CIs to main results table |
| Expand HR Policies discussion | 1 hr | Analyze which query types fail in HR category specifically |
| Add Fig. 6 (GPT-5 error distribution) | 1 hr | Pie/bar chart from llm_failure_analysis.json |
| Add SPLADE/ColBERT discussion | 1 hr | Explain exclusion and cite BEIR zero-shot results |
| Add corpus release plan | 20 min | GitHub link or "available upon request" statement |

### Optional Improvements

- Upgrade abstract to mention +15.8% MRR improvement with delta context
- Soften "5.6 million×" latency comparison to "several orders of magnitude"
- Clarify mock provider implementation for RQ5 discussion
- Add API model version/snapshot date for GPT-5 reproducibility

---

## VII. Korean Abstract Character Count Verification

The Korean abstract (`abstract_ko.tex`) contains approximately **647 Korean characters** across 5 structured sections:

1. 연구 배경 (Research Background) — ~120 chars
2. 연구 목적 (Research Objectives) — ~100 chars
3. 연구 방법 (Research Methods) — ~150 chars
4. 실험 결과 (Experimental Results) — ~170 chars
5. 연구 의의 (Research Significance) — ~107 chars

This falls within the 600–800 character target specified in Part D requirements.

---

## VIII. Technical Terms Consistency (Korean ↔ English)

| English Term | Korean Translation | In-text Notation |
|---|---|---|
| Retrieval-Augmented Generation | 검색 증강 생성 | RAG |
| Enterprise Knowledge Management | 기업 지식 관리 | — |
| Dense Retrieval | 밀집 검색 | Dense Retrieval |
| Hybrid Retrieval | 하이브리드 검색 | Hybrid RRF |
| Reciprocal Rank Fusion | 상호 순위 융합 | RRF |
| Strategy Selection Accuracy | 전략 선택 정확도 | SSA |
| Recall@k | Recall@k | (kept in English) |
| Mean Reciprocal Rank | 평균 역순위 | MRR |
| nDCG | 정규화 할인 누적 이득 | nDCG |
| BM25 | BM25 | (kept in English) |
| single\_hop | 단일 홉 | single\_hop |
| multi\_hop | 다중 홉 | multi\_hop |
| aggregation | 집계 | aggregation |
| comparison | 비교 | comparison |
| exception | 예외 | exception |
| temporal | 시간적 | temporal |

---

## IX. Compilation Commands Reference

**English Thesis** (from `outputs/thesis_latex/`):
```bash
pdflatex thesis.tex && bibtex thesis && pdflatex thesis.tex && pdflatex thesis.tex
```

**Korean Thesis** (from `outputs/thesis_latex_korean/`):
```bash
xelatex thesis_ko.tex && bibtex thesis_ko && xelatex thesis_ko.tex && xelatex thesis_ko.tex
```

---

## X. Phase 11 Completion Status

| Part | Task | Status |
|---|---|---|
| A | Thesis Committee Review | ✅ Complete (thesis_review_report.md) |
| B | Formatting Audit | ✅ Complete (formatting_audit.md) |
| B | Formatting Improvements Applied | ✅ Complete (formatting_changes.md) |
| C | Korean Thesis (8 chapter files) | ✅ Complete |
| D | Korean Abstract (600–800 chars) | ✅ Complete (647 chars) |
| E | Korean PDF Compilation | ✅ Complete (51 pages, xelatex) |
| F | Final Report | ✅ This document |
