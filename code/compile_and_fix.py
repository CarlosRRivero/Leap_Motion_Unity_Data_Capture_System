"""
compile_and_fix.py

Tests LaTeX hyperlink fixes using a minimal article-class document
(avoids wlpeerj.cls dependency), then applies the winning fix to paper.ltx.

The fix for starred sections is verified by checking:
  1. No '??' in the PDF (unresolved refs)
  2. At least one hyperlink annotation in the PDF

Usage:  python compile_and_fix.py
"""

import os, sys, re, shutil, subprocess, urllib.request, zipfile

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_SRC   = os.path.join(SCRIPTS_DIR, "paper.ltx")
OUT_DIR     = os.path.join(SCRIPTS_DIR, "_build")
TECTONIC    = os.path.join(SCRIPTS_DIR, "_tools", "tectonic.exe")
TEST_SRC    = os.path.join(OUT_DIR, "test_refs.tex")

os.makedirs(OUT_DIR,                         exist_ok=True)
os.makedirs(os.path.dirname(TECTONIC),       exist_ok=True)

TARGET_LABELS = {
    "subsec:sysarch":        "System architecture",
    "subsec:sus":            "System Usability Scale",
    "subsec:exercises":      "Explanation of Exercises",
    "subsec:hardware_setup": "Hardware Setup",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Tectonic
# ═══════════════════════════════════════════════════════════════════════════════

TECTONIC_URL = (
    "https://github.com/tectonic-typesetting/tectonic/releases/download/"
    "tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip"
)

def ensure_tectonic():
    if os.path.exists(TECTONIC):
        print(f"[tectonic] Found.")
        return
    print("[tectonic] Downloading...")
    tmp = TECTONIC + ".zip"
    urllib.request.urlretrieve(TECTONIC_URL, tmp)
    with zipfile.ZipFile(tmp) as zf:
        for m in zf.namelist():
            if m.endswith("tectonic.exe"):
                with zf.open(m) as s, open(TECTONIC, "wb") as d:
                    shutil.copyfileobj(s, d)
                break
    os.remove(tmp)
    print(f"[tectonic] Ready.")

def ensure_pymupdf():
    try:
        import fitz; return
    except ImportError:
        print("[pymupdf] Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "pymupdf", "--quiet"])

def compile_tex(src):
    result = subprocess.run(
        [TECTONIC, "--outdir", OUT_DIR, src],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode == 0, result.stdout + result.stderr

def check_pdf(pdf_path):
    """Returns (has_question_marks, link_count)."""
    import fitz
    doc = fitz.open(pdf_path)
    has_qm, links = False, 0
    for page in doc:
        if "??" in page.get_text():
            has_qm = True
        links += len(page.get_links())
    doc.close()
    return has_qm, links


# ═══════════════════════════════════════════════════════════════════════════════
#  Minimal test document builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_test_doc(hyperref_options, ref_style, extra_packages=""):
    """
    ref_style: "ref" | "nameref" | "hyperlink"
    """
    if ref_style == "ref":
        ref1 = r"\ref{subsec:sysarch}"
        ref2 = r"\ref{subsec:sus}"
    elif ref_style == "nameref":
        ref1 = r"\nameref{subsec:sysarch}"
        ref2 = r"\nameref{subsec:sus}"
    else:  # hyperlink
        ref1 = r"\hyperlink{subsec:sysarch}{System architecture}"
        ref2 = r"\hyperlink{subsec:sus}{System Usability Scale}"

    return rf"""
\documentclass{{article}}
\usepackage[{hyperref_options}]{{hyperref}}
{extra_packages}
\begin{{document}}

\section*{{Introduction}}
See the {ref1} subsection and {ref2} subsection.

\phantomsection
\subsection*{{System architecture}}
\label{{subsec:sysarch}}
Content here.

\phantomsection
\subsection*{{System Usability Scale}}
\label{{subsec:sus}}
Content here.

\end{{document}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategies to test on minimal doc
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = [
    {
        "name":          "colorlinks + \\nameref",
        "hyperref_opts": "colorlinks=true,linkcolor=blue",
        "ref_style":     "nameref",
        "extra_pkg":     "",
    },
    {
        "name":          "colorlinks + \\nameref + explicit nameref pkg",
        "hyperref_opts": "colorlinks=true,linkcolor=blue",
        "ref_style":     "nameref",
        "extra_pkg":     r"\usepackage{nameref}",
    },
    {
        "name":          "colorlinks + \\hyperlink (hypertarget style)",
        "hyperref_opts": "colorlinks=true,linkcolor=blue",
        "ref_style":     "hyperlink",
        "extra_pkg":     "",
    },
    {
        "name":          "hidelinks + \\nameref",
        "hyperref_opts": "hidelinks",
        "ref_style":     "nameref",
        "extra_pkg":     "",
    },
    {
        "name":          "default hyperref + \\nameref",
        "hyperref_opts": "",
        "ref_style":     "nameref",
        "extra_pkg":     "",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Apply winning strategy to paper.ltx
# ═══════════════════════════════════════════════════════════════════════════════

def patch_paper(strategy):
    with open(PAPER_SRC, "r", encoding="utf-8") as f:
        src = f.read()

    # ── 1. Fix hyperref options ────────────────────────────────────────────────
    opts = strategy["hyperref_opts"]
    if opts:
        # Replace any existing \usepackage[...]{hyperref} or \usepackage{hyperref}
        src = re.sub(
            r"\\usepackage(?:\[[^\]]*\])?\{hyperref\}",
            r"\\usepackage[" + opts + r"]{hyperref}",
            src, count=1
        )
    # Add extra package after hyperref if needed
    if strategy["extra_pkg"] and strategy["extra_pkg"] not in src:
        src = re.sub(
            r"(\\usepackage(?:\[[^\]]*\])?\{hyperref\})",
            r"\1\n" + strategy["extra_pkg"],
            src, count=1
        )

    # ── 2. Ensure \phantomsection before every starred section with a label ───
    for label, title in TARGET_LABELS.items():
        # Case A: \phantomsection already on its own line before \subsection*
        # Case B: missing — add it
        # Normalize: remove any existing \phantomsection before this heading,
        # then re-insert cleanly
        src = re.sub(
            r"(\\phantomsection\s*\n)?(\\(?:sub)*section\*\{" + re.escape(title) + r"\}[^\n]*\n)(\\label\{" + re.escape(label) + r"\})",
            r"\\phantomsection\n\2\3",
            src
        )
        # Case: label on same line as section*
        src = re.sub(
            r"(\\phantomsection\s*\n)?(\\(?:sub)*section\*\{" + re.escape(title) + r"\})\\label\{" + re.escape(label) + r"\}",
            r"\\phantomsection\n\2\n\\label{" + label + r"}",
            src
        )

    # ── 3. Fix references ──────────────────────────────────────────────────────
    ref_style = strategy["ref_style"]
    for label, title in TARGET_LABELS.items():
        if ref_style == "nameref":
            # Replace \ref{} and ensure \nameref{}
            src = src.replace(r"\ref{" + label + "}", r"\nameref{" + label + "}")
            # Already-converted ones stay as \nameref
        elif ref_style == "hyperlink":
            # Replace both \ref and \nameref with \hyperlink
            src = src.replace(r"\ref{" + label + "}",     r"\hyperlink{" + label + "}{" + title + "}")
            src = src.replace(r"\nameref{" + label + "}", r"\hyperlink{" + label + "}{" + title + "}")
            # And replace \phantomsection+\label with \hypertarget
            src = re.sub(
                r"\\phantomsection\s*\n(\\(?:sub)*section\*\{" + re.escape(title) + r"\}[^\n]*)\n\\label\{" + re.escape(label) + r"\}",
                r"\\hypertarget{" + label + r"}{}\n\1",
                src
            )

    return src


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_tectonic()
    ensure_pymupdf()

    # Backup
    backup = PAPER_SRC + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(PAPER_SRC, backup)
        print(f"[backup] Saved to {backup}")

    print("\n" + "="*60)
    print("  Phase 1: Test fix strategies on minimal article document")
    print("="*60)

    winning = None

    for i, strat in enumerate(STRATEGIES):
        name = strat["name"]
        print(f"\n[{i+1}/{len(STRATEGIES)}] {name}")

        doc = build_test_doc(strat["hyperref_opts"], strat["ref_style"], strat["extra_pkg"])
        with open(TEST_SRC, "w", encoding="utf-8") as f:
            f.write(doc)

        ok, log = compile_tex(TEST_SRC)
        if not ok:
            print(f"  Compile FAILED")
            for line in log.splitlines()[-8:]:
                if "error" in line.lower() or "warning" in line.lower():
                    print(f"    {line.strip()}")
            continue

        pdf = os.path.join(OUT_DIR, "test_refs.pdf")
        if not os.path.exists(pdf):
            print("  No PDF produced.")
            continue

        has_qm, link_count = check_pdf(pdf)
        print(f"  ??: {has_qm}  |  links: {link_count}")

        if not has_qm and link_count > 0:
            print(f"  -> PASS")
            winning = strat
            break
        else:
            print(f"  -> FAIL")

    if winning is None:
        print("\n[ERROR] No strategy passed the minimal test.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Winning strategy: {winning['name']}")
    print(f"{'='*60}")

    print("\n  Phase 2: Applying fix to paper.ltx ...")
    patched = patch_paper(winning)
    with open(PAPER_SRC, "w", encoding="utf-8") as f:
        f.write(patched)
    print("  paper.ltx updated.")

    # Show diff summary
    with open(backup, "r", encoding="utf-8") as f:
        orig = f.read()

    orig_lines  = set(orig.splitlines())
    patch_lines = set(patched.splitlines())
    added   = [l for l in patch_lines - orig_lines if l.strip()]
    removed = [l for l in orig_lines  - patch_lines if l.strip()]
    print("\n  Changes made to paper.ltx:")
    for l in sorted(removed)[:10]: print(f"    - {l.strip()}")
    for l in sorted(added)[:10]:   print(f"    + {l.strip()}")

    print(f"""
  Done. Upload paper.ltx to Overleaf (or your LaTeX editor) and recompile.
  The fix uses: {winning['name']}
""")


if __name__ == "__main__":
    main()
