"""
Office Math Markup Language (OMML) to LaTeX converter.

Converts OpenXML mathematical markup (<m:oMath>, <m:oMathPara>) into standard
LaTeX expressions suitable for academic analysis, RAG retrieval, and frontend display.
"""

from xml.etree import ElementTree as ET

OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_NS = {"m": OMML_NS}

# Common mathematical symbol translations to LaTeX commands
_CHAR_TO_LATEX: dict[str, str] = {
    "±": r"\pm ",
    "∓": r"\mp ",
    "×": r"\times ",
    "÷": r"\div ",
    "·": r"\cdot ",
    "°": r"^\circ ",
    "≠": r"\neq ",
    "≤": r"\leq ",
    "≥": r"\geq ",
    "≈": r"\approx ",
    "≡": r"\equiv ",
    "∝": r"\propto ",
    "∞": r"\infty ",
    "∂": r"\partial ",
    "∇": r"\nabla ",
    "√": r"\sqrt",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∬": r"\iint",
    "∭": r"\iiint",
    "∮": r"\oint",
    "∈": r"\in ",
    "∉": r"\notin ",
    "⊂": r"\subset ",
    "⊆": r"\subseteq ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "∧": r"\land ",
    "∨": r"\lor ",
    "¬": r"\neg ",
    "⇒": r"\Rightarrow ",
    "⇔": r"\Leftrightarrow ",
    "→": r"\rightarrow ",
    "←": r"\leftarrow ",
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "ε": r"\epsilon ",
    "ζ": r"\zeta ",
    "η": r"\eta ",
    "θ": r"\theta ",
    "ι": r"\iota ",
    "κ": r"\kappa ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "ν": r"\nu ",
    "ξ": r"\xi ",
    "π": r"\pi ",
    "ρ": r"\rho ",
    "σ": r"\sigma ",
    "τ": r"\tau ",
    "υ": r"\upsilon ",
    "φ": r"\phi ",
    "χ": r"\chi ",
    "ψ": r"\psi ",
    "ω": r"\omega ",
    "Γ": r"\Gamma ",
    "Δ": r"\Delta ",
    "Θ": r"\Theta ",
    "Λ": r"\Lambda ",
    "Ξ": r"\Xi ",
    "Π": r"\Pi ",
    "Σ": r"\Sigma ",
    "Υ": r"\Upsilon ",
    "Φ": r"\Phi ",
    "Ψ": r"\Psi ",
    "Ω": r"\Omega ",
}


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def omml_to_latex(element: ET.Element | object) -> str:
    """
    Recursively converts an OMML XML element into a LaTeX string.

    Supports fractions, superscripts, subscripts, sub-superscripts, radicals,
    n-ary operators (integrals, sums, products), delimiters, and math runs.
    """
    # Accept lxml or xml.etree elements
    if hasattr(element, "_element"):
        element = element._element

    raw_element: ET.Element = element  # type: ignore[assignment]
    tag = _strip_ns(raw_element.tag)

    if tag in ("oMathPara", "oMath"):
        inner_parts = [omml_to_latex(child) for child in raw_element]
        return "".join(inner_parts).strip()

    if tag == "r":
        # Math run
        parts: list[str] = []
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "t":
                text = child.text or ""
                for char, latex in _CHAR_TO_LATEX.items():
                    text = text.replace(char, latex)
                parts.append(text)
            else:
                parts.append(omml_to_latex(child))
        return "".join(parts)

    if tag == "t":
        text = raw_element.text or ""
        for char, latex in _CHAR_TO_LATEX.items():
            text = text.replace(char, latex)
        return text

    if tag == "f":
        # Fraction <m:f> -> <m:num> and <m:den>
        num = ""
        den = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "num":
                num = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "den":
                den = "".join(omml_to_latex(c) for c in child).strip()
        return rf"\frac{{{num}}}{{{den}}}"

    if tag == "sSup":
        # Superscript <m:sSup> -> <m:e> (base) and <m:sup> (exponent)
        base = ""
        sup = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "e":
                base = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "sup":
                sup = "".join(omml_to_latex(c) for c in child).strip()
        return f"{{{base}}}^{{{sup}}}"

    if tag == "sSub":
        # Subscript <m:sSub> -> <m:e> (base) and <m:sub> (subscript)
        base = ""
        sub = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "e":
                base = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "sub":
                sub = "".join(omml_to_latex(c) for c in child).strip()
        return f"{{{base}}}_{{{sub}}}"

    if tag == "sSubSup":
        # Sub-superscript <m:sSubSup> -> <m:e>, <m:sub>, <m:sup>
        base = ""
        sub = ""
        sup = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "e":
                base = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "sub":
                sub = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "sup":
                sup = "".join(omml_to_latex(c) for c in child).strip()
        return f"{{{base}}}_{{{sub}}}^{{{sup}}}"

    if tag == "rad":
        # Radical <m:rad> -> <m:deg> and <m:e>
        deg = ""
        base = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "deg":
                deg = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "e":
                base = "".join(omml_to_latex(c) for c in child).strip()
        if deg:
            return rf"\sqrt[{deg}]{{{base}}}"
        return rf"\sqrt{{{base}}}"

    if tag == "nary":
        # N-ary operator <m:nary> -> <m:naryPr> (operator char), <m:sub>, <m:sup>, <m:e>
        operator = r"\int "
        sub = ""
        sup = ""
        base = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "naryPr":
                for pr_child in child:
                    if _strip_ns(pr_child.tag) == "chr":
                        val = pr_child.attrib.get(f"{{{OMML_NS}}}val") or pr_child.attrib.get("val")
                        if val:
                            operator = _CHAR_TO_LATEX.get(val, val) + " "
            elif child_tag == "sub":
                sub = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "sup":
                sup = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "e":
                base = "".join(omml_to_latex(c) for c in child).strip()

        bounds = ""
        if sub and sup:
            bounds = f"_{{{sub}}}^{{{sup}}}"
        elif sub:
            bounds = f"_{{{sub}}}"
        elif sup:
            bounds = f"^{{{sup}}}"
        return f"{operator.strip()}{bounds} {{{base}}}"

    if tag == "d":
        # Delimiters <m:d> -> <m:dPr> (begChr, endChr), <m:e>
        beg_chr = "("
        end_chr = ")"
        content = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "dPr":
                for pr_child in child:
                    pr_tag = _strip_ns(pr_child.tag)
                    val = pr_child.attrib.get(f"{{{OMML_NS}}}val") or pr_child.attrib.get("val")
                    if pr_tag == "begChr" and val is not None:
                        beg_chr = val
                    elif pr_tag == "endChr" and val is not None:
                        end_chr = val
            elif child_tag == "e":
                content = "".join(omml_to_latex(c) for c in child).strip()

        # Escape brackets where needed in LaTeX
        beg_latex = r"\{" if beg_chr == "{" else beg_chr
        end_latex = r"\}" if end_chr == "}" else end_chr
        if beg_latex or end_latex:
            return rf"\left{beg_latex} {content} \right{end_latex}"
        return content

    if tag == "func":
        # Function <m:func> -> <m:fName> and <m:e>
        fname = ""
        base = ""
        for child in raw_element:
            child_tag = _strip_ns(child.tag)
            if child_tag == "fName":
                fname = "".join(omml_to_latex(c) for c in child).strip()
            elif child_tag == "e":
                base = "".join(omml_to_latex(c) for c in child).strip()
        return f"{fname} {{{base}}}"

    if tag == "m":
        # Matrix <m:m> -> <m:mr> (rows) -> <m:e> (cells)
        rows: list[str] = []
        for child in raw_element:
            if _strip_ns(child.tag) == "mr":
                cells = [
                    "".join(omml_to_latex(c) for c in mr_child).strip()
                    for mr_child in child
                    if _strip_ns(mr_child.tag) == "e"
                ]
                rows.append(" & ".join(cells))
        matrix_body = r" \\ ".join(rows)
        return rf"\begin{{matrix}} {matrix_body} \end{{matrix}}"

    # Generic element fallback: collect children
    return "".join(omml_to_latex(child) for child in raw_element)


def convert_omml_to_latex_math(element: ET.Element | object, *, is_block: bool = False) -> str:
    """
    Converts an OMML element to a LaTeX expression enclosed in math delimiters.

    Returns:
        '$<latex>$' for inline math, or '$$<latex>$$' for block equations.
    """
    latex = omml_to_latex(element).strip()
    if not latex:
        return ""
    if is_block:
        return f"\n\n$${latex}$$\n\n"
    return f"${latex}$"

