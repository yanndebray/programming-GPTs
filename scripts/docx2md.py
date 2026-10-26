"""Convert the book .docx into one Markdown file per chapter for GitHub Pages.

Usage (from the repo root, requires pandoc >= 3):
    python scripts/docx2md.py "Programming GPTs - Book - v1.2 - print.docx"
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILTER = Path(__file__).with_name("docx2md.lua")


def pandoc(args, data=None):
    return subprocess.run(["pandoc", *args], input=data, capture_output=True,
                          text=True, check=True, cwd=ROOT).stdout


def stringify(inlines):
    out = []
    for el in inlines:
        t, c = el["t"], el.get("c")
        if t == "Str":
            out.append(c)
        elif t in ("Space", "SoftBreak"):
            out.append(" ")
        elif t in ("Emph", "Strong", "Underline", "Span", "Link"):
            out.append(stringify(c[-2] if t == "Link" else c[-1] if t == "Span" else c))
    return "".join(out).strip()


def to_md(blocks, meta, api):
    doc = {"pandoc-api-version": api, "meta": meta, "blocks": blocks}
    md = pandoc(["-f", "json", "-t", "gfm", "--wrap=none"], json.dumps(doc))
    md = md.replace("](./media/", "](media/").replace("*\\*\n\n", "")
    # Keep Jekyll from interpreting Liquid tags inside code samples
    return re.sub(r"^(`{3,}).*?^\1$", lambda m: "{% raw %}\n" + m[0] + "\n{% endraw %}"
                  if "{{" in m[0] or "{%" in m[0] else m[0], md, flags=re.S | re.M)


def slug(title):
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def main(docx):
    doc = json.loads(pandoc(["-f", "docx+styles", "-t", "json",
                             "--extract-media=.", "--lua-filter", str(FILTER), docx]))
    meta, api = doc["meta"], doc["pandoc-api-version"]

    # Split on level-1 headings; everything before the first one is front matter.
    sections, current = [], {"title": None, "blocks": []}
    for blk in doc["blocks"]:
        if blk["t"] == "Header" and blk["c"][0] == 1:
            sections.append(current)
            current = {"title": stringify(blk["c"][2]), "blocks": []}
        else:
            current["blocks"].append(blk)
    sections.append(current)

    front = sections[0]
    chapters = [s for s in sections[1:] if s["title"] != "Table of Contents"]

    num = 0
    for ch in chapters:
        if ch["title"] == "Preface":
            ch["num"], ch["file"] = None, "0_Preface.md"
        else:
            num += 1
            ch["num"], ch["file"] = num, f"{num}_{slug(ch['title'])}.md"

    for i, ch in enumerate(chapters):
        n, sub = ch["num"], 0
        for blk in ch["blocks"]:
            if n and blk["t"] == "Header" and blk["c"][0] == 2:
                sub += 1
                blk["c"][2] = [{"t": "Str", "c": f"{n}.{sub}."}, {"t": "Space"}] + blk["c"][2]
        heading = f"# {n}. {ch['title']}" if n else f"# {ch['title']}"
        nav = ["[🏠 Home](README.md)"]
        if i > 0:
            nav.insert(0, f"[⬅️ {chapters[i-1]['title']}]({chapters[i-1]['file']})")
        if i + 1 < len(chapters):
            nav.append(f"[{chapters[i+1]['title']} ➡️]({chapters[i+1]['file']})")
        body = to_md(ch["blocks"], meta, api)
        page = f"{heading}\n\n{body}"
        if n is None:  # cover, praise and dedication open the book
            page = to_md(front["blocks"], meta, api) + "\n---\n\n" + page
        (ROOT / ch["file"]).write_text(f"{page}\n---\n\n{' | '.join(nav)}\n")
        print("wrote", ch["file"])

    toc = []
    for ch in chapters:
        label = f"{ch['num']}. {ch['title']}" if ch["num"] else ch["title"]
        toc.append(f"- [{label}]({ch['file']})")
    print("\n".join(toc))


if __name__ == "__main__":
    main(sys.argv[1])
