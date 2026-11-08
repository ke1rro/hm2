import copy
import html
import json
import logging
import sys
from pathlib import Path

import nbformat
import plotly.graph_objects as go
import pymupdf
from bs4 import BeautifulSoup
from nbconvert.filters import markdown2html_mistune
from playwright.sync_api import sync_playwright
from plotly.offline import get_plotlyjs
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger(__name__)
ROOT = Path.cwd()
OUTPUT = ROOT / "output/pdf"
WORK = ROOT / "tmp/pdfs"
OUTPUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

CSS = """
@page {size: A4 landscape; margin: 12mm 12mm 15mm;}
@page tall {size: A4 portrait; margin: 12mm 12mm 15mm;}
html,body {margin:0; padding:0; background:white; color:#17202b;}
body {font-family:Arial,sans-serif; font-size:12px; line-height:1.45;}
.input {background:#f6f8fa; border:1px solid #dce2e8; border-radius:3px;
    padding:8px; margin:10px 0;}
pre {font-family:Menlo,monospace; font-size:10.5px; line-height:1.4;
    white-space:pre-wrap; overflow-wrap:anywhere; margin:0; orphans:4; widows:4;}
h1 {font-size:25px; color:#182c45;}
h2 {font-size:19px; color:#182c45; margin-top:20px;}
h1,h2,h3 {break-after:avoid-page;}
.anchor-link {display:none;}
p {orphans:3; widows:3; margin:9px 0;}
.output {margin:8px 0 14px;}
.figure {break-inside:avoid-page; margin:12px 0;}
.figure img {max-width:100%; max-height:135mm; width:auto; height:auto;
    object-fit:contain; display:block; margin:0 auto;}
table {font-size:10.5px; border-collapse:collapse; max-width:100%; margin:8px 0;}
th,td {padding:5px 7px; border:1px solid #dce2e8; color:#17202b;
    background:white; max-width:220px; overflow-wrap:anywhere;}
th {background:#eef2f6; font-weight:600;}
thead {display:table-header-group;}
tr {break-inside:avoid-page;}
.chart-static {width:100%; break-inside:avoid-page;}
.chart-static img {width:100%; max-height:145mm;}
.tall-chart {page:tall; break-before:page; break-after:page;}
.tall-chart img {width:100%; max-height:245mm;}
a {color:#165787!important; text-decoration:none!important;}
"""


def _render_output(data: dict) -> str:
    if "text/html" in data:
        fragment = BeautifulSoup(data["text/html"], "html.parser")
        for style in fragment.find_all("style"):
            style.decompose()
        return str(fragment)
    for mime in ("image/jpeg", "image/png"):
        if mime in data:
            return f'<img src="data:{mime};base64,{data[mime]}">'
    if "text/markdown" in data:
        return markdown2html_mistune(data["text/markdown"])
    return f'<pre>{html.escape(data.get("text/plain", ""))}</pre>'

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="chrome", headless=True)
    for source in sorted((ROOT / "notebooks").glob("*.ipynb")):
        original = nbformat.read(source, as_version=4)
        notebook = copy.deepcopy(original)
        figures = []
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            filtered_outputs = []
            for output in cell.outputs:
                data = output.get("data", {})
                if "application/vnd.jupyter.widget-view+json" in data:
                    continue
                if "application/vnd.plotly.v1+json" in data:
                    figure = go.Figure(data["application/vnd.plotly.v1+json"])
                    height = int(figure.layout.height or 440)
                    figure.update_layout(
                        template="plotly_white", width=980, height=height,
                        font={"family": "Arial", "size": 14},
                        margin={"l": 240 if height > 700 else 85, "r": 45, "t": 70, "b": max(140, int(figure.layout.margin.b or 0))},
                    )
                    identifier = f"chart-{len(figures)}"
                    figures.append({"id": identifier, "figure": json.loads(figure.to_json()),
                                    "height": height})
                    wrapper = "chart-static tall-chart" if height > 700 else "chart-static"
                    output.data = {"text/html": f'<div class="{wrapper}"><div id="{identifier}"></div></div>'}
                filtered_outputs.append(output)
            cell.outputs = filtered_outputs
        fragments = []
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                fragments.append(markdown2html_mistune(cell.source))
                continue
            if cell.cell_type != "code":
                continue
            fragments.append('<div class="input">' + highlight(cell.source, PythonLexer(),
                             HtmlFormatter(nowrap=False)) + '</div>')
            index = 0
            while index < len(cell.outputs):
                output = cell.outputs[index]
                data = output.get("data", {})
                if output.output_type == "stream":
                    fragments.append('<div class="output"><pre>' + html.escape(output.text) + '</pre></div>')
                elif "text/markdown" in data and index + 1 < len(cell.outputs) and any(
                    key in cell.outputs[index + 1].get("data", {}) for key in ("image/jpeg", "image/png")
                ):
                    fragments.append('<div class="figure">' + _render_output(data) +
                                     _render_output(cell.outputs[index + 1].data) + '</div>')
                    index += 1
                elif any(key in data for key in ("image/jpeg", "image/png")):
                    fragments.append('<div class="figure">' + _render_output(data) + '</div>')
                elif "chart-static" in data.get("text/html", ""):
                    fragments.append(_render_output(data))
                elif data:
                    fragments.append('<div class="output">' + _render_output(data) + '</div>')
                index += 1
        body = '<!doctype html><html><head><meta charset="utf-8"></head><body>' + ''.join(fragments) + '</body></html>'
        soup = BeautifulSoup(body, "html.parser")
        style = soup.new_tag("style")
        style.string = CSS + HtmlFormatter().get_style_defs(".highlight")
        soup.head.append(style)
        if figures:
            library = soup.new_tag("script")
            library.string = get_plotlyjs()
            soup.head.append(library)
        script = soup.new_tag("script")
        script.string = """
        window.__pdfReady = false;
        window.__pdfError = null;
        (async () => {
          try {
            for (const item of FIGURES) {
              const chart = document.getElementById(item.id);
              await Plotly.newPlot(chart, item.figure.data, item.figure.layout,
                                   {staticPlot: true, displayModeBar: false});
              const image = new Image();
              image.src = await Plotly.toImage(chart, {format: 'svg', width: 980, height: item.height});
              await image.decode();
              chart.replaceWith(image);
            }
            await document.fonts.ready;
            await Promise.all(Array.from(document.images).map(image => image.decode()));
            window.__pdfReady = true;
          } catch (error) { window.__pdfError = String(error); }
        })();
        """.replace("FIGURES", json.dumps(figures))
        soup.body.append(script)
        html_path = WORK / f"{source.stem}.html"
        html_path.write_text(str(soup))
        page = browser.new_page(viewport={"width": 1100, "height": 850})
        page.goto(html_path.as_uri(), wait_until="load")
        page.wait_for_function("window.__pdfReady || window.__pdfError", timeout=120000)
        error = page.evaluate("window.__pdfError")
        if error:
            raise RuntimeError(error)
        pdf_path = OUTPUT / f"{source.stem}.pdf"
        page.pdf(
            path=str(pdf_path), format="A4", landscape=True, print_background=True,
            prefer_css_page_size=True, display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="width:100%;padding:0 12mm;font:8px Arial;color:#606a78;'
                'display:flex;justify-content:space-between">'
                f'<span>{source.name}</span>'
                '<span><span class="pageNumber"></span> / '
                '<span class="totalPages"></span></span></div>'
            ),
        )
        page.close()
        with pymupdf.open(pdf_path) as document:
            compact = document.tobytes(garbage=4, deflate=True)
        if len(compact) < pdf_path.stat().st_size:
            pdf_path.write_bytes(compact)
        with pymupdf.open(pdf_path) as document:
            for index, pdf_page in enumerate(document):
                if not pdf_page.get_text().strip():
                    raise ValueError(f"Empty PDF page: {source.name} / {index + 1}")
            LOGGER.info("Exported %s: %s pages, %s rendered Plotly charts, %.2f MiB",
                        pdf_path.name, len(document), len(figures), pdf_path.stat().st_size / 1024**2)
    browser.close()
