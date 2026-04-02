"""
ingest.py — Crawl website → markdown → parent+child chunks → Pinecone.

Chunking strategy:
  Every page produces:
    1 parent chunk  — full page content, type="parent"
    N child chunks  — one per H2 section,  type="child", carries parent_id

  Pages under CHILD_SPLIT_THRESHOLD words → parent only (not enough to split).
  H2 sections under MIN_SECTION_WORDS     → merged into previous child.

  At query time:  search child chunks for precision
                  return parent content to LLM for full context

Usage:
    python ingest.py --url https://docs.example.com
    python ingest.py --url https://docs.example.com --max-pages 50
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlparse, urljoin

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import NoExtractionStrategy
from bs4 import BeautifulSoup
from markdownify import markdownify as md_convert
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_API_VERSION, AZURE_EMBEDDING_DEPLOYMENT_NAME,
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION,
    MAX_PAGES, MAX_DEPTH, CRAWL_CONCURRENCY,
    MIN_PAGE_WORDS, MAX_PAGE_WORDS, MIN_SECTION_WORDS, CHILD_SPLIT_THRESHOLD,
    PAGES_DIR,
)

# ── JS: open tabs + accordions before snapshot ────────────────────────────────
_TAB_JS = """
(async () => {
    const delay = ms => new Promise(r => setTimeout(r, ms));
    const seen = new Set();
    for (const sel of ['[role="tab"]','[data-tab]','.tab','[aria-selected]']) {
        for (const el of document.querySelectorAll(sel)) {
            if (!seen.has(el)) { seen.add(el); try { el.click(); await delay(200); } catch(e) {} }
        }
    }
    for (const el of document.querySelectorAll('[role="tabpanel"],[data-panel],.tab-panel')) {
        el.style.display='block'; el.style.visibility='visible';
        el.removeAttribute('hidden'); el.removeAttribute('aria-hidden');
    }
    for (const el of document.querySelectorAll('details:not([open])')) {
        try { el.open = true; } catch(e) {}
    }
    await delay(500);
})();
"""


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — CRAWLER
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_url(url: str) -> str:
    return urlparse(url)._replace(fragment="").geturl().rstrip("/")

def _base_domain(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"

def _is_allowed(url: str, start_url: str, base_domain: str) -> bool:
    if not url.startswith(base_domain):
        return False
    start_path  = urlparse(start_url).path.rstrip("/")
    target_path = urlparse(url).path
    if start_path and not target_path.startswith(start_path):
        return False
    skip = {".pdf",".zip",".png",".jpg",".jpeg",".gif",".svg",
            ".webp",".ico",".mp4",".mp3",".css",".js"}
    if any(urlparse(url).path.lower().endswith(e) for e in skip):
        return False
    return True

def _html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in [
        "nav","header","footer","aside",
        "[role='navigation']","[role='banner']","[role='contentinfo']",
        ".sidebar",".navbar",".toc",".breadcrumb",
        "#sidebar","#nav","#header","#footer",
        ".on-this-page","[aria-label='On this page']",".pagination",
        "starlight-menu-button",".sl-sidebar","[data-pagefind-ignore]",
    ]:
        for tag in soup.select(sel):
            tag.decompose()

    main = (
        soup.find(class_="sl-markdown-content") or
        soup.find(class_="content-panel")        or
        soup.find(class_="markdown-body")        or
        soup.find(class_="prose")                or
        soup.find(id="content")                  or
        soup.find("main")                        or
        soup.find(attrs={"role": "main"})        or
        soup.body or soup
    )
    if len(main.get_text().strip()) < 100 and soup.body:
        main = soup.body

    for tag in main.find_all(["script","style","svg","button","noscript"]):
        tag.decompose()

    md = md_convert(str(main), heading_style="ATX", bullets="-",
                    strip=["script","style","svg","button","noscript","img"])
    lines = [l for l in md.splitlines()
             if not re.match(r'^\s*\[([^\]]{1,80})\]\([^\)]+\)\s*$', l.strip())]
    md = re.sub(r'\n{3,}', '\n\n', "\n".join(lines))
    return md.strip()

async def crawl_site(start_url: str, max_pages: int, max_depth: int) -> List[Dict]:
    start_url   = start_url.rstrip("/")
    base_domain = _base_domain(start_url)
    visited     = set()
    results     = []
    queue       = asyncio.Queue()
    await queue.put((start_url, 0))

    browser_cfg = BrowserConfig(
        headless=True, verbose=False,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        extra_args=["--disable-blink-features=AutomationControlled","--no-sandbox"],
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=NoExtractionStrategy(),
        word_count_threshold=10,
        remove_overlay_elements=True,
        exclude_external_links=True,
        js_code=_TAB_JS,
        wait_for="css:body",
        page_timeout=30000,
    )
    semaphore = asyncio.Semaphore(CRAWL_CONCURRENCY)

    async def crawl_one(crawler, url, depth):
        async with semaphore:
            try:
                result   = await crawler.arun(url=url, config=run_cfg)
                if not result.success:
                    return None
                title    = result.metadata.get("title", "") or url
                markdown = _html_to_markdown(result.html or "")
                if len(markdown.split()) < MIN_PAGE_WORDS:
                    md_obj   = result.markdown
                    fallback = getattr(md_obj, "raw_markdown", None) or str(md_obj) or ""
                    if fallback:
                        markdown = re.sub(r'\n{3,}', '\n\n', fallback).strip()
                links = [lnk.get("href","") for lnk in
                         (result.links or {}).get("internal", []) if lnk.get("href")]
                return {"url": url, "title": title, "markdown": markdown, "links": links}
            except Exception as e:
                print(f"  ❌ {url} — {e}")
                return None

    print(f"\n{'='*55}")
    print(f"🌐 Crawling: {start_url}")
    print(f"{'='*55}")

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        while not queue.empty() and len(results) < max_pages:
            batch = []
            while not queue.empty() and len(batch) < CRAWL_CONCURRENCY:
                url, depth = await queue.get()
                norm = _normalize_url(url)
                if norm in visited or not _is_allowed(norm, start_url, base_domain):
                    continue
                visited.add(norm)
                batch.append((norm, depth))
            if not batch:
                break

            outputs = await asyncio.gather(
                *[crawl_one(crawler, u, d) for u, d in batch],
                return_exceptions=True,
            )
            for (url, depth), out in zip(batch, outputs):
                if isinstance(out, Exception) or out is None:
                    continue
                words = len(out["markdown"].split())
                if words < MIN_PAGE_WORDS:
                    print(f"  ⏭️  Stub ({words}w): {url}")
                    continue
                results.append(out)
                print(f"  ✅ [{len(results):>3}] {words:>5}w  {out['title'][:50]}")
                if depth < max_depth:
                    for link in out["links"]:
                        full = urljoin(url, link)
                        norm_link = _normalize_url(full)
                        if norm_link not in visited and _is_allowed(norm_link, start_url, base_domain):
                            await queue.put((norm_link, depth + 1))

    print(f"\n✅ Crawled {len(results)} pages")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — SAVE MARKDOWN LOCALLY
# ═════════════════════════════════════════════════════════════════════════════

def _url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    raw    = f"{parsed.netloc}{parsed.path}"
    slug   = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return (slug or "page") + ".md"

def _url_to_id(url: str) -> str:
    parsed = urlparse(url)
    raw    = f"{parsed.netloc}{parsed.path}"
    vid    = re.sub(r"[^a-z0-9\-]", "-", raw.lower())
    return re.sub(r"-+", "-", vid).strip("-") or "page"

def save_pages_locally(pages: List[Dict], site_slug: str) -> Path:
    site_dir = PAGES_DIR / site_slug
    site_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for page in pages:
        fname   = _url_to_filename(page["url"])
        content = (
            f"---\nurl: {page['url']}\ntitle: {page['title']}\n"
            f"words: {len(page['markdown'].split())}\n---\n\n{page['markdown']}"
        )
        (site_dir / fname).write_text(content, encoding="utf-8")
        manifest.append({"file": fname, "url": page["url"], "title": page["title"],
                         "words": len(page["markdown"].split())})
    (site_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"💾 Saved {len(pages)} markdown files → {site_dir}/")
    return site_dir


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — PARENT + CHILD CHUNKING
# ═════════════════════════════════════════════════════════════════════════════

def _split_by_h2(markdown: str) -> List[Tuple[str, str]]:
    """
    Split markdown by H2 headings.
    Returns list of (section_title, section_content) tuples.
    The first tuple is content before any H2 (intro), titled "Introduction".
    """
    # Split on ## lines (not ### or deeper)
    parts   = re.split(r'^(## .+)$', markdown, flags=re.MULTILINE)
    sections: List[Tuple[str, str]] = []

    # parts = [before_first_h2, "## Title1", content1, "## Title2", content2, ...]
    intro = parts[0].strip()
    if intro:
        sections.append(("Introduction", intro))

    i = 1
    while i < len(parts) - 1:
        heading = parts[i].lstrip("#").strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            sections.append((heading, f"## {heading}\n\n{content}"))
        i += 2

    return sections

def make_chunks(pages: List[Dict]) -> List[Dict]:
    """
    For each page produce:
      - 1 parent chunk  (full page, type="parent")
      - N child chunks  (one per H2 section, type="child")
        only when page words >= CHILD_SPLIT_THRESHOLD

    Returns flat list of all chunks ready for embedding.
    """
    all_chunks    = []
    total_parents = 0
    total_children= 0
    skipped       = 0

    for page in pages:
        markdown = page["markdown"].strip()
        if not markdown:
            skipped += 1
            continue

        url      = page["url"]
        title    = page["title"]
        words    = len(markdown.split())
        page_id  = _url_to_id(url)

        if words < MIN_PAGE_WORDS:
            skipped += 1
            continue
        if words > MAX_PAGE_WORDS:
            print(f"  ⚠️  Large page ({words}w): {title}")

        # ── Parent chunk ──────────────────────────────────────────────────────
        parent_content = (
            f"Page: {title}\n"
            f"URL: {url}\n\n"
            f"{markdown}"
        )
        all_chunks.append({
            "chunk_id":      page_id,
            "parent_id":     None,          # parent has no parent
            "type":          "parent",
            "url":           url,
            "title":         title,
            "section_title": "",
            "heading_path":  title,
            "content":       parent_content,
            "word_count":    words,
        })
        total_parents += 1

        # ── Child chunks (only for pages long enough to split) ────────────────
        if words < CHILD_SPLIT_THRESHOLD:
            continue

        sections = _split_by_h2(markdown)
        if len(sections) <= 1:
            # No H2 headings — no point making children
            continue

        # Merge tiny sections into the previous one
        merged: List[Tuple[str, str]] = []
        for sec_title, sec_content in sections:
            sec_words = len(sec_content.split())
            if sec_words < MIN_SECTION_WORDS and merged:
                # Append to previous
                prev_title, prev_content = merged[-1]
                merged[-1] = (prev_title, prev_content + "\n\n" + sec_content)
            else:
                merged.append((sec_title, sec_content))

        for idx, (sec_title, sec_content) in enumerate(merged):
            child_id = f"{page_id}-s{idx}"
            child_content = (
                f"Page: {title}\n"
                f"URL: {url}\n"
                f"Section: {sec_title}\n"
                f"Heading path: {title} > {sec_title}\n\n"
                f"{sec_content}"
            )
            all_chunks.append({
                "chunk_id":      child_id,
                "parent_id":     page_id,
                "type":          "child",
                "url":           url,
                "title":         title,
                "section_title": sec_title,
                "heading_path":  f"{title} > {sec_title}",
                "content":       child_content,
                "word_count":    len(sec_content.split()),
            })
            total_children += 1

    print(
        f"✂️  Chunks: {total_parents} parents + {total_children} children "
        f"= {len(all_chunks)} total | {skipped} pages skipped"
    )
    return all_chunks


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — EMBED
# ═════════════════════════════════════════════════════════════════════════════

def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

def embed_texts(client: AzureOpenAI, texts: List[str]) -> List[List[float]]:
    BATCH = 16
    out   = []
    for i in range(0, len(texts), BATCH):
        batch    = texts[i:i + BATCH]
        response = client.embeddings.create(
            model=AZURE_EMBEDDING_DEPLOYMENT_NAME,
            input=batch,
        )
        out.extend([d.embedding for d in response.data])
        print(f"  🔢 Embedded {min(i + BATCH, len(texts))}/{len(texts)}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — STORE IN PINECONE
# ═════════════════════════════════════════════════════════════════════════════

def get_pinecone_index():
    pc       = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"🔧 Creating index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        import time as _t
        while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready"):
            print("  ⏳ Waiting...")
            _t.sleep(2)
    print(f"📌 Connected: {PINECONE_INDEX_NAME}")
    return pc.Index(PINECONE_INDEX_NAME)

def store_in_pinecone(index, chunks: List[Dict], embeddings: List[List[float]]):
    BATCH   = 100
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        vectors.append({
            "id":     chunk["chunk_id"],
            "values": emb,
            "metadata": {
                "type":          chunk["type"],
                "parent_id":     chunk["parent_id"] or "",
                "url":           chunk["url"],
                "title":         chunk["title"],
                "section_title": chunk["section_title"],
                "heading_path":  chunk["heading_path"],
                "content":       chunk["content"][:38000],
                "word_count":    chunk["word_count"],
            },
        })

    total_b = (len(vectors) + BATCH - 1) // BATCH
    for i in range(0, len(vectors), BATCH):
        index.upsert(vectors=vectors[i:i + BATCH])
        print(f"  📤 Batch {i // BATCH + 1}/{total_b}")

    parents  = sum(1 for c in chunks if c["type"] == "parent")
    children = sum(1 for c in chunks if c["type"] == "child")
    print(f"✅ Stored {len(vectors)} vectors ({parents} parents + {children} children)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def ingest(url: str, max_pages: int, max_depth: int):
    t0        = time.time()
    parsed    = urlparse(url.rstrip("/"))
    site_slug = re.sub(r"[^a-z0-9]+", "-",
                       f"{parsed.netloc}{parsed.path}".lower()).strip("-")

    print(f"\n{'#'*55}")
    print(f"INGESTION  (parent+child chunking)")
    print(f"  URL  : {url}")
    print(f"  Slug : {site_slug}")
    print(f"{'#'*55}\n")

    pages = await crawl_site(url, max_pages, max_depth)
    if not pages:
        print("❌ No pages crawled.")
        return

    save_pages_locally(pages, site_slug)
    chunks = make_chunks(pages)
    if not chunks:
        print("❌ No chunks produced.")
        return

    print(f"\n🔢 Embedding {len(chunks)} chunks...")
    oai    = get_openai_client()
    embs   = embed_texts(oai, [c["content"] for c in chunks])

    print(f"\n📤 Storing in Pinecone...")
    idx    = get_pinecone_index()
    store_in_pinecone(idx, chunks, embs)

    elapsed = time.time() - t0
    print(f"\n{'#'*55}")
    print(f"✅ DONE  ({elapsed:.1f}s)")
    print(f"   Pages   : {len(pages)}")
    print(f"   Chunks  : {len(chunks)}")
    print(f"   Index   : {PINECONE_INDEX_NAME}")
    print(f"{'#'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",       required=True)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    args = parser.parse_args()
    asyncio.run(ingest(args.url, args.max_pages, args.max_depth))