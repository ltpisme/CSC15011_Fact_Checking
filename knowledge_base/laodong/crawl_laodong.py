from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
from underthesea import sent_tokenize


def crawl_article(url):
    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, wait_until="domcontentloaded")

        html = page.content()

        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    # title
    title_tag = soup.select_one("h1.title") or soup.select_one("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # date
    date_tag = soup.select_one("span.time")
    date = date_tag.get_text(strip=True) if date_tag else ""

    # content
    paragraphs = soup.select("#gallery-ctt p")

    if not paragraphs:
        paragraphs = soup.select('div[itemprop="articleBody"] p')

    text_list = []

    for p in paragraphs:

        if "name" in (p.get("class") or []):
            continue

        text = p.get_text(strip=True)

        if text:
            text_list.append(text)

    full_text = "\n".join(text_list)

    return title, date, full_text


def chunk_text(text, chunk_size=1000):

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sent in sentences:

        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent + " "

        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def build_json(source_id, source, url, date, chunks, article_index, topic, title):

    data = []

    for chunk_index, chunk in enumerate(chunks):

        item = {
            "id": f"{source_id}_{article_index}_chunk_{chunk_index}",
            "source": source,
            "url": url,
            "text": chunk,
            "metadata": {
                "date": date,
                "topic": topic,
                "title": title
            }
        }

        data.append(item)

    return data

def main():

    source_id = "ld"   
    source = "laodong"

    all_data = []

    with open("../../link/raw/combine/laodong_CT.txt", "r", encoding="utf-8") as f:
        urls_CT = [line.strip() for line in f if line.strip()]

    with open("../../link/raw/combine/laodong_TG.txt", "r", encoding="utf-8") as f:
        urls_TG = [line.strip() for line in f if line.strip()]

    article_index = 0

    # crawl CT
    for url in urls_CT:

        title, date, text = crawl_article(url)
        chunks = chunk_text(text)

        items = build_json(source_id, source, url, date, chunks, article_index, topic="Domestic", title=title)
        all_data.extend(items)

        article_index += 1


    # crawl TG
    for url in urls_TG:

        title, date, text = crawl_article(url)
        chunks = chunk_text(text)

        items = build_json(source_id, source, url, date, chunks, article_index, topic="International", title=title)
        all_data.extend(items)

        article_index += 1

    with open("KB_laodong.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()