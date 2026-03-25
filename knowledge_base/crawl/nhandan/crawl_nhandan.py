import requests
from bs4 import BeautifulSoup
import json
from underthesea import sent_tokenize


def crawl_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    res = requests.get(url, headers=headers)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "html.parser")

    # title
    title = soup.find("h1").get_text(strip=True)

    # date
    date_tag = soup.find("time")
    date = date_tag.get_text(strip=True) if date_tag else ""

    # content
    paragraphs = soup.select('div[itemprop="articleBody"] p')

    text_list = []
    for p in paragraphs:

        if p.get("class") == ["name"]:
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
    
    source_id = "nd"
    source = "nhandan"

    with open("../../link/raw/combine/nhandan_CT.txt", "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    all_data = []

    for article_index, url in enumerate(urls):

        title, date, text = crawl_article(url)

        chunks = chunk_text(text)

        kb_items = build_json(source_id, source, url, date, chunks, article_index, topic="Domestic", title=title)

        all_data.extend(kb_items)


    with open("KB_nhandan.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()