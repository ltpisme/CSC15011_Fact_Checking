# Analysis of link:

## Number of link:
- Total number of links: 3374 
1. **vnexpress**: 1066 - **Van** 
2. **baochinhphu**: 12 - **Van**
3. **dantri**: 268 - **Van**
4. **kenh14**: 3 - **Van**
5. **laodong**: 131 - **Bao**
6. **nguoilaodong**: 39 - **Bao**
7. **nhandan**: 2 - **Bao**
8. **quandoinhandan**: 5 - **Bao**
9. **thanhnien**: 468 - **Phong**
10. **tienphong**: 6 - **Phong**
11. **tuoitre**: 655 - **Phong**
12. **vietnam**: 6 - **Phong**
13. **vietnamnet**: 328 - **Kien**
15. **vov**: 19 - **Kien**
16. **vtcnews**: 21 - **Kien**
17. **vtv**: 343 - **Kien**



# Knowledge base
## Tools: 
Using ``python`` with ``scrappy`` or other tools

## Format
File name: ``KB_source.json`` with source is the name taken from the list above.

- **id**: ``Source_a_chunk_b`` with ``Source`` is source name from the above list, ``a`` is the number of the link, ``b`` is the number of chunk crawled from the article.
- **source**: is source name, taken from above list.
- **url**: is the complete url of the article 
- **text** is the text extracted from the article to be a context
- **metadata**: Included 2 part: ``date`` is the release date of the article ``topic`` is the category of the article. Depend on the ``/raw/combine`` file type is **``CT``** will be  ``Domestic`` or **``TG``** will be ``International``

```Json
[
    {
        "id": "Source_0_chunk_0",
        "source": “vnexpress”,
        "url": "https://vnexpress.net/gia-vang-mieng-vuot-190-trieu-dong-5011315.html",
        "text": "Sáng 29/1, Công ty Vàng bạc Đá quý Sài Gòn (SJC) niêm yết giá vàng miếng tại 187,2 - 190,2 triệu đồng một lượng, tăng 5,5 triệu đồng,...",
        "metadata": {
          "date": "2026-01-29",
          "topic": " Domestic / International",
          "title": "Giá vàng miếng vượt 190 triệu đồng"
        }
  },
]
```

# Crawling step
+ Chunk size: 1000 character, paragraph base.
+ Each article: 3 chunk (context)
```
    Links
      ↓
  Crawl HTML
(using tools, python, something,...)
      ↓
  Extract article content
(using algorithm, libraries,...)
      ↓
  Clean text
      ↓
  Chunk text
      ↓
  Save to KB JSON
```

# Git
The git tree for the knowledge base will be

```pwsh
.root
├── knowledge_base
│   ├── tuoitre
│   │   ├── crawl_tuoitre.py
│   │   └── KB_tuoitre.json
├── link
│   ├── org
│   └── raw
│       └── combine
```

