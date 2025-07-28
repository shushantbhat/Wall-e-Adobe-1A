# 📘 PDF Outline Extractor - Connecting the Dots Challenge

A lightweight, multilingual, ML-powered solution to intelligently extract structured outlines (title, H1-H3 headings with page numbers) from PDFs. This approach blends transformer-based semantic analysis with traditional document layout features for robust and accurate extraction.

---

## 🚀 Overview

This system uses **DistilBERT** (`distilbert-base-multilingual-cased`) to semantically understand headings, combining it with formatting and positional cues to identify document structure.

- **Model Size**: ~135MB (under 200MB limit)
- **Languages Supported**: 104+ (English, Japanese, Chinese, Spanish, French, etc.)
- **Performance**: ~2–3 seconds for a 50-page PDF
- **Architecture**: CPU-only, runs offline, AMD64

---

## 🧠 Core Features

### 1. 🧩 Intelligent Heading Detection

- **Semantic Analysis**: Uses DistilBERT embeddings for text understanding.
- **Multi-feature Classification**: Combines font size, boldness, position, and semantics.
- **Language-Agnostic**: Handles documents in multiple scripts and languages.
- **ML Scoring**: Weighted combination of multiple signals to identify heading levels.

### 2. 🔍 Feature Engineering

- Font size & formatting
- Text position in layout
- Multilingual semantic embeddings
- Language-specific patterns

### 3. 🌐 Multilingual Support

- **Auto Language Detection**: Based on character sets and writing patterns
- **Unicode Normalization**: Proper handling of diacritics and special scripts
- **Cross-language heading recognition**: English, Japanese, Chinese, Spanish, French

---

## 🧰 Libraries & Models Used

- [`PyMuPDF`](https://pymupdf.readthedocs.io/) (fitz): PDF text extraction
- [`Transformers`](https://huggingface.co/docs/transformers): DistilBERT
- [`PyTorch`](https://pytorch.org/): ML backend (CPU only)
- `distilbert-base-multilingual-cased`: Pre-trained multilingual transformer

---

## 📈 Time and Space Complexity

| Aspect             | Estimate                          |
|--------------------|-----------------------------------|
| **Per Page Time**  | O(n × m) (n = text blocks, m = model inference) |
| **Total Time**     | O(p × n × m) (p = pages) |
| **Performance**    | ~2–3 seconds for 50-page PDF |
| **Memory Usage**   | ~200–300MB peak |
| **Storage**        | Minimal, no temp bloat |

---

## 🛠️ Performance Optimizations

- 🧠 Model Caching: Downloaded once, used offline
- ⚙️ Batch Processing: Fast text chunking
- 🧮 Efficient Feature Extraction
- 🧼 Memory Management: Clean resource handling
- 🧵 CPU-optimized PyTorch build

---

## 🧪 Input/Output

### 📂 Input

- Place PDFs in the `./input/` directory

### 📤 Output

- JSON files will appear in the `./output/` directory
- Format: Each `filename.pdf` → `filename.json`

#### 📝 Sample Output

```json
{
  "title": "Understanding AI",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "What is AI?",
      "page": 2
    },
    {
      "level": "H3",
      "text": "History of AI",
      "page": 3
    }
  ]
}
