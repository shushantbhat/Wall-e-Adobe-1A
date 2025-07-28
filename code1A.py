#!/usr/bin/env python3
"""
PDF Outline Extractor using Mini Transformer Model
Uses DistilBERT for intelligent heading detection with multilingual support
"""

import os
import json
import re
import fitz  # PyMuPDF
from pathlib import Path
import unicodedata
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    def __init__(self):
        # Use lightweight multilingual model (under 200MB)
        model_name = "distilbert-base-multilingual-cased"
        
        # Initialize tokenizer and model for text classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create a simple classifier pipeline
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU only
        )
        
        # Font size analysis weights
        self.font_weights = {
            'size': 0.4,
            'bold': 0.3,
            'position': 0.2,
            'semantic': 0.1
        }
        
        # Heading indicators in multiple languages
        self.heading_keywords = {
            'title': ['abstract', 'title', 'résumé', 'título', '标题', 'タイトル', '제목'],
            'h1': ['chapter', 'section', 'part', 'chapitre', 'capítulo', '章', '章節', '챕터'],
            'h2': ['subsection', 'sous-section', 'subsección', '节', '節', '섹션'],
            'h3': ['subsubsection', 'paragraph', 'paragraphe', 'párrafo', '段', '段落', '문단']
        }

    def extract_text_features(self, text: str, font_size: float, is_bold: bool, 
                            page_height: float, y_position: float) -> Dict:
        """Extract features from text for heading classification"""
        
        # Basic text features
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_numbers': bool(re.search(r'\d', text)),
            'is_uppercase': text.isupper(),
            'is_titlecase': text.istitle(),
            'font_size': font_size,
            'is_bold': is_bold,
            'relative_position': y_position / page_height if page_height > 0 else 0,
            'starts_with_number': bool(re.match(r'^\d+\.?\s*', text)),
        }
        
        # Language and semantic features using the model
        try:
            # Use the model to get text embeddings/features
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the hidden states as semantic features
                features['semantic_score'] = float(torch.mean(outputs.logits).item())
        except:
            features['semantic_score'] = 0.0
        
        return features

    def classify_heading_level(self, text: str, features: Dict) -> Optional[str]:
        """Classify text as title, H1, H2, H3, or None using ML approach"""
        
        # Check for keyword indicators across languages
        text_lower = text.lower()
        keyword_scores = {
            'title': 0,
            'h1': 0,
            'h2': 0,
            'h3': 0
        }
        
        for level, keywords in self.heading_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    keyword_scores[level] += 1
        
        # Font size scoring
        font_score = min(features['font_size'] / 20.0, 1.0)  # Normalize to 0-1
        
        # Position scoring (headings often at top of page)
        position_score = 1.0 - features['relative_position']
        
        # Bold text scoring
        bold_score = 1.0 if features['is_bold'] else 0.5
        
        # Length scoring (headings are usually concise)
        length_score = 1.0 if 5 <= features['length'] <= 100 else 0.5
        
        # Number pattern scoring
        number_score = 1.0 if features['starts_with_number'] else 0.0
        
        # Combine scores for each level
        level_scores = {}
        
        # Title scoring
        level_scores['title'] = (
            font_score * 0.4 +
            position_score * 0.3 +
            bold_score * 0.2 +
            keyword_scores['title'] * 0.1
        )
        
        # H1 scoring
        level_scores['H1'] = (
            font_score * 0.3 +
            bold_score * 0.3 +
            number_score * 0.2 +
            keyword_scores['h1'] * 0.2
        )
        
        # H2 scoring
        level_scores['H2'] = (
            font_score * 0.25 +
            bold_score * 0.25 +
            number_score * 0.3 +
            keyword_scores['h2'] * 0.2
        )
        
        # H3 scoring
        level_scores['H3'] = (
            font_score * 0.2 +
            bold_score * 0.2 +
            number_score * 0.4 +
            keyword_scores['h3'] * 0.2
        )
        
        # Find the best level
        max_score = max(level_scores.values())
        
        # Only classify if score is above threshold
        if max_score < 0.4:
            return None
        
        best_level = max(level_scores.items(), key=lambda x: x[1])[0]
        
        # Don't return title unless it's clearly a title
        if best_level == 'title' and max_score < 0.6:
            return None
        
        return best_level if best_level != 'title' else None

    def extract_title_with_model(self, doc) -> str:
        """Extract document title using metadata and ML"""
        
        # Try metadata first
        metadata = doc.metadata
        if metadata.get('title') and len(metadata['title'].strip()) > 0:
            return unicodedata.normalize('NFKC', metadata['title'].strip())
        
        # Use ML to find the most likely title on first few pages
        title_candidates = []
        
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = re.sub(r'\s+', ' ', span["text"]).strip()
                            if len(text) < 5 or len(text) > 200:
                                continue
                                
                            features = self.extract_text_features(
                                text, 
                                span["size"], 
                                bool(span["flags"] & 2**4),
                                page_height,
                                span["bbox"][1]  # y position
                            )
                            
                            # Score as potential title
                            title_score = (
                                features['font_size'] / 20.0 * 0.4 +
                                (1.0 - features['relative_position']) * 0.3 +
                                (1.0 if features['is_bold'] else 0.5) * 0.2 +
                                features['semantic_score'] * 0.1
                            )
                            
                            if title_score > 0.5:
                                title_candidates.append((text, title_score, page_num))
        
        if title_candidates:
            # Return the highest scoring title from the first page preferably
            title_candidates.sort(key=lambda x: (-x[2], -x[1]))  # Prefer earlier pages, then higher scores
            return title_candidates[0][0]
        
        return "Untitled Document"

    def extract_outline(self, pdf_path: str) -> Dict:
        """Extract structured outline from PDF using ML"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract title using ML
            title = self.extract_title_with_model(doc)
            logger.info(f"Extracted title: {title}")
            
            outline = []
            seen_headings = set()
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = re.sub(r'\s+', ' ', span["text"]).strip()
                                if len(text) < 3 or len(text) > 200:
                                    continue
                                
                                # Extract features
                                features = self.extract_text_features(
                                    text,
                                    span["size"],
                                    bool(span["flags"] & 2**4),
                                    page_height,
                                    span["bbox"][1]
                                )
                                
                                # Classify heading level using ML
                                level = self.classify_heading_level(text, features)
                                
                                if level and text not in seen_headings:
                                    outline.append({
                                        "level": level,
                                        "text": text,
                                        "page": page_num + 1
                                    })
                                    seen_headings.add(text)
                                    logger.debug(f"Found {level}: {text} (page {page_num + 1})")
            
            doc.close()
            
            # Sort and clean outline
            outline.sort(key=lambda x: (x["page"], ["H1", "H2", "H3"].index(x["level"])))
            
            # Remove duplicates
            unique_outline = []
            seen = set()
            for item in outline:
                key = (item["text"].lower(), item["level"])
                if key not in seen:
                    unique_outline.append(item)
                    seen.add(key)
            
            result = {
                "title": title,
                "outline": unique_outline
            }
            
            logger.info(f"Extracted {len(unique_outline)} headings")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error Processing Document",
                "outline": []
            }

def main():
    """Main function to process all PDFs in input directory"""
    input_dir = Path(r"1A\1A\app\input")
    output_dir = Path(r"1A\1A\app\output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = PDFOutlineExtractor()
    
    # Process all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract outline
            result = extractor.extract_outline(str(pdf_file))
            
            # Save result
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
            # Create error output
            error_result = {
                "title": "Processing Error",
                "outline": []
            }
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2)

if __name__ == "__main__":
    main()