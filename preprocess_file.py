#!/usr/bin/env python
# coding: utf-8

import re
import json
import os
from collections import defaultdict
from pypdf import PdfReader
import uuid
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from math import log
import numpy as np
from llama_index.llms.ollama import Ollama
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

class PdfTextExtractor:
    def __init__(self, pdf_path, output_txt_path):
        self.pdf_path = pdf_path
        self.output_txt_path = output_txt_path

    def extract_text(self):
        if not os.path.exists(self.output_txt_path):
            reader = PdfReader(self.pdf_path)
            text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
            with open(self.output_txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(self.output_txt_path, "r", encoding="utf-8") as f:
                text = f.read()
        return text

class TextCleaner:
    @staticmethod
    def clean_text(text):
        sub_pattern = r'Code des assurances - Dernière modification le 15 août 2025 - Document généré le 14 août 2025'
        text = re.sub(sub_pattern, '', text)
        return text.strip()

class HierarchyParser:
    def __init__(self):
        self.patterns = [
            r'(Partie législative|Partie réglementaire - Arrêtés|Partie réglementaire)\n',
            r"(^Livre [IVXLCDM]+.*$)",
            r"(^Titre [IVXLCDM]+.*$)",
            r"(^Chapitre [IVXLCDM]+.*$)",
            r"(^Section (?:[IVXLCDM]+|[0-9]+)+.*$)",
            r"(^Sous-section\s+(?:[IVXLCDM]+|[0-9]+).*$)"
        ]
        self.level_keys = ["partie", "livre", "titre", "chapitre", "section", "sous_section"]
        self.article_pattern = r'(Article\s+[A-Z]\*?\d+(?:-\d+)*)'

    def split_by_articles(self, text):
        articles_splits = re.split(self.article_pattern, text, flags=re.M)
        return articles_splits[1::2], articles_splits[2::2], articles_splits[0::2]

    def detect_hierarchy(self, preceding_text, prev_hierarchy):
        curr_hierarchy = prev_hierarchy.copy()
        for idx, pattern in enumerate(reversed(self.patterns)):
            splt = re.split(pattern, preceding_text, flags=re.M)
            if len(splt) > 1:
                new_val = f"{splt[-2].strip()} {splt[-1].strip()}".strip()
                preceding_text = "".join(s for s in splt[:-2])
                # print(new_val)
                curr_idx = len(self.level_keys) - idx - 1
                if curr_hierarchy[self.level_keys[curr_idx]] != new_val:
                    curr_hierarchy[self.level_keys[curr_idx]] = new_val
                    for lower_idx in range(curr_idx + 1, len(self.level_keys)):
                        if (curr_hierarchy[self.level_keys[lower_idx]]) == prev_hierarchy[self.level_keys[lower_idx]]:
                            curr_hierarchy[self.level_keys[lower_idx]] = ""
        return curr_hierarchy

class ArticleProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('french')) | {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'en', 'pour', 'par'}
        self.documents = []  # Store all article contents for TF-IDF
        self.word_doc_freq = defaultdict(int)  # Document frequency for words
        self.llm = Ollama(
            model="llama3.2:1b",
            temperature=0.1
        )

    def extract_keywords(self, content, top_n=5):
        # Tokenize and tag parts of speech
        words = word_tokenize(content.lower())
        tagged_words = pos_tag(words, lang='eng')  # Using English POS tagger as French is less reliable
        # Filter for nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS)
        candidates = [word for word, pos in tagged_words if pos in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS') and word not in self.stop_words and len(word) > 2]

        # Calculate TF-IDF
        word_freq = defaultdict(int)
        for word in candidates:
            word_freq[word] += 1

        tfidf_scores = {}
        total_docs = len(self.documents) if self.documents else 1
        for word, freq in word_freq.items():
            tf = freq / max(len(candidates), 1)
            idf = log(total_docs / (self.word_doc_freq[word] + 1)) + 1
            tfidf_scores[word] = tf * idf

        # Sort and select top keywords
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_keywords[:top_n]]

    def generate_summary(self, content):
        prompt = (
        "Provide a concise summary of the following French insurance code article in about 100 characters in french. "
        "Focus on key legal obligations and procedures, using precise legal terminology and only give the summary as answer:\n\n"
        f"{content}"
        )
        response = self.llm.complete(prompt)
        summary = str(response).strip()
        return summary
        

    def process_article(self, article_id, content, curr_hierarchy, reference_graph, all_articles):
        content = re.sub(r"\n", " ", content.strip())
        # Update document frequency for TF-IDF
        words = set(word_tokenize(content.lower()))
        for word in words:
            self.word_doc_freq[word] += 1
        self.documents.append(content)

        references = re.findall(r"[A-Z]\.\s*\d{3}-\d+(?:-\d+)?", content)
        for ref in references:
            reference_graph["Article " + re.sub(". ", "", ref)].append(article_id)
            
        patterns = [
            r'Partie législative',
            r'Partie réglementaire - Arrêtés',
            r'Partie réglementaire',
            r'Livre [IVXLCDM]+',
            r'Titre [IVXLCDM]+',
            r'Chapitre [IVXLCDM]+',
            r'Section (?:[IVXLCDM]+|[0-9]+)+',
            r'Sous-section\s+(?:[IVXLCDM]+|[0-9]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                # Cut everything from this match onward
                content = content[:match.start()]
                break  # Stop after the first match

        article = {
            "article_id": article_id,
            "content": content.strip(),
            "hierarchy": curr_hierarchy.copy(),
            "references": ["Article " + re.sub(". ", "", r) for r in references],
            "referenced_by": [],
            "summary": "", # self.generate_summary(content),
            "keywords": self.extract_keywords(content),
            "page_number": None
        }
        all_articles.add(article_id)
        return article

class CodeAssurancesProcessor:
    def __init__(self, pdf_path="LEGITEXT000006073984.pdf", txt_path="code_assurances_raw.txt", json_path="code_assurances.json"):
        self.pdf_path = pdf_path
        self.txt_path = txt_path
        self.json_path = json_path
        self.extractor = PdfTextExtractor(pdf_path, txt_path)
        self.cleaner = TextCleaner()
        self.parser = HierarchyParser()
        self.article_processor = ArticleProcessor()
        self.hierarchy_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
        self.articles_list = []
        self.reference_graph = defaultdict(list)
        self.all_articles = set()

    def process(self):
        # Extract and clean text
        text = self.extractor.extract_text()
        text = self.cleaner.clean_text(text)
        
        # Split text by articles
        articles_id, articles_content, preceding_texts = self.parser.split_by_articles(text)
        print("Number of articles:", len(articles_id))

        # Process articles
        prev_hierarchy = {lvl: "" for lvl in self.parser.level_keys}
        for i, article_id in tqdm(enumerate(articles_id), desc="Processing articles"):
            preceding_text = preceding_texts[i] if i < len(preceding_texts) else ""
            curr_hierarchy = self.parser.detect_hierarchy(preceding_text, prev_hierarchy)
            
            # Process article
            article = self.article_processor.process_article(
                article_id, articles_content[i], curr_hierarchy, 
                self.reference_graph, self.all_articles
            )
            self.articles_list.append(article)

            # Build hierarchy tree
            node = self.hierarchy_tree
            for lvl in self.parser.level_keys[:-1]:
                if curr_hierarchy[lvl]:
                    node = node[curr_hierarchy[lvl]]
            node["articles"] = node.get("articles", []) + [article_id]

            prev_hierarchy = curr_hierarchy.copy()

        # Populate referenced_by field
        for article in self.articles_list:
            article["referenced_by"] = self.reference_graph.get(article["article_id"], [])

        # Save output
        output = {
            "articles": self.articles_list,
            "hierarchy_tree": dict(self.hierarchy_tree)
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print("Data saved to", self.json_path)

if __name__ == "__main__":
    processor = CodeAssurancesProcessor()
    processor.process()