import os 
import json 
import re 
import time 
import pdfplumber 
import pymupdf 
import chromadb 
import torch
import pandas as pd
import numpy as np
import fitz
from pathlib import Path


Filings = [
    {"id":"1","company":"Apple","year":2023,"path":r"C:\Users\rushy\Downloads\RAG_for_Finance\Apple_2023.pdf"},
    {"id":"2","company":"Tesla","year":2023,"path":r"C:\Users\rushy\Downloads\RAG_for_Finance\Tesla_2023.pdf"},
]

#------------------------------------------------------------------------------------------------
# EXTRACT TEXT FROM FILINGS
#------------------------------------------------------------------------------------------------
def clean_text(text:str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]","",text).strip()

def extract_text_from_filings(filings):
    all_pages = []

    for filing in filings:
        doc = fitz.open(filing['path'])
        source_name  = Path(filing['path']).name


        for page_idx,page in enumerate(doc,start = 1):
            text = page.get_text("text")
            all_pages.append({
                "filing_id":filing['id'],
                "company":filing['company'],
                "year":filing['year'],
                "source":source_name,
                "page":page_idx,
                "text":clean_text(text),
            })

        doc.close()

    return all_pages
#------------------------------------------------------------------------------------------------
# EXTRACT TABLES FROM FILINGS (CONVERTING  TABLES TO DATAFRAMES AND APPENDING TO ALL_TABLES)
#------------------------------------------------------------------------------------------------
def extract_tables_from_filings(filings,*,max_tables_per_page=None):
    all_tables = []

    for filing in filings:
        pdf_path = filing['path']
        source_name = Path(pdf_path).name

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx,page in enumerate(pdf.pages,start=1):
                tables = page.extract_tables()

                if not tables:
                    continue

                if max_tables_per_page is not None:
                    tables = tables[:max_tables_per_page]

                for t_idx,table in enumerate(tables):
                    df = pd.DataFrame(table)
                    all_tables.append({
                        "filing_id":filing['id'],
                        "company":filing['company'],
                        "year":filing['year'],
                        "source":source_name,
                        "page":page_idx,
                        "table_id":f"{source_name}_p{page_idx}_t{t_idx}",
                        "df":df,
                    })
        return all_tables

#------------------------------------------------------------------------------------------------
# CONVERT TABLE DATAFRAMES TO TEXT
#------------------------------------------------------------------------------------------------

def table_to_text(df):
    df = df.fillna("").astype(str)
    lines = []
    for row in df.values.tolist():
        row = [cell.strip() for cell in row if cell.strip()]
        lines.append(" | ".join(row))
    return "\n".join(lines)

#------------------------------------------------------------------------------------------------
# EXTRACT YEAR HEADERS FROM PAGE TEXT
#------------------------------------------------------------------------------------------------
def extract_year_headers(page_text, max_years=6):
    years = re.findall(r"\b(20\d{2})\b", page_text)
    seen = []
    for y in years:
        if y not in seen:
            seen.append(y)
    return seen[:max_years]


#------------------------------------------------------------------------------------------------
# ATTACH YEARS TO TABLE TEXT IF POSSIBLE
#------------------------------------------------------------------------------------------------
def attach_years_if_possible(df, years):
    df = df.fillna("").astype(str)
    if not years:
        return table_to_text(df)
    lines = []
    for row in df.values.tolist():
        row = [cell.strip() for cell in row if cell.strip()]
        if not row:
            continue
        label = row[0]
        vals = [c for c in row[1:] if c not in {"$","%"}]
        if len(vals) >= len(years) and len(years) >= 2:
            pairs = [f"{y}: {vals[i]}" for i, y in enumerate(years)]
            lines.append(label + " | " + " | ".join(pairs))
        else:
            lines.append(" | ".join(row))
    return "\n".join(lines)




#------------------------------------------------------------------------------------------------
# BUILD DOCS FROM PAGES AND TABLES
#------------------------------------------------------------------------------------------------
def build_table_docs(tables, pages):
    """
    tables: output of extract_tables_from_filings
    pages:  output of Step 1 (page-by-page text extraction)
            IMPORTANT: pages must contain dicts with keys: company, year, page, text
    returns: list of table documents (embedding-ready later)
    """
    # Create a quick lookup for page text by (company, year, page)
    page_text_lookup = {(p["company"], p["year"], p["page"]): p["text"] for p in pages}

    table_docs = []
    for t in tables:
        page_text = page_text_lookup.get((t["company"], t["year"], t["page"]), "")
        years = extract_year_headers(page_text)

        table_body = attach_years_if_possible(t["df"], years)

        table_docs.append({
            "id": t["table_id"],
            "type": "table",
            "filing_id": t["filing_id"],
            "company": t["company"],
            "year": t["year"],
            "source": t["source"],
            "page": t["page"],
            "years_detected": years,  # helpful for debugging
            "text": (
                f"[TABLE]\n"
                f"Company: {t['company']}\n"
                f"FilingYear: {t['year']}\n"
                f"Source: {t['source']}\n"
                f"Page: {t['page']}\n"
                f"Years: {', '.join(years) if years else 'N/A'}\n\n"
                f"{table_body}"
            )
        })

    return table_docs

   






pages = extract_text_from_filings(Filings)
tables = extract_tables_from_filings(Filings)
table_docs = build_table_docs(tables, pages)


print("Total table docs:", len(table_docs))

# preview one
print("\n--- Sample Table Doc ---")
print("ID:", table_docs[10]["id"])
print("Meta:", table_docs[10]["company"], table_docs[10]["year"], "page", table_docs[10]["page"])
print(table_docs[10]["text"][:800])
