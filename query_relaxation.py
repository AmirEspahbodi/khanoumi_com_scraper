import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openpyxl import load_workbook
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv(override=True)

# Configuration
INPUT_FILE = "Book1.xlsx"
COLUMN_NAME = "title"
MODEL_NAME = "gemini-3-flash-preview"


# Define the Pydantic schema for structured output
class NormalizedResult(BaseModel):
    normalized_search_query: str
    normalized_product_name1: str
    normalized_product_name2: str
    normalized_product_name3: str
    normalized_product_name4: str
    normalized_product_name5: str


class KeyManager:
    def __init__(self, keys):
        if not keys:
            raise ValueError("API Keys list is empty! Check your .env file format.")
        self.keys = keys
        self.current_index = 0
        self.client = genai.Client(api_key=self.keys[self.current_index])
        print(f"Initialized with {len(self.keys)} API keys. Starting with Key 1.")

    def get_client(self):
        return self.client

    def switch_key(self):
        self.current_index += 1
        # Reached the end of the key list
        if self.current_index >= len(self.keys):
            print(
                "\n[!] All API keys exhausted. Waiting for 1 minute before restarting cycle..."
            )
            time.sleep(60)
            self.current_index = 0

        print(f"\n[>] Switched to API Key {self.current_index + 1} of {len(self.keys)}")
        # Instantiate a new client with the new key
        self.client = genai.Client(api_key=self.keys[self.current_index])


def load_api_keys() -> list[str]:
    """
    Loads API keys from a comma-separated list in the environment variable.
    """
    # 1. Fetch the raw string from the environment
    raw_keys = os.getenv("GEMINI_API_KEYS", "")

    if not raw_keys:
        # Fallback to the singular old name just in case
        raw_keys = os.getenv("GEMINI_API_KEY", "")

    if not raw_keys:
        raise ValueError("GEMINI_API_KEYS environment variable is missing or empty.")

    # 2. Split, strip whitespace, and filter out empty strings
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]

    if not keys:
        raise ValueError("No valid API keys found after parsing.")

    return keys


# Initialize our KeyManager globally
try:
    API_KEYS = load_api_keys()
    key_manager = KeyManager(API_KEYS)
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit(1)


def normalize_query(raw_query: str) -> NormalizedResult | None:
    prompt = f"""# System Role
You are an expert Data Normalizer and SEO Specialist in the cosmetics and beauty industry. Your exact task is to take messy, human-written cosmetic product names and convert them into highly optimized search queries and exactly 5 structured, normalized product names.

# Objective
For the provided raw product name, you must generate a JSON response exactly matching this schema:
{{
    "normalized_search_query": "string",
    "normalized_product_name1": "string",
    "normalized_product_name2": "string",
    "normalized_product_name3": "string",
    "normalized_product_name4": "string",
    "normalized_product_name5": "string"
    }}

# 🚫 ABSOLUTE RED LINE (CRITICAL TO AVOID FALSE POSITIVES):
- NO ADDITIONS: If the raw name DOES NOT contain a specific color, weight, volume, size, amount, or specific model, NONE of the 5 normalized names should have them. DO NOT guess, hallucinate, or add default values.
- NO DELETIONS: If the raw name HAS a specific shade (e.g., NC41) or volume (e.g., 125m), it MUST be present in ALL 5 variations.

# Rules for "normalized_search_query":
- Broad enough to yield results, specific enough to filter out junk.
- Include ONLY the Brand, Core Product Type, and Main Line/Model.
- EXCLUDE highly specific details like exact volume (e.g., 100ml) or specific color codes (e.g., NC41, m010) because searching for the exact shade might yield zero results.

# Rules for "normalized_product_name" Variations:
- normalized_product_name1 (Standard Format): [Product Type] + [Brand] + [Model/Feature] + [Code/Color] + [Volume (Standardized)]. Fix spacing and typos.
- normalized_product_name2 (Mixed Format): [Brand (Keep original)] + [Product Type] + [Model/Code].
- normalized_product_name3 (Expanded/Cleaned): Expand abbreviations (e.g., "م" or "میل" to "میلی لیتر"). Reorder slightly to mimic formal catalog names.
- normalized_product_name4 (Fully Persianized Standard E-commerce): Translate brand and category to Persian if possible. Convert any Finglish terms to Persian script. Normalize volume/weight to a standard Persian format (e.g., "میلی لیتر"). Fix spelling mistakes.
- normalized_product_name5 (English Brand/Model Standard E-commerce): Keep the base text and volume/weight formatting in PERSIAN (e.g., "حجم ... میلی لیتر"). However, translate ONLY the Brand, Category (if applicable), and Finglish terms to pure ENGLISH. Example: "کرم پودر MAC مدل Studio Fix".

# Output Format
You must respond ONLY with a valid JSON object. Do not include markdown tags like ```json or any conversational text.

# Examples
Input: "مک کرم پودراستودیو فیکس NC 25"
Output:
{{
"normalized_search_query": "کرم پودر مک استودیو فیکس",
"normalized_product_name1": "کرم پودر مک استودیو فیکس NC25",
"normalized_product_name2": "مک کرم پودر Studio Fix NC25",
"normalized_product_name3": "کرم پودر مک استودیو فیکس NC 25",
"normalized_product_name4": "کرم پودر مک استودیو فیکس ان سی 25",
"normalized_product_name5": "کرم پودر MAC Studio Fix NC25"
}}

Input: "لورال پرایمر تیوپی اینفالیبل 35 میل"
Output:
{{
"normalized_search_query": "پرایمر تیوپی لورال اینفالیبل",
"normalized_product_name1": "پرایمر تیوپی لورال اینفالیبل 35 میلی لیتر",
"normalized_product_name2": "لورال پرایمر تیوپی اینفالیبل 35 میل",
"normalized_product_name3": "پرایمر تیوپی لورال اینفالیبل 35 میلی لیتر",
"normalized_product_name4": "پرایمر تیوپی لورآل اینفالیبل 35 میلی لیتر",
"normalized_product_name5": "پرایمر تیوپی Loreal Infallible 35 میلی لیتر"
}}

Input: "بل ریمل فول لش"
Output:
{{
"normalized_search_query": "ریمل بل فول لش",
"normalized_product_name1": "ریمل بل فول لش",
"normalized_product_name2": "بل ریمل فول لش",
"normalized_product_name3": "ریمل بل فول لش",
"normalized_product_name4": "ریمل بل فول لش",
"normalized_product_name5": "ریمل Bell Full Lash"
}}

Input: "ادکلن دیویدوف کول واتر 125م"
Output:
{{
"normalized_search_query": "ادکلن دیویدوف کول واتر",
"normalized_product_name1": "ادکلن دیویدوف کول واتر 125 میلی لیتر",
"normalized_product_name2": "دیویدوف ادکلن کول واتر 125m",
"normalized_product_name3": "ادکلن دیویدوف کول واتر 125 میلی لیتر",
"normalized_product_name4": "ادکلن دیویدوف کول واتر 125 میلی لیتر",
"normalized_product_name5": "ادکلن Davidoff Cool Water 125 میلی لیتر"
}}

# Real Task
Process the following raw product name exactly according to the rules and return the JSON:
Input: "{raw_query}"
"""

    while True:
        try:
            client = key_manager.get_client()

            # Use structured output config to enforce the JSON schema
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=NormalizedResult,
                    temperature=0.1,
                ),
            )

            # Parse and validate the response automatically using Pydantic
            parsed_result = NormalizedResult.model_validate_json(response.text)
            return parsed_result

        except ValidationError as ve:
            print(f"Pydantic Validation error for '{raw_query}': {ve}")
            return None
        except Exception as e:
            error_msg = str(e).lower()

            # Catch 429 Too Many Requests, 403 Quota Exceeded, or resource exhaustion
            if any(
                keyword in error_msg
                for keyword in [
                    "429",
                    "403",
                    "quota",
                    "exhausted",
                    "rate limit",
                    "too many requests",
                ]
            ):
                print(f"Rate limit or quota hit (Error: {e}). Switching API key...")
                key_manager.switch_key()
                continue
            else:
                # If it's a different kind of error, log it and move to the next item
                print(f"Error processing '{raw_query}': {e}")
                return None


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Load the workbook and get the active worksheet
    wb = load_workbook(INPUT_FILE)
    ws = wb.active

    # Extract headers from the first row to find column indices
    headers = [cell.value for cell in ws[1]]

    if COLUMN_NAME not in headers:
        print(f"Error: Column '{COLUMN_NAME}' not found in the Excel file.")
        return

    # openpyxl uses 1-based indexing for columns
    title_col_idx = headers.index(COLUMN_NAME) + 1

    # Define the required output columns
    output_columns = [
        "normalized_search_query",
        "normalized_product_name1",
        "normalized_product_name2",
        "normalized_product_name3",
        "normalized_product_name4",
        "normalized_product_name5",
    ]

    col_indices = {}

    # Ensure all 4 columns exist in the header, otherwise append them
    for col_name in output_columns:
        if col_name in headers:
            col_indices[col_name] = headers.index(col_name) + 1
        else:
            new_col_idx = len(headers) + 1
            ws.cell(row=1, column=new_col_idx, value=col_name)
            headers.append(col_name)  # Keep track dynamically
            col_indices[col_name] = new_col_idx

    wb.save(INPUT_FILE)  # Save headers immediately

    total_rows = ws.max_row - 1  # Subtract 1 for the header row
    print(f"Start processing {total_rows} rows...\n")

    # Iterate starting from row 2 (skipping header)
    for row_num in range(2, ws.max_row + 1):
        # Check if the row has already been processed successfully
        already_processed = True
        for col_name in output_columns:
            cell_val = ws.cell(row=row_num, column=col_indices[col_name]).value
            if cell_val is None or str(cell_val).strip().lower() in ["", "nan"]:
                already_processed = False
                break

        if already_processed:
            continue

        title_cell = ws.cell(row=row_num, column=title_col_idx)
        raw_q = title_cell.value

        # Skip if the source title is empty
        if raw_q is None or str(raw_q).strip() == "":
            continue

        raw_q_str = str(raw_q)
        print(f"Processing ({row_num - 1}/{total_rows}): {raw_q_str}")

        # Fetch using the Pydantic/Gemini integration
        normalized_result = normalize_query(raw_q_str)

        if normalized_result:
            # Map Pydantic model values directly back into the specific Excel cells
            ws.cell(
                row=row_num,
                column=col_indices["normalized_search_query"],
                value=normalized_result.normalized_search_query,
            )
            ws.cell(
                row=row_num,
                column=col_indices["normalized_product_name1"],
                value=normalized_result.normalized_product_name1,
            )
            ws.cell(
                row=row_num,
                column=col_indices["normalized_product_name2"],
                value=normalized_result.normalized_product_name2,
            )
            ws.cell(
                row=row_num,
                column=col_indices["normalized_product_name3"],
                value=normalized_result.normalized_product_name3,
            )
            ws.cell(
                row=row_num,
                column=col_indices["normalized_product_name4"],
                value=normalized_result.normalized_product_name4,
            )
            ws.cell(
                row=row_num,
                column=col_indices["normalized_product_name5"],
                value=normalized_result.normalized_product_name5,
            )
            # Overwrite the input file to save progress safely
            wb.save(INPUT_FILE)
            print("Saved outputs successfully.")

        # Sleep to respect base rate limits
        time.sleep(2)


if __name__ == "__main__":
    main()
