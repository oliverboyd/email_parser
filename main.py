#!/usr/bin/env python3
"""
OpenAI PDF Email Processor
Extracts structured email data from PDF files using GPT-4 Vision API
"""

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
INPUT_DIR = os.getenv("INPUT_DIR", ".")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "email_data.csv")
PROCESSED_LOG = os.getenv("PROCESSED_LOG", "processed_files.txt")
SKIPPED_LOG = os.getenv("SKIPPED_LOG", "skipped_non_emails.txt")
ERROR_LOG = os.getenv("ERROR_LOG", "error_log.txt")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini")
MAX_PAGES_PER_PDF = os.getenv("MAX_PAGES_PER_PDF")
MAX_PAGES_PER_PDF = int(MAX_PAGES_PER_PDF) if MAX_PAGES_PER_PDF else None
SPLIT_MULTI_EMAILS = os.getenv("SPLIT_MULTI_EMAILS", "true").lower() == "true"

# CSV columns as specified
CSV_COLUMNS = ["date", "FROM", "TO", "CC", "subject", "attachment", "full_body", "file_name"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(ERROR_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def pdf_to_base64_images(pdf_path: str, max_pages: Optional[int] = None, page_range: Optional[tuple] = None) -> List[str]:
    """
    Convert PDF to base64-encoded images for OpenAI Vision API.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (None = all pages)
        page_range: Optional tuple (start, end) for specific page range (1-indexed)
    
    Returns:
        List of base64-encoded image strings
    """
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=150)
        
        # Apply page range if specified
        if page_range:
            start, end = page_range
            images = images[start-1:end]  # Convert to 0-indexed
        elif max_pages:
            # Limit number of pages to avoid excessive API costs
            images = images[:max_pages]
        
        base64_images = []
        for img in images:
            # Resize if too large to reduce costs
            max_dimension = 2000
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_base64)
        
        return base64_images
    
    except Exception as e:
        logger.error(f"Error converting PDF to images: {pdf_path} - {str(e)}")
        raise


def classify_document(base64_image: str, client: OpenAI) -> bool:
    """
    Classify if a PDF is an email document using a lightweight model.
    
    Args:
        base64_image: Base64-encoded image of the first page
        client: OpenAI client instance
    
    Returns:
        True if document is an email, False otherwise
    """
    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Is this document an email? Look for typical email elements such as:
- From/To/CC fields
- Email addresses
- Subject line
- Email header information
- Reply/Forward indicators
- Email signature

Answer with ONLY "YES" if this is clearly an email, or "NO" if it is not an email (e.g., invoice, report, letter, form, etc.)."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low"  # Use low detail for faster/cheaper classification
                            }
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    
    except Exception as e:
        logger.warning(f"Error during classification, assuming it's an email: {str(e)}")
        return True  # Default to processing if classification fails


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from each page of a PDF.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of text strings, one per page
    """
    try:
        reader = PdfReader(pdf_path)
        page_texts = []
        for page in reader.pages:
            text = page.extract_text()
            page_texts.append(text if text else "")
        return page_texts
    except Exception as e:
        logger.warning(f"Could not extract text from PDF: {str(e)}")
        return []


def detect_email_boundaries_from_text(page_texts: List[str], client: OpenAI) -> List[tuple]:
    """
    Detect email boundaries by analyzing text patterns from all pages.
    Uses rule-based detection for accuracy.
    
    Args:
        page_texts: List of text strings, one per page
        client: OpenAI client instance
    
    Returns:
        List of tuples (start_page, end_page) for each email (1-indexed)
    """
    try:
        num_pages = len(page_texts)
        
        if num_pages == 0:
            return [(1, 1)]
        
        if num_pages == 1:
            return [(1, 1)]
        
        # Detect email start pages using pattern matching
        email_start_pages = []
        
        for i, text in enumerate(page_texts, 1):
            # Look for strong email header patterns
            has_from = bool(re.search(r'^\s*From:\s*\S', text, re.IGNORECASE | re.MULTILINE))
            has_to = bool(re.search(r'^\s*To:\s*\S', text, re.IGNORECASE | re.MULTILINE))
            has_subject = bool(re.search(r'^\s*Subject:\s*\S', text, re.IGNORECASE | re.MULTILINE))
            has_sent = bool(re.search(r'^\s*(Date|Sent):\s*\S', text, re.IGNORECASE | re.MULTILINE))
            
            # Also check for "Original Message" dividers which often indicate forwarded/replied emails
            has_original_msg = bool(re.search(r'-----\s*Original Message\s*-----', text, re.IGNORECASE))
            
            # A page is likely an email start if it has at least 3 of the 4 main headers
            header_count = sum([has_from, has_to, has_subject, has_sent])
            
            if header_count >= 3:
                email_start_pages.append(i)
            elif has_original_msg and header_count >= 2:
                # Original message with some headers is also a good indicator
                email_start_pages.append(i)
        
        # If we found no email starts, treat as single email
        if not email_start_pages:
            logger.warning("No email boundaries detected, treating as single email")
            return [(1, num_pages)]
        
        # Create boundaries from start pages
        boundaries = []
        for i in range(len(email_start_pages)):
            start = email_start_pages[i]
            # End is one page before next email start, or last page
            end = email_start_pages[i + 1] - 1 if i + 1 < len(email_start_pages) else num_pages
            boundaries.append((start, end))
        
        logger.info(f"Rule-based detection: found {len(boundaries)} emails (start pages: {email_start_pages[:10]}{'...' if len(email_start_pages) > 10 else ''})")
        
        return boundaries
    
    except Exception as e:
        logger.warning(f"Error detecting email boundaries from text: {str(e)}")
        return [(1, len(page_texts) if page_texts else 1)]


def extract_email_data_with_openai(base64_images: List[str], file_name: str, client: OpenAI) -> Dict:
    """
    Use OpenAI Vision API to extract structured email data from PDF images.
    
    Args:
        base64_images: List of base64-encoded images
        file_name: Original filename for reference
        client: OpenAI client instance
    
    Returns:
        Dictionary with extracted email data
    """
    # Construct messages with images
    content = [
        {
            "type": "text",
            "text": """You are analyzing an email PDF. Extract the following information and return it as a JSON object:

{
  "date": "The date of the email (format as found)",
  "FROM": "Sender email address or name",
  "TO": "Recipient email addresses (comma-separated if multiple)",
  "CC": "CC recipients (comma-separated if multiple, or empty string if none)",
  "subject": "Email subject line",
  "attachment": "List any attachments mentioned (comma-separated, or empty string if none)",
  "full_body": "The complete email body text"
}

If any field cannot be found, use an empty string. Be thorough in extracting the full email body."""
        }
    ]
    
    # Add images to content
    for img_base64 in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high"
            }
        })
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=4096,
        response_format={"type": "json_object"}
    )
    
    # Parse response
    result = json.loads(response.choices[0].message.content)
    
    # Add filename to result
    result["file_name"] = file_name
    
    # Ensure all required fields exist
    for col in CSV_COLUMNS:
        if col not in result:
            result[col] = ""
    
    return result


def process_pdf_with_retry(pdf_path: str, client: OpenAI, max_retries: int = MAX_RETRIES) -> Optional[List[Dict]]:
    """
    Process a single PDF with exponential backoff retry logic.
    Can return multiple emails if the PDF contains multiple email messages.
    
    Args:
        pdf_path: Path to PDF file
        client: OpenAI client instance
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of extracted email data dictionaries, None if failed, or [{"SKIP": True}] if not an email
    """
    file_name = os.path.basename(pdf_path)
    
    for attempt in range(max_retries):
        try:
            # First, extract text from all pages for accurate boundary detection
            page_texts = extract_text_from_pdf(pdf_path)
            num_pages = len(page_texts) if page_texts else 0
            
            # Apply MAX_PAGES_PER_PDF limit if set
            if MAX_PAGES_PER_PDF and num_pages > MAX_PAGES_PER_PDF:
                page_texts = page_texts[:MAX_PAGES_PER_PDF]
                num_pages = MAX_PAGES_PER_PDF
                logger.info(f"Limiting {file_name} to {MAX_PAGES_PER_PDF} pages")
            
            if num_pages == 0:
                logger.error(f"No pages found in {file_name}")
                return None
            
            # Convert first page to image for classification
            first_page_images = pdf_to_base64_images(pdf_path, max_pages=1)
            
            if not first_page_images:
                logger.error(f"Could not convert first page of {file_name}")
                return None
            
            # Classify if this is an email document
            is_email = classify_document(first_page_images[0], client)
            
            if not is_email:
                logger.info(f"Skipping non-email document: {file_name}")
                return [{"SKIP": True}]
            
            # Detect email boundaries using text analysis (checks ALL pages)
            if SPLIT_MULTI_EMAILS and num_pages > 1:
                boundaries = detect_email_boundaries_from_text(page_texts, client)
                logger.info(f"Detected {len(boundaries)} email(s) in {file_name} ({num_pages} pages)")
            else:
                # Treat as single email
                boundaries = [(1, num_pages)]
            
            # Extract data for each email
            email_results = []
            for idx, (start_page, end_page) in enumerate(boundaries, 1):
                # Convert only the pages needed for this email
                email_images = pdf_to_base64_images(pdf_path, page_range=(start_page, end_page))
                
                # Create identifier for this email
                if len(boundaries) > 1:
                    email_identifier = f"{file_name} [email {idx}/{len(boundaries)}, pages {start_page}-{end_page}]"
                else:
                    email_identifier = file_name
                
                # Extract data
                email_data = extract_email_data_with_openai(email_images, email_identifier, client)
                email_results.append(email_data)
            
            return email_results
        
        except RateLimitError as e:
            wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
            logger.warning(f"Rate limit hit for {file_name}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)
            
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} retries due to rate limit: {file_name}")
                return None
        
        except APIError as e:
            logger.error(f"OpenAI API error processing {file_name}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            return None
    
    return None


def load_processed_files() -> set:
    """Load the set of already processed file names."""
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_processed_file(file_name: str):
    """Append a processed file name to the log."""
    with open(PROCESSED_LOG, "a") as f:
        f.write(f"{file_name}\n")


def load_skipped_files() -> set:
    """Load the set of already skipped file names."""
    if os.path.exists(SKIPPED_LOG):
        with open(SKIPPED_LOG, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_skipped_file(file_name: str):
    """Append a skipped file name to the log."""
    with open(SKIPPED_LOG, "a") as f:
        f.write(f"{file_name}\n")


def save_to_csv(data: Dict, output_path: str):
    """
    Save extracted email data to CSV, appending if file exists.
    
    Args:
        data: Dictionary with email data
        output_path: Path to output CSV file
    """
    df = pd.DataFrame([data], columns=CSV_COLUMNS)
    
    # Append to existing CSV or create new one
    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, mode="w", header=True, index=False)


def get_pdf_files(directory: str) -> List[str]:
    """Get all PDF files in the directory."""
    pdf_files = list(Path(directory).glob("*.pdf"))
    return sorted([str(f) for f in pdf_files])


def main():
    """Main execution function."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables!")
        print("Error: Please set OPENAI_API_KEY in your environment or .env file")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Get all PDF files
    pdf_files = get_pdf_files(INPUT_DIR)
    
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Load already processed and skipped files
    processed_files = load_processed_files()
    skipped_files = load_skipped_files()
    already_handled = processed_files | skipped_files
    
    # Filter out already processed/skipped files
    remaining_files = [f for f in pdf_files if os.path.basename(f) not in already_handled]
    
    if processed_files:
        print(f"Skipping {len(processed_files)} already processed email files")
    if skipped_files:
        print(f"Skipping {len(skipped_files)} previously identified non-email files")
    
    if not remaining_files:
        print("All files have been processed!")
        return
    
    print(f"Processing {len(remaining_files)} files...")
    print(f"Output will be saved to: {OUTPUT_CSV}")
    
    # Statistics
    success_count = 0  # Number of emails successfully extracted
    skipped_count = 0  # Number of non-email documents skipped
    failure_count = 0  # Number of PDFs that failed processing
    pdf_processed_count = 0  # Number of PDFs successfully processed
    
    # Process each PDF with progress bar
    for pdf_path in tqdm(remaining_files, desc="Processing PDFs", unit="file"):
        file_name = os.path.basename(pdf_path)
        
        try:
            # Process PDF (returns list of emails)
            email_results = process_pdf_with_retry(pdf_path, client)
            
            if email_results is None:
                # Processing failed
                failure_count += 1
                logger.error(f"Failed to process: {file_name}")
            elif email_results and "SKIP" in email_results[0]:
                # Document is not an email, skip it
                save_skipped_file(file_name)
                skipped_count += 1
            elif email_results:
                # Save each email to CSV
                for email_data in email_results:
                    save_to_csv(email_data, OUTPUT_CSV)
                    success_count += 1
                
                # Mark PDF as processed
                save_processed_file(file_name)
                pdf_processed_count += 1
                
                if len(email_results) > 1:
                    logger.info(f"Extracted {len(email_results)} emails from {file_name}")
            else:
                failure_count += 1
                logger.error(f"Failed to process: {file_name}")
        
        except Exception as e:
            failure_count += 1
            logger.error(f"Unexpected error processing {file_name}: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"PDFs processed successfully: {pdf_processed_count}")
    print(f"Emails extracted: {success_count}")
    print(f"Non-email documents skipped: {skipped_count}")
    print(f"Failed PDFs: {failure_count}")
    print(f"Total PDFs: {pdf_processed_count + skipped_count + failure_count}")
    print(f"\nOutput saved to: {OUTPUT_CSV}")
    print(f"Error log: {ERROR_LOG}")
    print(f"Processed files log: {PROCESSED_LOG}")
    print(f"Skipped files log: {SKIPPED_LOG}")


if __name__ == "__main__":
    main()

