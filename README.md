# OpenAI PDF Email Processor

A production-ready Python script that uses OpenAI's GPT-4 Vision API to extract structured email data from thousands of PDF files and export to CSV format.

## Features

- **Smart Classification**: Automatically detects and skips non-email documents (invoices, reports, etc.) to save API costs
- **GPT-4 Vision Integration**: Uses advanced vision models to extract email data from PDFs
- **Batch Processing**: Handles thousands of files with progress tracking
- **Resume Capability**: Automatically skips already processed and non-email files
- **Error Handling**: Robust retry logic with exponential backoff for rate limits
- **Cost Optimization**: Uses cheaper model for classification, limits pages, and resizes images
- **Detailed Logging**: Tracks errors, processed files, and skipped non-emails

## Installation

1. Install system dependencies (required for pdf2image):
   - **macOS**: `brew install poppler`
   - **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
   - **Windows**: Download poppler from [https://github.com/oschwartz10612/poppler-windows/releases/](https://github.com/oschwartz10612/poppler-windows/releases/)

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

3. Optional: Configure other settings in `.env`:
   - `OPENAI_MODEL`: Model to use for extraction (default: gpt-4o)
   - `CLASSIFIER_MODEL`: Model to use for classification (default: gpt-4o-mini, cheaper)
   - `INPUT_DIR`: Directory containing PDF files (default: current directory)
   - `OUTPUT_CSV`: Output CSV filename (default: email_data.csv)
   - `MAX_RETRIES`: Number of retry attempts for failed API calls (default: 3)

## Usage

1. Place your PDF files in the input directory (or current directory by default)

2. Run the script:
```bash
python main.py
```

3. The script will:
   - Scan for all PDF files
   - Skip any previously processed or non-email files
   - Classify each document (is it an email?)
   - Skip non-email documents (invoices, reports, etc.)
   - Extract email data from emails using GPT-4 Vision
   - Save results incrementally to CSV
   - Show progress bar and statistics

## Output

### CSV File (`email_data.csv`)
Contains the following columns:
- `date`: Email date
- `FROM`: Sender email/name
- `TO`: Recipient email(s)
- `CC`: CC recipients
- `subject`: Email subject line
- `attachment`: Mentioned attachments
- `full_body`: Complete email body text
- `file_name`: Original PDF filename

### Log Files
- `processed_files.txt`: List of successfully processed email files
- `skipped_non_emails.txt`: List of files identified as non-email documents
- `error_log.txt`: Detailed error logs for troubleshooting

## Cost Considerations

- GPT-4 Vision pricing: ~$0.01-0.05 per PDF (varies by page count and resolution)
- For 1000 PDFs: Estimate $10-50 in API costs
- The script automatically:
  - Limits to first 5 pages per PDF
  - Resizes large images to 2000px max dimension
  - Uses efficient encoding

## Document Classification

The script includes intelligent document classification to save API costs:

1. **First Pass - Classification**: Each PDF is quickly analyzed using GPT-4o-mini (cheaper, faster model) to determine if it's an email
2. **Documents Skipped**: Non-email documents like invoices, reports, forms, letters, etc. are automatically skipped
3. **Cost Savings**: Classification costs ~$0.001 per PDF vs ~$0.01-0.05 for full extraction
4. **Tracked Separately**: Skipped files are logged in `skipped_non_emails.txt` and won't be re-checked

This feature is especially valuable if your PDF collection contains mixed document types.

## Resume Processing

If the script is interrupted, simply run it again. It will automatically skip:
- Files already processed as emails (tracked in `processed_files.txt`)
- Files previously identified as non-emails (tracked in `skipped_non_emails.txt`)

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Ensure you've created a `.env` file with your API key

**"No PDF files found"**
- Check the `INPUT_DIR` path in `.env`
- Ensure PDF files have `.pdf` extension

**Rate limit errors**
- The script automatically retries with exponential backoff
- Consider reducing processing speed or upgrading your OpenAI plan

**Import errors**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

## Example

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
# Edit .env to add OPENAI_API_KEY

# Run
python main.py

# Output
Found 1523 PDF files
Processing 1523 files...
Output will be saved to: email_data.csv
Processing PDFs: 100%|████████████████| 1523/1523 [2:34:12<00:00,  6.07s/file]

============================================================
Processing Complete!
============================================================
Successfully processed emails: 1340
Skipped non-email documents: 175
Failed: 8
Total processed: 1523

Output saved to: email_data.csv
Skipped files log: skipped_non_emails.txt
```

