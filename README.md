# mindspore-misinformation-detector
AI-powered misinformation detection using MindSpore 2.7.0. Features: 9 detection tools, real-time fact-checking (Wikipedia, Snopes, FactCheck.org), medical claim detection, deepfake detection, hybrid ML + web verification. Built for education and media literacy research.

## What This Does

Detects:
- Fake news and misinformation
- Deepfake videos and AI-generated images
- False medical/health claims
- Death/alive claims (verifies against Wikipedia)
- Clickbait headlines
- Unreliable sources

---

## Quick Start

```bash
# 1. Install system packages (REQUIRED!)
sudo apt update
sudo apt install -y python3.10 python3.10-venv tesseract-ocr build-essential

# 2. Setup virtual environment
python3.10 -m venv mindenv310
source mindenv310/bin/activate

# 3. Install Python dependencies
cd mindspore_misinformation_backend
pip install -r requirements.txt
cd ..

# 4. Run the system
chmod +x start_system.sh
./start_system.sh
```

Open: http://localhost:3000/phone_base_model.html

---

## What You Need

REQUIRED:
- Ubuntu/Debian Linux (or WSL2)
- Python 3.10
- Internet connection

INSTALL:
```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv tesseract-ocr build-essential
```

HARDWARE:
- 8GB RAM (16GB better)
- 5GB disk space

---

## Installation

### Step 1: Install System Packages

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv tesseract-ocr build-essential
```

### Step 2: Create Virtual Environment

```bash
cd /path/to/project
python3.10 -m venv mindenv310
source mindenv310/bin/activate
```

Note: Run `source mindenv310/bin/activate` every time you open a new terminal.

### Step 3: Install Dependencies

```bash
cd mindspore_misinformation_backend
pip install -r requirements.txt
cd ..
```

This takes 5-10 minutes and downloads ~2-3GB.

### Step 4: Make Scripts Executable

```bash
chmod +x start_system.sh
chmod +x watchdog.sh
```

### Step 5: Start System

```bash
./start_system.sh
```

You should see:
```
Starting MindSpore Misinformation Detection System...
Backend starting on port 5000...
Frontend starting on port 3000...
System ready!
```

### Step 6: Open Application

Go to: http://localhost:3000/phone_base_model.html

---

## Project Structure

```
mindspore-detection-system/
├── phone_base_model.html          # Main web interface
├── start_system.sh                # Startup script
├── mindenv310/                    # Python virtual environment
└── mindspore_misinformation_backend/
    ├── main.py                    # Backend server
    ├── requirements.txt           # Dependencies
    └── modules/
        ├── text_analyzer.py
        ├── real_time_fact_checker.py
        ├── image_analyzer.py
        ├── video_deepfake_detector.py
        ├── fact_checker.py
        ├── headline_analyzer.py
        └── ... (more tools)
```

---

## Features

9 DETECTION TOOLS:

1. Text Analyzer - Fake news, clickbait, emotional manipulation
2. Image Detector - Photo manipulation, AI-generated images
3. Video Deepfake Detector - Deepfake detection, frame analysis
4. Source Credibility Checker - Domain reputation, SSL verification
5. Fact Cross-Reference Tool - Web verification, death/alive claims, medical claims
6. Headline Analyzer - Clickbait detection, claim verification
7. PDF Analyzer - Text extraction with OCR
8. AI Article Detector - AI-generated text detection
9. Bubble Insight - Clipboard analysis with floating UI

REAL-TIME FACT CHECKING:
- Wikipedia verification
- DuckDuckGo search
- Snopes, FactCheck.org, PolitiFact, Reuters
- 100+ medical/health claim patterns
- Death/alive claim verification

---

## Usage

### Start System

```bash
source mindenv310/bin/activate
./start_system.sh
```

### Stop System

```bash
pkill -f "python.*main.py"
pkill -f "http.server 3000"
```

### View Logs

```bash
tail -f mindspore_misinformation_backend/backend.log
```

---

## Common Issues

PORT ALREADY IN USE:
```bash
lsof -i :5000
kill -9 <PID>
./start_system.sh
```

MINDSPORE NOT FOUND:
```bash
source mindenv310/bin/activate
pip install --upgrade mindspore==2.7.0
```

PERMISSION DENIED:
```bash
chmod +x start_system.sh
```

WEB VERIFICATION FAILED:
```bash
ping -c 3 wikipedia.org
```

---

## How It Works

HYBRID DETECTION (60% AI + 40% WEB):

- MindSpore AI analyzes patterns in text, images, videos
- Real-Time Web verifies claims against trusted sources
- Combined Score gives final verdict

EXAMPLE:
- Text: "Person X died" → AI detects death claim → Web checks Wikipedia → If alive, flags as FALSE

DETECTION SPEED:
- Text: 1-2 seconds
- Image: 2-3 seconds  
- Video: 5-10 seconds
- Fact-check: 2-4 seconds

---

## Changelog

### November 2025 - Major Updates

CODE CLEANUP:
- Removed all emojis from backend and frontend code
- Simplified comments and docstrings

UI IMPROVEMENTS:
- Redesigned tools selector: scrollable list → button grid with icons
- Enhanced intro eye logo: better design, smoother animation
- Fixed statistics page text collision

BUG FIXES:
- Fixed duplicate death claim warnings (added deduplication)
- Fixed "Facts Checked" statistics counter not updating

NEW FEATURES:
- Added 100+ medical/health claim detection patterns (cancer, COVID, diabetes, etc.)
- Added alive claim detection (opposite of death claims)
- Enhanced real-time fact checker across 6 tools

---

## Tech Stack

BACKEND:
- MindSpore 2.7.0
- Flask
- Python 3.10

FRONTEND:
- HTML5, CSS3, JavaScript

PROCESSING:
- OpenCV (images/videos)
- Pillow (images)
- PyPDF2, pdfplumber, pytesseract (PDFs)

WEB VERIFICATION:
- requests, aiohttp

---

## Dependencies

Main packages (see `requirements.txt` for full list):
```
mindspore>=2.7.0
flask>=2.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
PyPDF2>=3.0.0
pdfplumber>=0.9.0
pytesseract>=0.3.10
requests>=2.28.0
```

Total download size: ~2-3GB

---

## License

Educational project using MindSpore framework.

**Last Updated:** November 2025  
**Version:** 1.0  
**MindSpore:** 2.7.0
