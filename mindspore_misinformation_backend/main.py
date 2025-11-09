"""
MindSpore-based Misinformation Detection Backend
===============================================

A comprehensive misinformation detection system using Huawei MindSpore
for text, PDF, and image analysis - designed for Huawei competition.

Author: ShoAI Team
Date: October 2025
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import traceback

# Import our MindSpore modules
from modules.text_analyzer import TextAnalyzer
from modules.pdf_analyzer import PDFAnalyzer
from modules.image_analyzer import ImageAnalyzer
from modules.online_verifier import OnlineFactVerifier, EnhancedTextAnalyzer
from modules.ai_detector import AIContentDetector
from modules.video_deepfake_detector import analyze_video_with_mindspore
from modules.audio_deepfake_detector import analyze_audio_with_mindspore
from modules.source_credibility_checker import check_source_credibility
from modules.fact_checker import check_fact
from modules.headline_analyzer import analyze_headline_vs_content
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize MindSpore analyzers
logger.info("Initializing MindSpore analyzers...")
text_analyzer = TextAnalyzer()
pdf_analyzer = PDFAnalyzer()
image_analyzer = ImageAnalyzer()
ai_detector = AIContentDetector()

# Initialize online verifier
online_verifier = OnlineFactVerifier()
enhanced_analyzer = EnhancedTextAnalyzer(text_analyzer, online_verifier)

logger.info("All analyzers initialized successfully!")

@app.route('/')
def home():
    """API status page"""
    return {
        "status": "running",
        "service": "MindSpore Misinformation Detection Backend",
        "version": "1.0.0",
        "endpoints": {
            "text_analysis": "/api/analyze/text",
            "pdf_analysis": "/api/analyze/pdf",
            "image_analysis": "/api/analyze/image",
            "ai_detection": "/api/detect/ai",
            "health_check": "/api/health"
        }
    }

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mindspore_version": "2.7.0",
        "analyzers": {
            "text": text_analyzer.is_ready(),
            "pdf": pdf_analyzer.is_ready(),
            "ai_detector": ai_detector.is_ready(),
            "image": image_analyzer.is_ready()
        }
    }

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text for misinformation using MindSpore"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        if len(text) < 10:
            return jsonify({"error": "Text too short for analysis"}), 400
        
        logger.info(f"Analyzing text of length: {len(text)}")
        result = text_analyzer.analyze(text)
        
        return jsonify({
            "success": True,
            "analysis": result,
            "service": "MindSpore Text Analyzer"
        })
        
    except Exception as e:
        logger.error(f"Text analysis error: {str(e)}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/detect/ai', methods=['POST'])
def detect_ai():
    """Detect if text is AI-generated or human-written using MindSpore"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        if len(text) < 20:
            return jsonify({"error": "Text too short for AI detection (minimum 20 characters)"}), 400
        
        logger.info(f"AI detection for text of length: {len(text)}")
        result = ai_detector.detect(text)
        
        return jsonify({
            "success": True,
            "detection": result,
            "service": "MindSpore AI Content Detector"
        })
        
    except Exception as e:
        logger.error(f"AI detection error: {str(e)}")
        return jsonify({
            "error": f"AI detection failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/analyze/pdf', methods=['POST'])
def analyze_pdf():
    """Universal File Analysis: Extract content from any file + MindSpore analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Accept all file types: PDF, TXT, DOC, DOCX, images with text, etc.
        filename = file.filename.lower()
        supported_extensions = ['.pdf', '.txt', '.doc', '.docx', '.rtf', '.odt', 
                              '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        if not any(filename.endswith(ext) for ext in supported_extensions):
            return jsonify({"error": f"Unsupported file type. Supported: {', '.join(supported_extensions)}"}), 400
        
        logger.info(f"Universal file analysis: {file.filename}")
        
        # Step 1: Extract text content from file (universal handler)
        file_ext = filename.split('.')[-1]
        
        if file_ext in ['txt', 'rtf', 'doc', 'docx', 'odt']:
            # Plain text and document files - direct read
            try:
                extracted_text = file.read().decode('utf-8', errors='ignore').strip()
                pdf_content = {'text': extracted_text, 'pages': 1}
                logger.info(f"Successfully read {file_ext.upper()} file: {len(extracted_text)} characters")
            except Exception as e:
                logger.error(f"Error reading {file_ext.upper()} file: {str(e)}")
                extracted_text = ""
                pdf_content = {'text': '', 'pages': 0}
        elif file_ext in ['pdf']:
            # PDF files - use PDF analyzer
            pdf_content = pdf_analyzer._extract_pdf_content(file)
            extracted_text = pdf_content.get('text', '').strip()
            logger.info(f"Successfully extracted PDF: {len(extracted_text)} characters")
        elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            # Image files - extract any embedded text or metadata
            try:
                # Try to extract text from image (OCR or metadata)
                extracted_text = file.read().decode('utf-8', errors='ignore').strip()
                # If no readable text, provide a default message
                if not extracted_text or len(extracted_text) < 10:
                    extracted_text = f"Image file uploaded: {file.filename}. Universal File Detector supports text analysis from documents. For image manipulation detection, please use the Image Detector tool instead."
                pdf_content = {'text': extracted_text, 'pages': 1}
                logger.info(f"Processed {file_ext.upper()} image file")
            except Exception as e:
                logger.error(f"Error processing {file_ext.upper()} image: {str(e)}")
                extracted_text = f"Image file uploaded: {file.filename}. For image analysis, please use the Image Detector tool."
                pdf_content = {'text': extracted_text, 'pages': 1}
        else:
            # Unknown format - still accept it
            try:
                extracted_text = file.read().decode('utf-8', errors='ignore').strip()
                pdf_content = {'text': extracted_text, 'pages': 1}
                logger.info(f"Attempting to read unknown format: {file_ext.upper()}")
            except:
                extracted_text = ""
                pdf_content = {'text': '', 'pages': 0}
        
        if not extracted_text:
            return jsonify({
                "success": True,
                "analysis": {
                    "judgment": "No Data",
                    "explanation": f"Could not extract readable text from this {file_ext.upper()} file. This may be:\n\n- An image-based file (scanned document)\n- A password-protected file\n- A corrupted or unsupported file format\n\nTip: Try converting to plain text format first."
                },
                "service": "MindSpore Universal File Analyzer"
            })
        
        # Step 2: Run comprehensive fact-checking on extracted text
        if len(extracted_text) < 10:
            return jsonify({
                "success": True,
                "analysis": {
                    "judgment": "No Data", 
                    "explanation": f"PDF text too short for analysis ({len(extracted_text)} characters).\n\nExtracted content: \"{extracted_text[:50]}...\"\n\nNote: Need at least 10 characters for reliable fact-checking."
                },
                "service": "MindSpore Universal File Analyzer"
            })
        
        logger.info(f"Extracted {len(extracted_text)} characters from {file_ext.upper()} file. Running MindSpore analysis...")
        
        # Step 3: Optimized analysis strategy based on content length
        explanation_parts = []  # Initialize explanation_parts for all cases
        
        if len(extracted_text) > 2000:
            # For long texts, use faster local analysis only
            logger.info("Using fast local analysis for long PDF content")
            local_result = text_analyzer.analyze(extracted_text[:1500])  # Limit to first 1500 chars for speed
            judgment = local_result.get('judgment', 'No Data')
            confidence = local_result.get('confidence', 0)
            
            # Simple explanation for fast processing
            explanation_parts = [
                f"Quick Analysis: {judgment} (Confidence: {confidence:.1f}%)",
                f"Content Length: {len(extracted_text)} characters",
                "Fast processing mode used for large PDF content"
            ]
            
            recommendation = f"This document appears to be {judgment.lower()}. For detailed fact-checking of specific claims, consider analyzing shorter excerpts."
            breakdown = {}
            
        else:
            # For shorter texts, use enhanced analysis with online verification
            logger.info("Using enhanced analysis for moderate PDF content")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                enhanced_result = loop.run_until_complete(enhanced_analyzer.analyze_comprehensive(extracted_text))
                
                # Transform enhanced result to expected format
                local_analysis = enhanced_result.get('mindspore_analysis', {})
                online_analysis = enhanced_result.get('online_verification', {})
                
                # Extract judgment from local analysis
                judgment = local_analysis.get('judgment', 'No Data')
                
                # Create comprehensive explanation
                recommendation = enhanced_result.get('recommendation', 'No recommendation available')
                breakdown = enhanced_result.get('detailed_breakdown', {})
            finally:
                loop.close()
            
        # Add enhanced details only for detailed analysis (shorter texts)
        if len(extracted_text) <= 2000 and breakdown:
            # Build enhanced explanation_parts for detailed analysis
            enhanced_parts = [f"Analysis Result: {judgment}"]
            
            # Only add recommendation if it's meaningful
            if recommendation and recommendation != 'No recommendation available' and len(recommendation.strip()) > 10:
                enhanced_parts.append(f"Assessment: {recommendation}")
            
            # Add pattern analysis details only if they contain meaningful data
            pattern_analysis = breakdown.get('pattern_analysis', {})
            has_pattern_data = False
            pattern_details = []
            
            if pattern_analysis.get('misinformation_indicators', 0) > 0:
                pattern_details.append(f"Misinformation indicators detected: {pattern_analysis['misinformation_indicators']}")
                has_pattern_data = True
                
            emotional_level = pattern_analysis.get('emotional_language', 0)
            factual_level = pattern_analysis.get('factual_language', 0)
            
            if emotional_level > 0.1 or factual_level > 0.1:
                pattern_details.append(f"Emotional language level: {emotional_level:.1f}/1.0")
                pattern_details.append(f"Factual language score: {factual_level:.1f}/1.0")
                has_pattern_data = True
            
            if has_pattern_data:
                enhanced_parts.append("Pattern Analysis:")
                enhanced_parts.extend(pattern_details)
            
            # Add source verification details only if there's actual verification data
            source_verification = breakdown.get('source_verification', {})
            has_source_data = False
            source_details = []
            
            high_cred = source_verification.get('high_credibility_sources', 0)
            supporting = source_verification.get('supporting_evidence', 0) 
            contradictory = source_verification.get('contradictory_evidence', 0)
            
            if high_cred > 0 or supporting > 0 or contradictory > 0:
                if high_cred > 0:
                    source_details.append(f"High-credibility sources found: {high_cred}")
                if supporting > 0:
                    source_details.append(f"Supporting evidence: {supporting}")
                if contradictory > 0:
                    source_details.append(f"Contradictory evidence: {contradictory}")
                has_source_data = True
            
            if has_source_data:
                enhanced_parts.append("Online Verification:")
                enhanced_parts.extend(source_details)
            
            # Replace explanation_parts with enhanced version
            explanation_parts = enhanced_parts
        
        # Add minimal PDF context for all cases
        explanation_parts.append(f"\nFile: {file.filename} ({len(extracted_text)} characters extracted)")
        
        final_explanation = "\n".join(explanation_parts)
        
        return jsonify({
            "success": True,
            "analysis": {
                "judgment": judgment,
                "explanation": final_explanation
            },
            "service": "MindSpore PDF Analysis (Optimized)"
        })
        
    except Exception as e:
        logger.error(f"Enhanced PDF analysis error: {str(e)}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyze image for manipulation using MindSpore"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if it's an image
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({"error": "File must be an image"}), 400
        
        logger.info(f"Analyzing image: {file.filename}")
        result = image_analyzer.analyze(file)
        
        return jsonify({
            "success": True,
            "analysis": result,
            "service": "MindSpore Image Analyzer"
        })
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/analyze/enhanced', methods=['POST'])
def analyze_enhanced():
    """Enhanced analysis combining MindSpore + Online Verification"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        if len(text) < 10:
            return jsonify({"error": "Text too short for analysis"}), 400
        
        logger.info(f"Enhanced analysis for text of length: {len(text)}")
        
        # Run enhanced analysis with online verification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(enhanced_analyzer.analyze_comprehensive(text))
        finally:
            loop.close()
        
        return jsonify({
            "success": True,
            "enhanced_analysis": result,
            "service": "MindSpore + Online Verification"
        })
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {str(e)}")
        return jsonify({
            "error": f"Enhanced analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/analyze/video-deepfake', methods=['POST'])
def analyze_video_deepfake():
    """Analyze video for deepfake/manipulation using MindSpore"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['file']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Save temporary file
        import tempfile
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(video_path)
        
        logger.info(f"Analyzing video: {video_file.filename}")
        
        try:
            # Analyze video with MindSpore
            result = analyze_video_with_mindspore(video_path)
            
            return jsonify({
                "success": True,
                "result": result,
                "service": "MindSpore Video Deepfake Detector"
            })
        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        
    except Exception as e:
        logger.error(f"Video deepfake detection error: {str(e)}")
        return jsonify({
            "error": f"Video analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/analyze/audio-deepfake', methods=['POST'])
def analyze_audio_deepfake():
    """Analyze audio for deepfake/synthesis using MindSpore"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Save temporary file
        import tempfile
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, audio_file.filename)
        audio_file.save(audio_path)
        
        logger.info(f"Analyzing audio: {audio_file.filename}")
        
        try:
            # Analyze audio with MindSpore
            result = analyze_audio_with_mindspore(audio_path)
            
            return jsonify({
                "success": True,
                "result": result,
                "service": "MindSpore Audio Deepfake Detector"
            })
        finally:
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        
    except Exception as e:
        logger.error(f"Audio deepfake detection error: {str(e)}")
        return jsonify({
            "error": f"Audio analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/check/source-credibility', methods=['POST'])
def check_source():
    """Check source credibility using MindSpore"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400
        
        url = data['url'].strip()
        if not url:
            return jsonify({"error": "Empty URL"}), 400
        
        logger.info(f"Checking source credibility for: {url}")
        
        # Check source with MindSpore
        result = check_source_credibility(url)
        
        return jsonify({
            "success": True,
            "result": result,
            "service": "MindSpore Source Credibility Checker"
        })
        
    except Exception as e:
        logger.error(f"Source credibility check error: {str(e)}")
        return jsonify({
            "error": f"Source check failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/check/fact', methods=['POST'])
def check_fact_claim():
    """Cross-reference fact claim using MindSpore with real-time web verification"""
    try:
        data = request.get_json()
        if not data or 'claim' not in data:
            return jsonify({"error": "No claim provided"}), 400
        
        claim = data['claim'].strip()
        if len(claim) < 10:
            return jsonify({"error": "Claim too short"}), 400
        
        # Check if real-time verification is requested (default: True)
        use_realtime = data.get('use_realtime', True)
        
        logger.info(f"Fact-checking claim of length: {len(claim)} (realtime: {use_realtime})")
        
        # Check fact with MindSpore + real-time verification
        result = check_fact(claim, use_realtime=use_realtime)
        
        return jsonify({
            "success": True,
            "result": result,
            "service": "MindSpore Fact Cross-Reference Tool with Real-time Verification"
        })
        
    except Exception as e:
        logger.error(f"Fact check error: {str(e)}")
        return jsonify({
            "error": f"Fact check failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/check/fact/realtime', methods=['POST'])
def verify_claim_realtime():
    """Direct real-time web verification of claims (bypasses ML analysis)"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        if len(text) < 5:
            return jsonify({"error": "Text too short"}), 400
        
        logger.info(f"Real-time verification of text length: {len(text)}")
        
        # Import and use real-time fact checker directly
        from modules.real_time_fact_checker import get_fact_checker
        rt_checker = get_fact_checker()
        result = rt_checker.verify_claims(text)
        
        return jsonify({
            "success": True,
            "result": result,
            "service": "Real-time Fact Verification (Web Search)"
        })
        
    except Exception as e:
        logger.error(f"Real-time verification error: {str(e)}")
        return jsonify({
            "error": f"Real-time verification failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/analyze/headline', methods=['POST'])
def analyze_headline():
    """Analyze headline vs content for clickbait using MindSpore"""
    try:
        data = request.get_json()
        if not data or 'headline' not in data or 'content' not in data:
            return jsonify({"error": "Both headline and content required"}), 400
        
        headline = data['headline'].strip()
        content = data['content'].strip()
        
        if len(headline) < 5:
            return jsonify({"error": "Headline too short"}), 400
        if len(content) < 20:
            return jsonify({"error": "Content too short"}), 400
        
        logger.info(f"Analyzing headline vs content: {len(headline)} / {len(content)} chars")
        
        # Analyze with MindSpore
        result = analyze_headline_vs_content(headline, content)
        
        return jsonify({
            "success": True,
            "result": result,
            "service": "MindSpore Headline vs Content Analyzer"
        })
        
    except Exception as e:
        logger.error(f"Headline analysis error: {str(e)}")
        return jsonify({
            "error": f"Headline analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting MindSpore Misinformation Detection Backend...")
    logger.info("Ready to serve misinformation detection requests!")
    
    # Run on localhost only for security
    app.run(
        host='127.0.0.1',  # Localhost only - no external access
        port=5000,
        debug=False,  # Disable debug mode for stability
        threaded=True
    )
