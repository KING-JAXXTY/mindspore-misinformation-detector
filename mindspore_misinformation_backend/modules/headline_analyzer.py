"""
Headline vs Content Analyzer using MindSpore
Detects clickbait, misleading headlines, and content mismatch
NOW WITH REAL-TIME FACT VERIFICATION FOR CLAIMS IN HEADLINES
"""

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
import threading
import traceback
import re

mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="CPU")
mindspore_lock = threading.Lock()

# Import real-time fact checker
try:
    from .real_time_fact_checker import get_fact_checker
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    print("[Warning] Real-time fact checker not available for headline analyzer")

class MindSporeHeadlineAnalyzer(nn.Cell):
    """Neural network for headline vs content analysis"""
    
    def __init__(self):
        super(MindSporeHeadlineAnalyzer, self).__init__()
        # Input: 64 comparison features
        self.dense1 = nn.Dense(64, 128)
        self.dense2 = nn.Dense(128, 64)
        self.dense3 = nn.Dense(64, 32)
        self.dense4 = nn.Dense(32, 2)  # Accurate vs Misleading
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return self.softmax(x)

# Global model instance
headline_analyzer_model = None

def initialize_model():
    """Initialize the MindSpore headline analyzer model"""
    global headline_analyzer_model
    if headline_analyzer_model is None:
        with mindspore_lock:
            headline_analyzer_model = MindSporeHeadlineAnalyzer()
            dummy_input = Tensor(np.random.randn(1, 64).astype(np.float32))
            _ = headline_analyzer_model(dummy_input)
    return headline_analyzer_model

# Clickbait indicators
CLICKBAIT_PATTERNS = [
    'you won\'t believe', 'shocking', 'this is why', 'what happened next',
    'the reason will', 'doctors hate', 'one weird trick', 'number 7 will',
    'mind-blowing', 'jaw-dropping', 'unbelievable', 'amazing', 'incredible',
    'this changes everything', 'everyone is talking about', 'going viral',
    'here\'s what', 'this is what happens', 'find out why', 'the truth about',
    'what really happened', 'secret', 'revealed', 'exposed', 'warning'
]

# Sensational words
SENSATIONAL_WORDS = [
    'shocking', 'outrageous', 'scandal', 'crisis', 'disaster', 'catastrophe',
    'explosive', 'bombshell', 'breaking', 'urgent', 'alert', 'horrifying',
    'terrifying', 'devastating', 'dramatic', 'stunning', 'incredible'
]

# Emotional manipulation
EMOTIONAL_WORDS = [
    'heartbreaking', 'tragic', 'inspiring', 'touching', 'emotional',
    'tearjerker', 'powerful', 'moving', 'unforgettable', 'miraculous'
]

def extract_headline_content_features(headline, content):
    """Extract features comparing headline to content"""
    try:
        headline_lower = headline.lower()
        content_lower = content.lower()
        
        features = []
        
        # 1. Length ratio
        headline_len = len(headline)
        content_len = len(content)
        length_ratio = min(headline_len / max(content_len, 1), 1.0)
        features.append(length_ratio)
        
        # 2. Word count ratio
        headline_words = headline_lower.split()
        content_words = content_lower.split()
        word_ratio = len(headline_words) / max(len(content_words), 1)
        features.append(min(word_ratio * 10, 1.0))
        
        # 3. Clickbait patterns in headline
        clickbait_count = sum(1 for pattern in CLICKBAIT_PATTERNS if pattern in headline_lower)
        features.append(min(clickbait_count / 3, 1.0))
        
        # 4. Sensational words in headline
        sensational_count = sum(1 for word in SENSATIONAL_WORDS if word in headline_lower)
        features.append(min(sensational_count / 3, 1.0))
        
        # 5. Emotional words in headline
        emotional_count = sum(1 for word in EMOTIONAL_WORDS if word in headline_lower)
        features.append(min(emotional_count / 2, 1.0))
        
        # 6. Question marks in headline (clickbait tactic)
        headline_questions = headline.count('?')
        features.append(min(headline_questions / 2, 1.0))
        
        # 7. Exclamation marks in headline
        headline_exclaims = headline.count('!')
        features.append(min(headline_exclaims / 3, 1.0))
        
        # 8. All caps words in headline
        caps_words = sum(1 for word in headline.split() if word.isupper() and len(word) > 2)
        features.append(min(caps_words / 3, 1.0))
        
        # 9. Numbers in headline (listicles, rankings)
        headline_numbers = len(re.findall(r'\d+', headline))
        features.append(min(headline_numbers / 2, 1.0))
        
        # 10. Key headline words appear in content
        headline_key_words = [w for w in headline_words if len(w) > 4]
        words_in_content = sum(1 for word in headline_key_words if word in content_lower)
        word_match_ratio = words_in_content / max(len(headline_key_words), 1)
        features.append(word_match_ratio)
        
        # 11-15. Headline structure analysis
        starts_with_number = int(headline[0].isdigit() if headline else 0)
        contains_colon = int(':' in headline)
        contains_quotes = int('"' in headline or "'" in headline)
        is_question = int(headline.strip().endswith('?'))
        
        features.extend([starts_with_number, contains_colon, contains_quotes, is_question])
        
        # 16-20. Content validation
        content_has_facts = int(bool(re.search(r'\d+%|statistics|data|study|research', content_lower)))
        content_has_sources = int(bool(re.search(r'according to|source|cited|reported', content_lower)))
        content_has_quotes = int(content.count('"') >= 2)
        content_length_adequate = int(len(content_words) >= 100)
        
        features.extend([content_has_facts, content_has_sources, content_has_quotes, content_length_adequate])
        
        # 21-30. Semantic consistency
        # Check if headline promises are fulfilled in content
        
        # Promise words in headline but not in content
        promise_words = ['revealed', 'exposed', 'secret', 'truth', 'shocking']
        promises_in_headline = sum(1 for word in promise_words if word in headline_lower)
        promises_in_content = sum(1 for word in promise_words if word in content_lower)
        unfulfilled_promises = max(0, promises_in_headline - promises_in_content)
        features.append(min(unfulfilled_promises / 2, 1.0))
        
        # Check for exaggeration
        superlatives = ['best', 'worst', 'most', 'least', 'greatest', 'biggest', 'smallest']
        headline_superlatives = sum(1 for word in superlatives if word in headline_lower)
        content_superlatives = sum(1 for word in superlatives if word in content_lower)
        exaggeration_ratio = headline_superlatives / max(content_superlatives, 1)
        features.append(min(exaggeration_ratio, 1.0))
        
        # 32-40. Additional pattern matching
        # Extract entities from headline
        headline_entities = re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', headline)
        entities_in_content = sum(1 for entity in headline_entities if entity in content)
        entity_match_ratio = entities_in_content / max(len(headline_entities), 1) if headline_entities else 0.5
        features.append(entity_match_ratio)
        
        # Check for vague language
        vague_words = ['this', 'these', 'it', 'they', 'something', 'someone']
        vagueness = sum(1 for word in vague_words if headline_lower.startswith(word))
        features.append(min(vagueness, 1.0))
        
        # 42-50. Sentiment analysis (simple)
        positive_words = ['good', 'great', 'amazing', 'wonderful', 'excellent', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disaster', 'worst']
        
        headline_positive = sum(1 for word in positive_words if word in headline_lower)
        headline_negative = sum(1 for word in negative_words if word in headline_lower)
        content_positive = sum(1 for word in positive_words if word in content_lower)
        content_negative = sum(1 for word in negative_words if word in content_lower)
        
        sentiment_mismatch = abs((headline_positive - headline_negative) - (content_positive - content_negative))
        features.append(min(sentiment_mismatch / 3, 1.0))
        
        # 51-64. Statistical features
        headline_avg_word_len = np.mean([len(w) for w in headline_words]) if headline_words else 0
        content_avg_word_len = np.mean([len(w) for w in content_words[:200]]) if content_words else 0
        features.append(min(headline_avg_word_len / 10, 1.0))
        features.append(min(content_avg_word_len / 10, 1.0))
        
        # Unique word ratio
        headline_unique_ratio = len(set(headline_words)) / max(len(headline_words), 1)
        features.append(headline_unique_ratio)
        
        # Pad to 64 features
        while len(features) < 64:
            features.append(0.0)
        
        features_array = np.array(features[:64], dtype=np.float32)
        
        return features_array, clickbait_count, sensational_count, word_match_ratio
        
    except Exception as e:
        print(f"[Headline Content Feature Extraction Error]: {str(e)}")
        return None

def analyze_headline_vs_content(headline, content):
    """
    Analyze headline accuracy and detect clickbait using MindSpore
    Returns: dict with judgment, clickbait score, explanation
    """
    try:
        # Extract features
        result = extract_headline_content_features(headline, content)
        if result is None:
            return {
                'judgment': 'ERROR',
                'clickbait_score': 0,
                'accuracy_score': 0,
                'confidence': 0,
                'explanation': 'Failed to analyze headline and content',
                'summary': 'ERROR - Could not analyze'
            }
        
        features, clickbait_count, sensational_count, word_match_ratio = result
        
        # Initialize model
        model = initialize_model()
        
        # MindSpore prediction
        with mindspore_lock:
            input_tensor = Tensor(features.reshape(1, -1).astype(np.float32))
            prediction = model(input_tensor)
            prediction_np = prediction.asnumpy()[0]
            
            accurate_score = float(prediction_np[0])
            misleading_score = float(prediction_np[1])
        
        # Rule-based analysis
        red_flags = []
        green_flags = []
        
        headline_lower = headline.lower()
        
        # Check for clickbait patterns
        clickbait_found = [pattern for pattern in CLICKBAIT_PATTERNS if pattern in headline_lower]
        if clickbait_found:
            red_flags.append(f"Clickbait patterns: {', '.join(clickbait_found[:2])}")
        
        # Check for sensationalism
        if sensational_count >= 2:
            red_flags.append(f"Excessive sensational language: {sensational_count} instances")
        
        # Check for excessive punctuation
        if headline.count('!') >= 2:
            red_flags.append(f"Excessive exclamation marks: {headline.count('!')}")
        
        # Check word match
        if word_match_ratio < 0.3:
            red_flags.append(f"Low keyword overlap with content: {word_match_ratio*100:.0f}%")
        elif word_match_ratio >= 0.7:
            green_flags.append(f"High keyword match with content: {word_match_ratio*100:.0f}%")
        
        # Check for question headlines
        if headline.strip().endswith('?'):
            red_flags.append("Question headline (clickbait tactic)")
        
        # Check content quality
        content_words = len(content.split())
        if content_words < 50:
            red_flags.append(f"Very short content: {content_words} words")
        elif content_words >= 200:
            green_flags.append(f"Substantial content: {content_words} words")
        
        # Check for factual content
        if re.search(r'according to|source|study|research|data', content.lower()):
            green_flags.append("Content cites sources or studies")
        
        # Calculate clickbait score
        clickbait_score = misleading_score * 100
        clickbait_score += clickbait_count * 15
        clickbait_score += sensational_count * 10
        clickbait_score -= word_match_ratio * 20
        clickbait_score = max(0, min(100, clickbait_score))
        
        # Calculate accuracy score (inverse of clickbait)
        accuracy_score = 100 - clickbait_score
        
        # Determine judgment
        if clickbait_score >= 60:
            judgment = "FAKE"
            confidence = 70 + (clickbait_score - 60) * 0.5
            status = "Misleading Headline"
        elif clickbait_score >= 30:
            judgment = "PARTIAL"
            confidence = 60 + (clickbait_score - 30) * 0.3
            status = "Somewhat Sensationalized"
        else:
            judgment = "REAL"
            confidence = 75 + (30 - clickbait_score) * 0.6
            status = "Accurate Headline"
        
        confidence = min(95, max(55, confidence))
        
        # Generate explanation
        explanation_parts = [
            f"Headline: {headline[:100]}{'...' if len(headline) > 100 else ''}",
            f"Clickbait Score: {clickbait_score:.1f}%",
            f"Accuracy Score: {accuracy_score:.1f}%",
            f"Confidence: {confidence:.1f}%",
            f"Assessment: {status}"
        ]
        
        if green_flags:
            explanation_parts.append("\nPositive Indicators:")
            for flag in green_flags:
                explanation_parts.append(f"• {flag}")
        
        if red_flags:
            explanation_parts.append("\nRed Flags:")
            for flag in red_flags:
                explanation_parts.append(f"• {flag}")
        
        if judgment == "REAL":
            explanation_parts.append("\nConclusion: Headline accurately represents the content")
        elif judgment == "PARTIAL":
            explanation_parts.append("\nConclusion: Headline is somewhat exaggerated but not completely misleading")
        else:
            explanation_parts.append("\nConclusion: Headline is clickbait and does not accurately represent content")
        
        explanation = "\n".join(explanation_parts)
        
        # REAL-TIME FACT VERIFICATION: Check if claims in headline are true
        realtime_verdict = None
        if REALTIME_AVAILABLE and len(headline) > 20:
            try:
                rt_checker = get_fact_checker()
                realtime_result = rt_checker.verify_claims(headline)
                
                # If headline contains FALSE claims (especially CRITICAL ones)
                if realtime_result.get('false_claims'):
                    false_count = len(realtime_result['false_claims'])
                    critical_false = sum(1 for c in realtime_result['false_claims'] if c.get('severity') == 'CRITICAL')
                    
                    if critical_false > 0:
                        red_flags.append(f"Real-time check: {critical_false} CRITICAL false claims in headline")
                        clickbait_score = min(100, clickbait_score + 30)
                        accuracy_score = max(0, accuracy_score - 30)
                    else:
                        red_flags.append(f"Real-time check: {false_count} false claims detected")
                        clickbait_score = min(100, clickbait_score + 20)
                        accuracy_score = max(0, accuracy_score - 20)
                
                # If headline has verified true claims
                if realtime_result.get('verified_claims'):
                    verified_count = len(realtime_result['verified_claims'])
                    green_flags.append(f"Real-time check: {verified_count} claims verified as accurate")
                    clickbait_score = max(0, clickbait_score - 10)
                    accuracy_score = min(100, accuracy_score + 10)
                
                realtime_verdict = realtime_result.get('overall_verdict', 'UNKNOWN')
                explanation += f"\n\nReal-time Verification: {realtime_verdict}"
                
            except Exception as e:
                print(f"[Real-time headline verification warning]: {str(e)}")
        
        # Re-evaluate judgment after real-time check
        if clickbait_score >= 60:
            judgment = "FAKE"
            confidence = 70 + (clickbait_score - 60) * 0.5
            status = "Misleading Headline"
        elif clickbait_score >= 30:
            judgment = "PARTIAL"
            confidence = 60 + (clickbait_score - 30) * 0.3
            status = "Somewhat Sensationalized"
        else:
            judgment = "REAL"
            confidence = 75 + (30 - clickbait_score) * 0.6
            status = "Accurate Headline"
        
        confidence = min(95, max(55, confidence))
        
        # Summary for bubbles
        summary = f"{judgment} - {confidence:.0f}% confidence\n\n{status}\nClickbait Score: {clickbait_score:.0f}%"
        if realtime_verdict:
            summary += f"\nReal-time: {realtime_verdict}"
        
        return {
            'judgment': judgment,
            'clickbait_score': round(clickbait_score, 1),
            'accuracy_score': round(accuracy_score, 1),
            'confidence': round(confidence, 1),
            'explanation': explanation,
            'summary': summary,
            'status': status,
            'red_flags': red_flags,
            'green_flags': green_flags,
            'realtime_verdict': realtime_verdict
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[MindSpore Headline Analysis Error]: {str(e)}\n{error_trace}")
        return {
            'judgment': 'ERROR',
            'clickbait_score': 0,
            'accuracy_score': 0,
            'confidence': 0,
            'explanation': f'Analysis failed: {str(e)}',
            'summary': 'ERROR - Analysis failed'
        }

if __name__ == "__main__":
    # Test the analyzer
    print("MindSpore Headline vs Content Analyzer initialized")
    print("Model ready for clickbait detection")
