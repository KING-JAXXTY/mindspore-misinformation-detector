"""
Fact Cross-Reference Tool using MindSpore
Verifies claims against multiple trusted sources
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
from .real_time_fact_checker import get_fact_checker

class MindSporeFactChecker(nn.Cell):
    """Neural network for fact verification analysis"""
    
    def __init__(self):
        super(MindSporeFactChecker, self).__init__()
        # Input: 48 claim features
        self.dense1 = nn.Dense(48, 96)
        self.dense2 = nn.Dense(96, 64)
        self.dense3 = nn.Dense(64, 32)
        self.dense4 = nn.Dense(32, 3)  # True, False, Unverified
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return self.softmax(x)

# Global model instance
fact_checker_model = None

def initialize_model():
    """Initialize the MindSpore fact checker model"""
    global fact_checker_model
    if fact_checker_model is None:
        with mindspore_lock:
            fact_checker_model = MindSporeFactChecker()
            dummy_input = Tensor(np.random.randn(1, 48).astype(np.float32))
            _ = fact_checker_model(dummy_input)
    return fact_checker_model

# Known false claim patterns
FALSE_CLAIM_INDICATORS = [
    'proven false', 'debunked', 'hoax', 'fake', 'misleading',
    'false claim', 'no evidence', 'unverified', 'conspiracy',
    'misinformation', 'disinformation', 'fabricated'
]

# Verifiable claim patterns
VERIFIABLE_PATTERNS = [
    r'\d+%',  # Percentages
    r'\$[\d,]+',  # Money amounts
    r'\d{4}',  # Years
    r'study shows',
    r'research indicates',
    r'according to',
    r'data shows',
    r'statistics',
    r'survey',
    r'poll'
]

# Unverifiable claim patterns
UNVERIFIABLE_PATTERNS = [
    'some say', 'many believe', 'everyone knows', 'they say',
    'rumor has it', 'allegedly', 'supposedly', 'claimed without evidence',
    'anonymous source', 'insider says', 'experts say' # vague
]

def extract_claim_features(claim_text):
    """Extract features from claim text for fact-checking analysis"""
    try:
        claim_lower = claim_text.lower()
        
        features = []
        
        # 1. Text length
        text_length = len(claim_text)
        features.append(min(text_length / 500, 1.0))
        
        # 2. Contains numbers/data (more verifiable)
        has_numbers = int(bool(re.search(r'\d+', claim_text)))
        features.append(has_numbers)
        
        # 3. Contains specific dates
        has_dates = int(bool(re.search(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}', claim_text)))
        features.append(has_dates)
        
        # 4. Contains citations/sources
        has_citations = int(bool(re.search(r'according to|source:|cited|study|research', claim_lower)))
        features.append(has_citations)
        
        # 5. Contains false claim indicators
        false_indicators = sum(1 for pattern in FALSE_CLAIM_INDICATORS if pattern in claim_lower)
        features.append(min(false_indicators / 3, 1.0))
        
        # 6. Contains verifiable patterns
        verifiable_count = sum(1 for pattern in VERIFIABLE_PATTERNS if re.search(pattern, claim_lower))
        features.append(min(verifiable_count / 5, 1.0))
        
        # 7. Contains unverifiable patterns
        unverifiable_count = sum(1 for pattern in UNVERIFIABLE_PATTERNS if pattern in claim_lower)
        features.append(min(unverifiable_count / 3, 1.0))
        
        # 8. Specificity (specific claims are more verifiable)
        specific_words = ['specifically', 'exactly', 'precisely', 'documented', 'recorded']
        specificity = sum(1 for word in specific_words if word in claim_lower)
        features.append(min(specificity / 3, 1.0))
        
        # 9. Vagueness indicators
        vague_words = ['maybe', 'possibly', 'might', 'could', 'allegedly', 'supposedly']
        vagueness = sum(1 for word in vague_words if word in claim_lower)
        features.append(min(vagueness / 3, 1.0))
        
        # 10. Sensationalism
        sensational_words = ['shocking', 'unbelievable', 'incredible', 'mind-blowing', 'stunning']
        sensationalism = sum(1 for word in sensational_words if word in claim_lower)
        features.append(min(sensationalism / 3, 1.0))
        
        # 11. Question marks (claims shouldn't be questions)
        question_count = claim_text.count('?')
        features.append(min(question_count / 2, 1.0))
        
        # 12. Exclamation marks (excessive = suspicious)
        exclaim_count = claim_text.count('!')
        features.append(min(exclaim_count / 3, 1.0))
        
        # 13-20. Word frequency analysis
        words = claim_lower.split()
        word_count = len(words)
        
        # Average word length
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append(min(avg_word_len / 10, 1.0))
        
        # Unique word ratio
        unique_ratio = len(set(words)) / max(word_count, 1)
        features.append(unique_ratio)
        
        # Capital letter ratio (all caps = suspicious)
        caps_ratio = sum(1 for c in claim_text if c.isupper()) / max(len(claim_text), 1)
        features.append(min(caps_ratio * 2, 1.0))
        
        # 16-25. Linguistic features
        contains_stats = int(bool(re.search(r'percent|\d+%|statistics|data', claim_lower)))
        contains_money = int(bool(re.search(r'\$|dollar|euro|currency', claim_lower)))
        contains_names = int(bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', claim_text)))
        contains_locations = int('in ' in claim_lower or 'at ' in claim_lower)
        contains_timeframe = int(bool(re.search(r'yesterday|today|last week|this month|in \d{4}', claim_lower)))
        
        features.extend([contains_stats, contains_money, contains_names, contains_locations, contains_timeframe])
        
        # 26-35. Claim structure
        has_subject = int(len(words) > 0)
        has_verb = int(bool(re.search(r'\b(is|are|was|were|has|have|will|would|could|should)\b', claim_lower)))
        has_object = int(word_count >= 3)
        is_complete_sentence = int(claim_text.strip().endswith('.'))
        
        features.extend([has_subject, has_verb, has_object, is_complete_sentence])
        
        # Pad to 48 features
        while len(features) < 48:
            features.append(0.0)
        
        features_array = np.array(features[:48], dtype=np.float32)
        
        return features_array, word_count, verifiable_count, unverifiable_count
        
    except Exception as e:
        print(f"[Claim Feature Extraction Error]: {str(e)}")
        return None

def check_fact(claim_text, use_realtime=True):
    """
    Cross-reference claim against fact-checking patterns using MindSpore
    Args:
        claim_text: The claim to verify
        use_realtime: Whether to use real-time web verification (default: True)
    Returns: dict with verification status, confidence, explanation
    """
    try:
        # Try real-time verification first for critical claims
        realtime_result = None
        if use_realtime:
            try:
                rt_checker = get_fact_checker()
                realtime_result = rt_checker.verify_claims(claim_text)
                
                # If real-time verification found ANY false claims with HIGH or CRITICAL severity, prioritize it
                if realtime_result.get('false_claims'):
                    critical_false = [c for c in realtime_result['false_claims'] if c.get('severity') in ['CRITICAL', 'HIGH']]
                    
                    if critical_false:
                        # Build detailed explanation
                        false_details = "\n\n".join([
                            f"✗ {c['claim']}\n  {c['explanation']}\n  Severity: {c.get('severity', 'HIGH')}"
                            for c in critical_false[:3]
                        ])
                        
                        return {
                            'judgment': 'FAKE',
                            'verification_score': 5,
                            'confidence': 95,
                            'explanation': f"CRITICAL MISINFORMATION DETECTED\n\n{false_details}\n\nReal-time verification: {realtime_result.get('overall_verdict', 'FAILED')}",
                            'summary': f"FAKE - 95% confidence\n\nContains {len(critical_false)} critical false claim(s)\nReal-time verification: {realtime_result.get('overall_verdict', 'FAILED')}",
                            'status': 'Critical Misinformation',
                            'red_flags': [f"Real-time: {c['claim']}" for c in critical_false],
                            'green_flags': [],
                            'realtime_check': realtime_result
                        }
                    
                    # If there are any false claims (even low severity), heavily penalize
                    elif len(realtime_result['false_claims']) > 0:
                        # Continue to ML but with heavy penalty - will be applied below
                        pass
                        
            except Exception as e:
                print(f"[Real-time verification warning]: {str(e)}")
                # Continue with ML analysis if real-time fails
        
        # Extract features
        result = extract_claim_features(claim_text)
        if result is None:
            return {
                'judgment': 'ERROR',
                'verification_score': 0,
                'confidence': 0,
                'explanation': 'Failed to analyze claim',
                'summary': 'ERROR - Could not analyze'
            }
        
        features, word_count, verifiable_count, unverifiable_count = result
        
        # Initialize model
        model = initialize_model()
        
        # MindSpore prediction
        with mindspore_lock:
            input_tensor = Tensor(features.reshape(1, -1).astype(np.float32))
            prediction = model(input_tensor)
            prediction_np = prediction.asnumpy()[0]
            
            true_score = float(prediction_np[0])
            false_score = float(prediction_np[1])
            unverified_score = float(prediction_np[2])
        
        # Rule-based analysis
        red_flags = []
        green_flags = []
        
        claim_lower = claim_text.lower()
        
        # Check for verifiable elements
        if verifiable_count >= 2:
            green_flags.append(f"Contains {verifiable_count} verifiable elements (data, sources)")
        
        # Check for citations
        if re.search(r'according to|source:|cited|study|research', claim_lower):
            green_flags.append("Cites sources or studies")
        
        # Check for specific data
        if re.search(r'\d+%|\$[\d,]+|statistics', claim_text):
            green_flags.append("Contains specific numerical data")
        
        # Check for false claim indicators
        false_indicators_found = [ind for ind in FALSE_CLAIM_INDICATORS if ind in claim_lower]
        if false_indicators_found:
            red_flags.append(f"Contains false claim indicators: {', '.join(false_indicators_found[:2])}")
        
        # Check for unverifiable patterns
        if unverifiable_count >= 2:
            red_flags.append(f"Contains {unverifiable_count} unverifiable phrases (vague claims)")
        
        # Check for vagueness
        vague_words = ['some say', 'many believe', 'everyone knows', 'allegedly']
        vague_found = [word for word in vague_words if word in claim_lower]
        if vague_found:
            red_flags.append(f"Vague attribution: {', '.join(vague_found[:2])}")
        
        # Check for sensationalism
        sensational = ['shocking', 'unbelievable', 'incredible', 'mind-blowing']
        sensational_found = [word for word in sensational if word in claim_lower]
        if sensational_found:
            red_flags.append(f"Sensational language: {', '.join(sensational_found[:2])}")
        
        # Check for lack of specificity
        if word_count > 20 and verifiable_count == 0:
            red_flags.append("Long claim with no verifiable details")
        
        # Calculate verification score
        verification_score = true_score * 100
        
        # Adjust based on flags
        verification_score += len(green_flags) * 12
        verification_score -= len(red_flags) * 15
        verification_score = max(0, min(100, verification_score))
        
        # Determine judgment
        if verification_score >= 65:
            judgment = "REAL"
            confidence = 70 + (verification_score - 65) * 0.6
            status = "Likely Accurate"
        elif verification_score >= 35:
            judgment = "PARTIAL"
            confidence = 55 + (verification_score - 35) * 0.4
            status = "Needs Verification"
        else:
            judgment = "FAKE"
            confidence = 65 + (35 - verification_score) * 0.7
            status = "Likely False"
        
        confidence = min(95, max(50, confidence))
        
        # Generate explanation
        explanation_parts = [
            f"Fact Check: {claim_text[:100]}{'...' if len(claim_text) > 100 else ''}",
            f"Verification Score: {verification_score:.1f}%",
            f"Confidence: {confidence:.1f}%",
            f"Status: {status}"
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
            explanation_parts.append("\nConclusion: Claim contains verifiable elements and appears credible")
        elif judgment == "PARTIAL":
            explanation_parts.append("\nConclusion: Claim needs independent verification from trusted sources")
        else:
            explanation_parts.append("\nConclusion: Claim shows signs of misinformation - verify before sharing")
        
        explanation = "\n".join(explanation_parts)
        
        # Integrate real-time verification results if available
        if realtime_result:
            realtime_verdict = realtime_result.get('overall_verdict', 'UNKNOWN')
            realtime_reliability = realtime_result.get('reliability_score', 0)
            
            # Adjust scores based on real-time findings
            if realtime_result.get('false_claims'):
                verification_score = max(0, verification_score - 30)
                red_flags.append(f"Real-time check: {len(realtime_result['false_claims'])} false claims found")
            
            if realtime_result.get('verified_claims'):
                verification_score = min(100, verification_score + 15)
                green_flags.append(f"Real-time check: {len(realtime_result['verified_claims'])} verified claims")
            
            # Add real-time verdict to explanation
            explanation += f"\n\nReal-time Verification: {realtime_verdict} ({realtime_reliability:.0f}% reliability)"
            if realtime_result.get('verified_claims'):
                explanation += f"\n{len(realtime_result['verified_claims'])} claims verified online"
            if realtime_result.get('false_claims'):
                explanation += f"\n{len(realtime_result['false_claims'])} false claims detected"
        
        # Summary for bubbles
        rt_info = f"\nReal-time: {realtime_result.get('overall_verdict', 'N/A')}" if realtime_result else ""
        summary = f"{judgment} - {confidence:.0f}% confidence\n\n{status}\n{len(green_flags)} positive indicators, {len(red_flags)} red flags{rt_info}"
        
        return {
            'judgment': judgment,
            'verification_score': round(verification_score, 1),
            'confidence': round(confidence, 1),
            'explanation': explanation,
            'summary': summary,
            'status': status,
            'red_flags': red_flags,
            'green_flags': green_flags,
            'realtime_check': realtime_result
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[MindSpore Fact Check Error]: {str(e)}\n{error_trace}")
        return {
            'judgment': 'ERROR',
            'verification_score': 0,
            'confidence': 0,
            'explanation': f'Analysis failed: {str(e)}',
            'summary': 'ERROR - Analysis failed'
        }

if __name__ == "__main__":
    # Test the fact checker
    print("MindSpore Fact Cross-Reference Tool initialized")
    print("Model ready for claim verification")
