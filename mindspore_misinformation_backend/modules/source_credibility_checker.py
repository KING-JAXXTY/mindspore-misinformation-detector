"""
Source Credibility Checker using MindSpore
Analyzes URLs and domains for fake news patterns and reputation
NOW WITH REAL-TIME ONLINE VERIFICATION
"""

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
import threading
import traceback
import re
from urllib.parse import urlparse
import requests

mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="CPU")
mindspore_lock = threading.Lock()

# Import real-time fact checker
try:
    from .real_time_fact_checker import get_fact_checker
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    print("[Warning] Real-time fact checker not available for source credibility")

class MindSporeSourceCredibilityChecker(nn.Cell):
    """Neural network for domain credibility analysis"""
    
    def __init__(self):
        super(MindSporeSourceCredibilityChecker, self).__init__()
        # Input: 32 domain features
        self.dense1 = nn.Dense(32, 64)
        self.dense2 = nn.Dense(64, 32)
        self.dense3 = nn.Dense(32, 16)
        self.dense4 = nn.Dense(16, 2)  # Credible vs Suspicious
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return self.softmax(x)

# Global model instance
credibility_model = None

def initialize_model():
    """Initialize the MindSpore source credibility model"""
    global credibility_model
    if credibility_model is None:
        with mindspore_lock:
            credibility_model = MindSporeSourceCredibilityChecker()
            dummy_input = Tensor(np.random.randn(1, 32).astype(np.float32))
            _ = credibility_model(dummy_input)
    return credibility_model

# Known credible sources
CREDIBLE_DOMAINS = [
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'nytimes.com',
    'washingtonpost.com', 'theguardian.com', 'economist.com', 'npr.org',
    'cnn.com', 'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'pbs.org',
    'bloomberg.com', 'wsj.com', 'ft.com', 'time.com', 'newsweek.com',
    'usatoday.com', 'latimes.com', 'chicagotribune.com', 'politico.com',
    'thehill.com', 'axios.com', 'propublica.org', 'factcheck.org',
    'snopes.com', 'reuters.com', 'apnews.com'
]

# Known suspicious indicators in domains
SUSPICIOUS_PATTERNS = [
    'fake', 'hoax', 'buzz', 'viral', 'shocking', 'breaking', 
    'truth', 'real', 'news', 'daily', 'chronicle', 'times', 
    'post', 'herald', 'tribune', 'gazette', 'journal'
]

# Suspicious TLDs
SUSPICIOUS_TLDS = [
    '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.club',
    '.info', '.biz', '.cn', '.ru'
]

def extract_domain_features(url):
    """Extract features from URL/domain for credibility analysis"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower() or parsed.path.lower()
        
        # Remove www
        domain = domain.replace('www.', '')
        
        features = []
        
        # 1. Domain length (suspicious if very long)
        domain_length = len(domain)
        features.append(min(domain_length / 50, 1.0))  # Normalize
        
        # 2. Number of subdomains (more = suspicious)
        subdomain_count = domain.count('.')
        features.append(min(subdomain_count / 5, 1.0))
        
        # 3. Contains numbers (suspicious)
        has_numbers = int(bool(re.search(r'\d', domain)))
        features.append(has_numbers)
        
        # 4. Contains hyphens (suspicious)
        hyphen_count = domain.count('-')
        features.append(min(hyphen_count / 3, 1.0))
        
        # 5. TLD analysis
        tld = '.' + domain.split('.')[-1] if '.' in domain else ''
        is_suspicious_tld = int(tld in SUSPICIOUS_TLDS)
        features.append(is_suspicious_tld)
        
        # 6. Contains suspicious keywords
        suspicious_keyword_count = sum(1 for pattern in SUSPICIOUS_PATTERNS if pattern in domain)
        features.append(min(suspicious_keyword_count / 3, 1.0))
        
        # 7. Is known credible source
        is_credible = int(any(cred in domain for cred in CREDIBLE_DOMAINS))
        features.append(is_credible)
        
        # 8. Domain mimics credible source (contains "news" but not credible)
        mimics_news = int('news' in domain and not is_credible)
        features.append(mimics_news)
        
        # 9-16. Character distribution features
        vowel_ratio = sum(domain.count(v) for v in 'aeiou') / max(len(domain), 1)
        consonant_ratio = sum(domain.count(c) for c in 'bcdfghjklmnpqrstvwxyz') / max(len(domain), 1)
        features.extend([vowel_ratio, consonant_ratio])
        
        # 10-16. Additional pattern features
        has_www = int('www' in url.lower())
        has_https = int(url.startswith('https'))
        path_length = len(parsed.path)
        has_query = int(bool(parsed.query))
        has_fragment = int(bool(parsed.fragment))
        
        features.extend([has_www, has_https, min(path_length / 100, 1.0), has_query, has_fragment])
        
        # Pad to 32 features
        while len(features) < 32:
            features.append(0.0)
        
        features_array = np.array(features[:32], dtype=np.float32)
        
        return features_array, domain, tld
        
    except Exception as e:
        print(f"[Domain Feature Extraction Error]: {str(e)}")
        return None

def check_source_credibility(url):
    """
    Check source credibility using MindSpore
    Returns: dict with judgment, credibility score, explanation
    """
    try:
        # Extract features
        result = extract_domain_features(url)
        if result is None:
            return {
                'judgment': 'ERROR',
                'credibility_score': 0,
                'confidence': 0,
                'explanation': 'Failed to parse URL',
                'summary': 'ERROR - Invalid URL'
            }
        
        features, domain, tld = result
        
        # Initialize model
        model = initialize_model()
        
        # MindSpore prediction
        with mindspore_lock:
            input_tensor = Tensor(features.reshape(1, -1).astype(np.float32))
            prediction = model(input_tensor)
            prediction_np = prediction.asnumpy()[0]
            
            credible_score = float(prediction_np[0])
            suspicious_score = float(prediction_np[1])
        
        # Rule-based analysis
        red_flags = []
        green_flags = []
        
        # Check if known credible source
        if any(cred in domain for cred in CREDIBLE_DOMAINS):
            green_flags.append(f"Known credible source: {domain}")
            credible_score += 0.3
        
        # Check for suspicious TLD
        if tld in SUSPICIOUS_TLDS:
            red_flags.append(f"Suspicious domain extension: {tld}")
        
        # Check for suspicious patterns
        found_patterns = [p for p in SUSPICIOUS_PATTERNS if p in domain]
        if len(found_patterns) >= 2 and not any(cred in domain for cred in CREDIBLE_DOMAINS):
            red_flags.append(f"Multiple suspicious keywords: {', '.join(found_patterns[:3])}")
        
        # Check for numbers in domain
        if re.search(r'\d{2,}', domain):
            red_flags.append("Contains multiple numbers in domain")
        
        # Check for excessive hyphens
        if domain.count('-') >= 2:
            red_flags.append("Multiple hyphens in domain")
        
        # Check for very long domain
        if len(domain) > 30:
            red_flags.append(f"Unusually long domain: {len(domain)} characters")
        
        # Check for news mimicry
        if 'news' in domain and not any(cred in domain for cred in CREDIBLE_DOMAINS):
            red_flags.append("Mimics news site name")
        
        # Calculate credibility score
        credibility_score = credible_score * 100
        
        # Adjust based on flags
        credibility_score += len(green_flags) * 15
        credibility_score -= len(red_flags) * 12
        credibility_score = max(0, min(100, credibility_score))
        
        # Determine judgment
        if credibility_score >= 70:
            judgment = "REAL"
            confidence = 75 + (credibility_score - 70) * 0.7
        elif credibility_score >= 40:
            judgment = "PARTIAL"
            confidence = 60 + (credibility_score - 40) * 0.3
        else:
            judgment = "FAKE"
            confidence = 70 + (40 - credibility_score) * 0.6
        
        confidence = min(95, max(55, confidence))
        
        # Generate explanation
        explanation_parts = [
            f"Source Analysis: {domain}",
            f"Credibility Score: {credibility_score:.1f}%",
            f"Confidence: {confidence:.1f}%"
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
            explanation_parts.append("\nConclusion: Source appears credible and trustworthy")
        elif judgment == "PARTIAL":
            explanation_parts.append("\nConclusion: Source has mixed credibility - verify information independently")
        else:
            explanation_parts.append("\nConclusion: Source appears suspicious - high risk of misinformation")
        
        explanation = "\n".join(explanation_parts)
        
        # REAL-TIME VERIFICATION: Check if domain is actually reachable and legit
        realtime_check = None
        if REALTIME_AVAILABLE:
            try:
                # Try to verify the domain exists and is reachable
                headers = {'User-Agent': 'MindSpore Misinformation Detector/1.0'}
                response = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
                
                if response.status_code == 200:
                    green_flags.append("Real-time check: Domain is reachable and active")
                    credibility_score = min(100, credibility_score + 5)
                    realtime_check = {'status': 'reachable', 'code': response.status_code}
                elif response.status_code == 404:
                    red_flags.append("Real-time check: Domain returned 404 (not found)")
                    credibility_score = max(0, credibility_score - 15)
                    realtime_check = {'status': 'not_found', 'code': 404}
                else:
                    realtime_check = {'status': 'unknown', 'code': response.status_code}
            except requests.exceptions.SSLError:
                red_flags.append("Real-time check: SSL certificate error (insecure)")
                credibility_score = max(0, credibility_score - 20)
                realtime_check = {'status': 'ssl_error'}
            except requests.exceptions.Timeout:
                red_flags.append("Real-time check: Domain unreachable (timeout)")
                credibility_score = max(0, credibility_score - 10)
                realtime_check = {'status': 'timeout'}
            except Exception as e:
                # Domain might not exist or be blocked
                if 'Name or service not known' in str(e) or 'getaddrinfo failed' in str(e):
                    red_flags.append("Real-time check: Domain does not exist")
                    credibility_score = max(0, credibility_score - 25)
                    realtime_check = {'status': 'non_existent'}
        
        # Re-evaluate judgment after real-time check
        if credibility_score >= 70:
            judgment = "REAL"
            confidence = 75 + (credibility_score - 70) * 0.7
        elif credibility_score >= 40:
            judgment = "PARTIAL"
            confidence = 60 + (credibility_score - 40) * 0.3
        else:
            judgment = "FAKE"
            confidence = 70 + (40 - credibility_score) * 0.6
        
        confidence = min(95, max(55, confidence))
        
        # Update summary
        summary = f"{judgment} - {confidence:.0f}% confidence\n\nDomain: {domain}\n{len(red_flags)} red flags, {len(green_flags)} positive indicators"
        if realtime_check:
            summary += f"\nReal-time: {realtime_check.get('status', 'unknown')}"
        
        return {
            'judgment': judgment,
            'credibility_score': round(credibility_score, 1),
            'confidence': round(confidence, 1),
            'explanation': explanation,
            'summary': summary,
            'domain': domain,
            'red_flags': red_flags,
            'green_flags': green_flags,
            'realtime_check': realtime_check
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[MindSpore Source Credibility Error]: {str(e)}\n{error_trace}")
        return {
            'judgment': 'ERROR',
            'credibility_score': 0,
            'confidence': 0,
            'explanation': f'Analysis failed: {str(e)}',
            'summary': 'ERROR - Analysis failed'
        }

if __name__ == "__main__":
    # Test the checker
    print("MindSpore Source Credibility Checker initialized")
    print("Model ready for domain analysis")
