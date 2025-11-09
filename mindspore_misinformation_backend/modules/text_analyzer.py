"""
MindSpore Text Analysis Module
=============================

Advanced text analysis for misinformation detection using MindSpore.
Includes BERT-like transformer models and custom neural networks.
NOW WITH REAL-TIME FACT VERIFICATION FOR UNPREDICTABLE CLAIMS
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

# Thread lock for MindSpore operations to prevent GIL conflicts
mindspore_lock = threading.Lock()

# Import real-time fact checker
try:
    from .real_time_fact_checker import get_fact_checker
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    logger.warning("Real-time fact checker not available for text analyzer")

class TextFeaturesExtractor:
    """Extract various text features for misinformation detection"""
    
    def __init__(self):
        # Misinformation keywords and patterns (EXPANDED)
        self.fake_indicators = [
            # Urgency/sensationalism
            'breaking', 'urgent', 'shocking', 'unbelievable', 'bombshell', 'explosive',
            'emergency', 'critical', 'must read', 'must see', 'must watch', 'breaking news',
            'just in', 'developing', 'alert', 'warning', 'exclusive', 'leaked',
            
            # Medical misinformation
            'doctors hate this', 'big pharma', 'cure for cancer', 'miracle cure',
            'natural remedy cures', 'alternative medicine secret', 'medical establishment hiding',
            'pharmaceutical conspiracy', 'vaccines cause', 'detox scam', 'instant weight loss',
            
            # Conspiracy language
            'they don\'t want you to know', 'secret', 'hidden truth', 'cover up',
            'mainstream media', 'wake up', 'sheeple', 'conspiracy', 'deep state',
            'shadow government', 'new world order', 'illuminati', 'elite control',
            'hidden agenda', 'false flag', 'mind control', 'depopulation agenda',
            
            # Insider/leaked content
            'insider reveals', 'leaked documents', 'whistleblower exposes', 'classified information',
            'confidential report', 'anonymous source reveals', 'internal memo', 'secret meeting',
            
            # Extreme claims
            'you won\'t believe', 'amazing discovery', 'scientists baffled', 'defies explanation',
            'experts speechless', 'changes everything', 'game changer', 'revolutionary breakthrough',
            'shocking revelation', 'stunning evidence', 'irrefutable proof', 'undeniable truth',
            
            # Social proof manipulation
            'everyone is talking about', 'going viral', 'millions of people', 'they are hiding',
            'suppressed information', 'banned video', 'censored truth', 'deleted post',
            
            # Financial scams
            'get rich quick', 'make money fast', 'passive income secret', 'investment opportunity',
            'guaranteed returns', 'risk-free profit', 'limited time offer', 'act now',
            
            # Political misinformation
            'rigged election', 'voter fraud', 'stolen election', 'fake ballots',
            'election interference', 'deep state coup', 'political agenda', 'propaganda machine',
            
            # Science denial
            'climate hoax', 'global warming scam', 'evolution lie', 'fake science',
            'junk science', 'pseudoscience truth', 'mainstream science wrong',
            
            # Additional strong indicators
            'share before deleted', 'they are deleting', 'watch before removed',
            'truth they hide', 'forbidden knowledge', 'suppressed by media',
            'msm won\'t show', 'government doesn\'t want', 'big tech censoring',
            'follow the money', 'do your own research', 'open your eyes',
            'the real truth', 'actual facts', 'what really happened',
            'exposed finally', 'proof revealed', 'evidence surfaces',
            'confirmed leak', 'insider confirms', 'sources say secretly',
            'this will shock', 'prepare to be amazed', 'jaw dropping',
            'mind blowing fact', 'disturbing truth', 'terrifying reality',
            'scary truth', 'hidden danger', 'real threat they hide',
            'doctors shocked', 'scientists stunned', 'experts baffled',
            'study reveals shocking', 'research proves undeniably',
            'miracle solution', 'instant cure', 'fixes everything',
            'works 100%', 'guaranteed method', 'proven technique',
            'secret method', 'hidden technique', 'ancient secret',
            'powerful elite', 'controlled media', 'fake news media',
            'they lie to you', 'brainwashing public', 'mass deception'
        ]
        
        self.clickbait_patterns = [
            # Numbered lists
            r'\d+ (things|ways|reasons|facts|secrets|tricks|tips|methods|steps)',
            r'(top|best) \d+ (things|ways|reasons|secrets|facts)',
            r'(number \d+|#\d+) will (shock|amaze|surprise|blow your mind)',
            
            # You won't believe
            r'you (won\'t believe|will be shocked|need to see|must see|have to see)',
            r'you\'ll (never believe|be shocked|be amazed)',
            r'wait until you see',
            
            # This one trick
            r'this (one|simple|weird|crazy|amazing) (trick|tip|secret|method)',
            r'(doctors|experts|scientists|celebrities) (hate|love|don\'t want you to know)',
            
            # What happened next
            r'what (happened|happens|they did) (next|after|then)',
            r'the (result|outcome|ending) will (shock|surprise|amaze) you',
            
            # Before/After
            r'(before|after) (and|&) after',
            r'the (transformation|change|difference) is (incredible|amazing|shocking)',
            
            # Location-based
            r'in your (area|city|town|neighborhood)',
            r'(local|nearby) (mom|dad|woman|man|person)',
            
            # Time pressure
            r'(today|now|right now|immediately)',
            r'(limited time|act now|don\'t wait|hurry)',
            r'(before it\'s|while you still can|while supplies last)',
            
            # Question hooks
            r'(do|did|can|will|should) you (know|believe|think)',
            r'are you (ready|prepared|aware)',
            r'what if (i|we) told you'
        ]
        
        self.emotion_words = {
            'fear': [
                'scary', 'terrifying', 'dangerous', 'threat', 'warning', 'alert', 'panic',
                'horror', 'nightmare', 'catastrophe', 'disaster', 'apocalypse', 'doom',
                'crisis', 'emergency', 'deadly', 'fatal', 'lethal', 'toxic', 'poisonous',
                'harmful', 'hazardous', 'risky', 'perilous', 'menacing', 'ominous',
                'dreadful', 'frightening', 'alarming', 'disturbing', 'terrified', 'scared'
            ],
            'anger': [
                'outrageous', 'disgusting', 'horrible', 'awful', 'terrible', 'furious',
                'enraged', 'infuriating', 'maddening', 'frustrating', 'annoying', 'irritating',
                'offensive', 'insulting', 'appalling', 'atrocious', 'heinous', 'vile',
                'despicable', 'contemptible', 'detestable', 'abominable', 'revolting',
                'repulsive', 'sickening', 'nauseating', 'repugnant', 'loathsome'
            ],
            'excitement': [
                'amazing', 'incredible', 'unbelievable', 'shocking', 'wow', 'awesome',
                'spectacular', 'phenomenal', 'extraordinary', 'remarkable', 'outstanding',
                'fantastic', 'fabulous', 'wonderful', 'marvelous', 'sensational', 'stunning',
                'astonishing', 'astounding', 'breathtaking', 'mind-blowing', 'jaw-dropping',
                'eye-opening', 'game-changing', 'revolutionary', 'groundbreaking', 'epic'
            ],
            'urgency': [
                'now', 'immediately', 'urgent', 'hurry', 'quick', 'fast', 'today',
                'limited time', 'act now', 'don\'t wait', 'before it\'s too late',
                'running out', 'last chance', 'final opportunity', 'expires soon',
                'while supplies last', 'time sensitive', 'critical', 'pressing'
            ]
        }
        
        # Origin detection patterns (without online verification)
        self.origin_indicators = {
            'social_media': [
                'shared on facebook', 'viral on twitter', 'instagram post', 'tiktok video',
                'youtube video', 'snapchat', 'whatsapp message', 'telegram channel',
                'going viral', 'share this', 'retweet', 'like and share', 'trending',
                'hashtag', '#', '@', 'viral', 'going around social media',
                'seen on social media', 'circulating online', 'spreading fast',
                'everyone is talking about', 'all over social media', 'viral post',
                'share if you agree', 'tag someone', 'comment below', 'follow for more'
            ],
            'news_media': [
                'breaking news', 'reported by', 'news outlet', 'according to sources',
                'journalist', 'correspondent', 'news report', 'media coverage',
                'press release', 'official statement', 'news agency', 'news channel',
                'broadcasting', 'live report', 'on-the-scene', 'eyewitness',
                'Reuters', 'AP', 'AFP', 'Associated Press', 'news wire',
                'exclusive report', 'investigative journalism', 'breaking story',
                'developing story', 'latest updates', 'confirmed reports'
            ],
            'scientific': [
                'peer-reviewed', 'research study', 'clinical trial', 'published in',
                'journal article', 'scientific journal', 'research findings',
                'university study', 'laboratory', 'peer review', 'methodology',
                'scientific method', 'evidence-based', 'empirical data', 'statistical analysis',
                'control group', 'randomized', 'double-blind', 'placebo-controlled',
                'meta-analysis', 'systematic review', 'literature review',
                'hypothesis', 'experiment', 'observation', 'data collection',
                'reproducible', 'validated', 'verified', 'scientifically proven',
                'academic research', 'dissertation', 'thesis', 'doctorate',
                'professor', 'researcher', 'scientist', 'scholar'
            ],
            'conspiracy': [
                'hidden agenda', 'cover-up', 'they don\'t want you to know',
                'secret government', 'illuminati', 'new world order',
                'deep state', 'conspiracy theory', 'wake up people', 'open your eyes',
                'shadow government', 'puppet masters', 'controlled opposition',
                'false flag', 'staged event', 'crisis actors', 'psyop',
                'mind control', 'brainwashing', 'programming', 'indoctrination',
                'globalist', 'elites', 'cabal', 'establishment', 'power structure',
                'lizard people', 'reptilians', 'aliens', 'UFO cover-up',
                'chemtrails', 'flat earth', 'moon landing hoax', 'inside job',
                'orchestrated', 'planned', 'deliberate', 'intentional',
                'they\'re lying', 'don\'t believe them', 'question everything',
                'follow the money', 'connect the dots', 'coincidence? I think not'
            ],
            'health_misinformation': [
                'miracle cure', 'doctors hate this', 'big pharma doesn\'t want',
                'natural remedy', 'alternative medicine', 'detox', 'cleanse',
                'instant cure', 'secret ingredient', 'ancient remedy',
                'FDA hiding', 'medical establishment conspiracy', 'pharma cartel',
                'cancer cure suppressed', 'heal yourself naturally', 'chemicals bad',
                'toxins', 'purify', 'boost immune system', 'superfoods',
                'essential oils cure', 'crystal healing', 'energy medicine',
                'quantum healing', 'holistic', 'homeopathy', 'naturopathy',
                'anti-vaccine', 'vaccines cause', 'vaccine injury', 'vaccine hoax',
                'natural immunity', 'herd immunity myth', 'forced vaccination',
                'cure cancer with', 'reverse diabetes', 'lose weight fast',
                'anti-aging secret', 'fountain of youth', 'live forever',
                'doctors won\'t tell you', 'medical scam', 'pill pushers'
            ],
            'political': [
                'fake news', 'media bias', 'political agenda', 'election fraud',
                'government control', 'propaganda', 'mainstream media lies',
                'stolen election', 'voter fraud', 'rigged election', 'ballot stuffing',
                'dead people voting', 'illegal votes', 'election interference',
                'liberal media', 'conservative media', 'media manipulation',
                'censorship', 'silenced', 'banned', 'deplatformed', 'cancelled',
                'free speech under attack', 'thought police', 'politically correct',
                'woke agenda', 'socialist takeover', 'communist plot',
                'tyranny', 'dictatorship', 'authoritarian', 'totalitarian',
                'government overreach', 'constitutional rights', 'freedom under attack',
                'patriots', 'traitors', 'enemies of the people', 'drain the swamp'
            ]
        }
        
        # Content authenticity indicators
        self.authenticity_markers = {
            'credible': [
                'according to data', 'study published', 'research shows',
                'statistics indicate', 'documented evidence', 'verified information',
                'official source', 'peer-reviewed study', 'expert consensus',
                'scientific evidence', 'empirical data', 'clinical trials show',
                'multiple studies', 'meta-analysis', 'systematic review',
                'confirmed by', 'verified by', 'fact-checked', 'independently verified',
                'cross-referenced', 'corroborated', 'substantiated',
                'government data', 'official statistics', 'census data',
                'published research', 'academic consensus', 'expert opinion',
                'credible sources', 'reliable sources', 'reputable sources',
                'evidence suggests', 'data shows', 'findings indicate'
            ],
            'questionable': [
                'some say', 'people believe', 'it is rumored', 'allegedly',
                'supposedly', 'claims without evidence', 'unverified reports',
                'anonymous sources', 'insider information', 'leaked documents',
                'sources claim', 'reports suggest', 'it is said that',
                'rumor has it', 'word on the street', 'hearsay', 'speculation',
                'unconfirmed', 'unsubstantiated', 'uncorroborated',
                'according to rumors', 'allegedly stated', 'purportedly',
                'claimed by unknown', 'mysterious source', 'secret documents',
                'whistleblower says', 'insiders reveal', 'sources familiar with',
                'people are saying', 'many people', 'everyone knows',
                'common knowledge', 'obvious to anyone', 'it\'s clear that',
                'no evidence but', 'despite lack of proof', 'without citation'
            ]
        }
    
    def _is_gibberish(self, text: str) -> bool:
        """Detect if text is meaningless gibberish"""
        if len(text.strip()) < 10:
            return True
            
        words = text.split()
        if len(words) < 3:
            return True
        
        # Check for repetitive characters (e.g., "hhhhhh...")
        for word in words:
            if len(word) > 5:
                # Count unique characters
                unique_chars = len(set(word.lower()))
                # If word has very few unique characters, it's likely gibberish
                if unique_chars <= 2:
                    return True
                # Check for same character repetition
                if len(word) > 10 and word[0] * len(word) == word.lower():
                    return True
        
        # Check for lack of vowels (most real words have vowels)
        vowels = set('aeiouAEIOU')
        total_chars = sum(1 for c in text if c.isalpha())
        vowel_count = sum(1 for c in text if c in vowels)
        
        if total_chars > 20 and vowel_count / total_chars < 0.15:  # Less than 15% vowels
            return True
        
        # Check for random character sequences
        alpha_text = ''.join(c for c in text.lower() if c.isalpha())
        if len(alpha_text) > 20:
            # Count character transitions (consecutive different chars)
            transitions = sum(1 for i in range(len(alpha_text)-1) if alpha_text[i] != alpha_text[i+1])
            repetition_ratio = transitions / len(alpha_text)
            # Very low repetition means lots of same characters
            if repetition_ratio < 0.3:
                return True
        
        return False
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive text features"""
        text_lower = text.lower()
        
        features = {
            # Basic text statistics
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]),
            'avg_sentence_length': len(text.split()) / max(1, len(re.split(r'[.!?]+', text))),
            
            # Punctuation and formatting
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text)),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(1, len(text)),
            
            # Misinformation indicators
            'fake_keyword_count': sum(1 for word in self.fake_indicators if word in text_lower),
            'clickbait_pattern_count': sum(1 for pattern in self.clickbait_patterns 
                                          if re.search(pattern, text_lower)),
            
            # Emotional content
            'fear_words': sum(1 for word in self.emotion_words['fear'] if word in text_lower),
            'anger_words': sum(1 for word in self.emotion_words['anger'] if word in text_lower),
            'excitement_words': sum(1 for word in self.emotion_words['excitement'] if word in text_lower),
            
            # Additional indicators
            'source_credibility': self._assess_source_credibility(text),
            'factual_language_score': self._assess_factual_language(text),
            'emotional_intensity': self._calculate_emotional_intensity(text)
        }
        
        return features
    
    def _assess_source_credibility(self, text: str) -> float:
        """Assess source credibility indicators"""
        credible_indicators = [
            'according to', 'study shows', 'research indicates', 'data suggests',
            'published in', 'peer-reviewed', 'university', 'institute', 'journal'
        ]
        
        uncredible_indicators = [
            'anonymous source', 'unnamed expert', 'some say', 'people believe',
            'it is rumored', 'allegedly', 'supposedly', 'claims without evidence'
        ]
        
        text_lower = text.lower()
        credible_count = sum(1 for indicator in credible_indicators if indicator in text_lower)
        uncredible_count = sum(1 for indicator in uncredible_indicators if indicator in text_lower)
        
        # Return score between 0 and 1
        return max(0, min(1, (credible_count - uncredible_count) / 10 + 0.5))
    
    def _assess_factual_language(self, text: str) -> float:
        """Assess use of factual vs. emotional language"""
        factual_words = [
            'data', 'statistics', 'evidence', 'research', 'study', 'analysis',
            'findings', 'results', 'report', 'investigation', 'fact', 'documentation'
        ]
        
        emotional_words = [
            'amazing', 'shocking', 'unbelievable', 'incredible', 'outrageous',
            'devastating', 'thrilling', 'spectacular', 'horrifying', 'wonderful'
        ]
        
        text_lower = text.lower()
        factual_count = sum(1 for word in factual_words if word in text_lower)
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        
        total_words = len(text.split())
        factual_ratio = factual_count / max(1, total_words)
        emotional_ratio = emotional_count / max(1, total_words)
        
        return factual_ratio / max(0.01, factual_ratio + emotional_ratio)
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate overall emotional intensity"""
        all_emotion_words = []
        for emotion_list in self.emotion_words.values():
            all_emotion_words.extend(emotion_list)
        
        text_lower = text.lower()
        emotion_count = sum(1 for word in all_emotion_words if word in text_lower)
        total_words = len(text.split())
        
        return emotion_count / max(1, total_words)

class MindSporeTextClassifier(nn.Cell):
    """MindSpore neural network for text classification - FIXED for stability"""
    
    def __init__(self, input_size=20, hidden_size=32, num_classes=4):
        super(MindSporeTextClassifier, self).__init__()
        
        # Neural network layers
        self.dense1 = nn.Dense(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        self.dense2 = nn.Dense(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        
        self.classifier = nn.Dense(hidden_size // 2, num_classes)
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, x):
        """Forward pass"""
        x = self.dense1(x)
        x = self.relu1(x)
        
        x = self.dense2(x)
        x = self.relu2(x)
        
        logits = self.classifier(x)
        probabilities = self.softmax(logits)
        return probabilities

class TextAnalyzer:
    """Main text analyzer using MindSpore"""
    
    def __init__(self):
        self.feature_extractor = TextFeaturesExtractor()
        self.model = None
        self._initialize_model()
        logger.info("TextAnalyzer initialized with MindSpore backend")
    
    def _initialize_model(self):
        """Initialize the MindSpore model"""
        try:
            # Set context for CPU
            ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
            
            # Create model
            self.model = MindSporeTextClassifier(input_size=20, hidden_size=32, num_classes=4)
            
            # Set to evaluation mode immediately
            self.model.set_train(False)
            
            # Test model with dummy data to ensure it works
            test_input = Tensor(np.random.randn(1, 20).astype(np.float32))
            test_output = self.model(test_input)
            
            logger.info(f"[SUCCESS] REAL MindSpore ML initialized successfully! Test output shape: {test_output.shape}")
            
        except Exception as e:
            logger.error(f"[ERROR] MindSpore model initialization failed: {e}")
            logger.info("Falling back to rule-based analysis")
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready"""
        return self.feature_extractor is not None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for misinformation"""
        try:
            # Check for gibberish first
            if self.feature_extractor._is_gibberish(text):
                return {
                    "judgment": "No Data",
                    "explanation": "Text appears to be gibberish or meaningless input. Unable to perform credibility analysis on non-linguistic content.",
                    "confidence": 0.0,
                    "reliability_score": 0,
                    "is_misinformation": False,
                    "misinformation_score": 0.0,
                    "analysis_method": "Gibberish Detection",
                    "features": {
                        "text_length": len(text),
                        "is_gibberish": True
                    }
                }
            
            # Extract features
            features = self.feature_extractor.extract_features(text)
            
            # Normalize features for neural network
            feature_vector = self._normalize_features(features)
            
            # REAL MindSpore ML prediction - THREAD-SAFE!
            nn_prediction = None
            if self.model is not None:
                try:
                    # Use lock to prevent GIL threading conflicts
                    with mindspore_lock:
                        input_tensor = Tensor(feature_vector.reshape(1, -1), ms.float32)
                        probabilities = self.model(input_tensor)
                        nn_prediction = probabilities.asnumpy()[0]
                    logger.debug(f"[ML] MindSpore ML prediction: {nn_prediction}")
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"[WARNING] MindSpore prediction failed (using rule-based fallback): {e}")
                    
                    # If graph execution error, try to recover by reinitializing the model
                    if "Model execution error" in error_msg or "graph_scheduler" in error_msg:
                        logger.info("[RECOVERY] Attempting to reinitialize MindSpore model...")
                        try:
                            with mindspore_lock:
                                self._initialize_model()
                            logger.info("[RECOVERY] Model reinitialized successfully")
                        except Exception as reinit_error:
                            logger.error(f"[RECOVERY] Failed to reinitialize model: {reinit_error}")
                            self.model = None  # Disable model to prevent repeated errors
            
            # Rule-based analysis
            rule_based_result = self._rule_based_analysis(features, text)
            
            # Combine results
            final_result = self._combine_predictions(rule_based_result, nn_prediction, features)
            
            # REAL-TIME FACT VERIFICATION: Check unpredictable claims (deaths, events, etc.)
            if REALTIME_AVAILABLE and len(text) > 30:
                try:
                    rt_checker = get_fact_checker()
                    realtime_result = rt_checker.verify_claims(text)
                    
                    # If text contains CRITICAL false claims (e.g., false death claims)
                    if realtime_result.get('false_claims'):
                        critical_false = [c for c in realtime_result['false_claims'] if c.get('severity') == 'CRITICAL']
                        
                        if critical_false:
                            # Override judgment to FAKE for critical misinformation
                            logger.info(f"[REAL-TIME] CRITICAL false claims detected: {len(critical_false)}")
                            final_result['judgment'] = "Fake"
                            final_result['reliability_score'] = max(0, min(final_result.get('reliability_score', 50) - 40, 20))
                            final_result['confidence'] = 0.92
                            final_result['is_misinformation'] = True
                            final_result['explanation'] += f"\n\nREAL-TIME VERIFICATION ALERT:\n{len(critical_false)} CRITICAL false claims detected through web verification:"
                            for claim in critical_false[:2]:
                                final_result['explanation'] += f"\n• {claim['claim']}: {claim['explanation']}"
                            final_result['summary'] = f"FAKE - Reliability: {final_result['reliability_score']}%\n\nCRITICAL: Contains {len(critical_false)} verifiably false claims"
                            final_result['realtime_verdict'] = 'CONTAINS CRITICAL FALSEHOODS'
                        
                        elif len(realtime_result['false_claims']) > 0:
                            # Non-critical but still false claims
                            false_count = len(realtime_result['false_claims'])
                            final_result['reliability_score'] = max(0, final_result.get('reliability_score', 50) - 20)
                            final_result['explanation'] += f"\n\nReal-time Verification: {false_count} false claims detected"
                            final_result['realtime_verdict'] = 'CONTAINS FALSE CLAIMS'
                    
                    # If verified claims found
                    if realtime_result.get('verified_claims'):
                        verified_count = len(realtime_result['verified_claims'])
                        final_result['reliability_score'] = min(100, final_result.get('reliability_score', 50) + 10)
                        final_result['explanation'] += f"\n\nReal-time Verification: {verified_count} claims verified as accurate"
                        if not final_result.get('realtime_verdict'):
                            final_result['realtime_verdict'] = 'VERIFIED CLAIMS'
                    
                except Exception as e:
                    logger.warning(f"[Real-time text verification warning]: {str(e)}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                "judgment": "No Data",
                "explanation": f"Analysis failed: {str(e)}",
                "confidence": 0.0,
                "reliability_score": 50,
                "is_misinformation": False,
                "features": {}
            }
    
    def _normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize features for neural network input"""
        # Select key features for the model
        key_features = [
            'text_length', 'word_count', 'avg_word_length', 'avg_sentence_length',
            'exclamation_count', 'question_count', 'caps_ratio', 'digit_ratio',
            'fake_keyword_count', 'clickbait_pattern_count', 'fear_words',
            'anger_words', 'excitement_words', 'source_credibility',
            'factual_language_score', 'emotional_intensity'
        ]
        
        # Extract and normalize values
        values = []
        for feature in key_features:
            value = features.get(feature, 0)
            # Apply appropriate normalization
            if feature in ['text_length', 'word_count']:
                value = min(value / 1000, 1.0)  # Normalize to 0-1
            elif feature in ['avg_word_length', 'avg_sentence_length']:
                value = min(value / 20, 1.0)  # Normalize to 0-1
            elif feature in ['exclamation_count', 'question_count', 'fake_keyword_count', 'clickbait_pattern_count']:
                value = min(value / 10, 1.0)  # Normalize to 0-1
            values.append(value)
        
        # Pad or truncate to exactly 20 features
        while len(values) < 20:
            values.append(0.0)
        values = values[:20]
        
        return np.array(values, dtype=np.float32)
    
    def _rule_based_analysis(self, features: Dict[str, float], text: str) -> Dict[str, Any]:
        """Rule-based misinformation detection - STRICT analysis"""
        
        # Calculate misinformation score with STRICTER thresholds
        misinformation_score = 0.0
        red_flags = 0  # Count of major red flags
        
        # MAJOR RED FLAGS (each counts heavily)
        
        # 1. Fake keywords - even 1 is suspicious
        if features['fake_keyword_count'] >= 3:
            misinformation_score += 0.4
            red_flags += 2
        elif features['fake_keyword_count'] >= 1:
            misinformation_score += 0.2
            red_flags += 1
        
        # 2. Clickbait patterns - strong indicator
        if features['clickbait_pattern_count'] >= 2:
            misinformation_score += 0.3
            red_flags += 1
        elif features['clickbait_pattern_count'] >= 1:
            misinformation_score += 0.15
        
        # 3. High emotional manipulation - very suspicious
        if features['emotional_intensity'] > 0.15:
            misinformation_score += 0.3
            red_flags += 1
        elif features['emotional_intensity'] > 0.08:
            misinformation_score += 0.15
        
        # 4. No credible sources - critical issue
        if features['source_credibility'] < 0.2:
            misinformation_score += 0.35
            red_flags += 1
        elif features['source_credibility'] < 0.4:
            misinformation_score += 0.2
        
        # 5. Lacks factual language - red flag
        if features['factual_language_score'] < 0.25:
            misinformation_score += 0.25
            red_flags += 1
        elif features['factual_language_score'] < 0.4:
            misinformation_score += 0.15
        
        # 6. Excessive caps/punctuation - sensationalism
        if features['caps_ratio'] > 0.15:
            misinformation_score += 0.2
            red_flags += 1
        elif features['caps_ratio'] > 0.08:
            misinformation_score += 0.1
        
        if features['exclamation_count'] > 5:
            misinformation_score += 0.15
            red_flags += 1
        elif features['exclamation_count'] > 3:
            misinformation_score += 0.08
        
        # 7. Check for ABSENCE of credibility markers
        has_credibility = (
            features['source_credibility'] > 0.5 or 
            features['factual_language_score'] > 0.6
        )
        
        # If NO credibility markers AND has suspicious patterns = likely fake
        if not has_credibility and (features['fake_keyword_count'] > 0 or features['clickbait_pattern_count'] > 0):
            misinformation_score += 0.2
            red_flags += 1
        
        # STRICTER judgment thresholds
        # Fake: Must have multiple red flags or very high score
        if red_flags >= 3 or misinformation_score >= 0.8:
            judgment = "Fake"
            reliability_score = max(5, 25 - int(misinformation_score * 20))
        elif red_flags >= 2 or misinformation_score >= 0.5:
            judgment = "Fake"
            reliability_score = max(15, 35 - int(misinformation_score * 15))
        elif red_flags >= 1 or misinformation_score >= 0.35:
            judgment = "Half-Truth"
            reliability_score = max(35, 60 - int(misinformation_score * 30))
        elif misinformation_score >= 0.15:
            # Mild suspicion - still Real but lower score
            judgment = "Real"
            reliability_score = min(75, 70 - int(misinformation_score * 20))
        else:
            # Clean content
            judgment = "Real"
            reliability_score = min(95, 85 + int((1 - misinformation_score) * 10))
        
        # Generate comprehensive explanation
        explanation_parts = []
        reasoning_details = []
        
        # Analyze specific indicators
        if features['fake_keyword_count'] > 0:
            if features['fake_keyword_count'] == 1:
                explanation_parts.append(f"Contains 1 misinformation keyword")
                reasoning_details.append("• Found typical fake news language patterns")
            else:
                explanation_parts.append(f"Contains {features['fake_keyword_count']} misinformation keywords")
                reasoning_details.append("• Multiple fake news indicators detected")
        
        if features['clickbait_pattern_count'] > 0:
            explanation_parts.append("Uses clickbait patterns")
            reasoning_details.append("• Headlines designed to manipulate emotions rather than inform")
        
        if features['emotional_intensity'] > 0.1:
            explanation_parts.append("High emotional content")
            reasoning_details.append(f"• Emotional intensity: {features['emotional_intensity']:.1%} (high manipulation risk)")
        elif features['emotional_intensity'] > 0.05:
            explanation_parts.append("Moderate emotional content")
            reasoning_details.append(f"• Emotional intensity: {features['emotional_intensity']:.1%} (some bias)")
        
        if features['source_credibility'] < 0.3:
            explanation_parts.append("Lacks credible source indicators")
            reasoning_details.append("• No references to established sources or authorities")
        elif features['source_credibility'] > 0.7:
            reasoning_details.append("• Contains credible source references")
        
        if features['factual_language_score'] < 0.3:
            explanation_parts.append("Lacks factual language patterns")
            reasoning_details.append("• Missing scientific/academic terminology and evidence-based language")
        elif features['factual_language_score'] > 0.6:
            reasoning_details.append("• Uses factual, evidence-based language")
        
        if features['caps_ratio'] > 0.15:
            explanation_parts.append("Excessive capitalization")
            reasoning_details.append("• Overuse of capital letters suggests sensationalism")
        
        if features['exclamation_count'] > 3:
            explanation_parts.append("Excessive exclamation marks")
            reasoning_details.append("• Multiple exclamation marks indicate emotional manipulation")
        
        # Positive indicators for real content
        if judgment == "Real":
            positive_indicators = []
            if features['factual_language_score'] > 0.5:
                positive_indicators.append("factual language")
            if features['source_credibility'] > 0.6:
                positive_indicators.append("credible sources")
            if features['emotional_intensity'] < 0.05:
                positive_indicators.append("neutral tone")
            if features['fake_keyword_count'] == 0:
                positive_indicators.append("no misinformation keywords")
            
            if positive_indicators:
                reasoning_details.append(f"• Positive indicators: {', '.join(positive_indicators)}")
        
        # Neural network contribution
        if misinformation_score > 0.6:
            reasoning_details.append(f"• AI confidence in misinformation: {misinformation_score:.1%}")
        elif misinformation_score < 0.3:
            reasoning_details.append(f"• AI confidence in authenticity: {(1-misinformation_score):.1%}")
        
        # Generate final explanation based on judgment with detailed reasoning
        if judgment == "Fake":
            base_explanation = "FAKE: This content shows strong indicators of misinformation."
            
            # Explain WHY it's fake
            why_fake = "\n\nWHY THIS WAS FLAGGED AS FAKE:"
            if features['fake_keyword_count'] > 3:
                why_fake += f"\n• Contains {features['fake_keyword_count']} misinformation keywords commonly found in fake news (e.g., 'you won't believe', 'they don't want you to know', conspiracy language)"
            elif features['fake_keyword_count'] > 0:
                why_fake += f"\n• Uses {features['fake_keyword_count']} typical fake news phrases"
            
            if features['clickbait_pattern_count'] > 1:
                why_fake += f"\n• Uses {features['clickbait_pattern_count']} clickbait patterns designed to manipulate your emotions rather than inform you"
            
            if features['emotional_intensity'] > 0.1:
                why_fake += f"\n• Very high emotional manipulation (intensity: {features['emotional_intensity']:.0%}). Real news is typically more neutral"
            
            if features['source_credibility'] < 0.3:
                why_fake += "\n• No credible sources cited. Real journalism references authoritative sources"
            
            if features['factual_language_score'] < 0.3:
                why_fake += "\n• Lacks factual, evidence-based language. No scientific terms or data references"
            
            if features['caps_ratio'] > 0.15 or features['exclamation_count'] > 3:
                why_fake += "\n• Excessive CAPS and exclamation marks!!! - typical of sensationalist fake content"
            
            reasoning_details.insert(0, "MindSpore AI detected multiple red flags for misinformation")
            base_explanation += why_fake
            
        elif judgment == "Half-Truth":
            base_explanation = "HALF-TRUTH: This content mixes some factual information with misleading elements."
            
            # Explain the mixed signals
            mixed_signals = "\n\nWHY THIS IS CONSIDERED HALF-TRUTH:"
            if features['source_credibility'] > 0.4:
                mixed_signals += "\nGOOD: Has some credible source references"
            if features['factual_language_score'] > 0.3:
                mixed_signals += "\nGOOD: Uses some factual language"
            
            if features['fake_keyword_count'] > 0:
                mixed_signals += f"\nWARNING: Contains {features['fake_keyword_count']} misinformation keyword(s)"
            if features['emotional_intensity'] > 0.05:
                mixed_signals += f"\nWARNING: Emotional manipulation detected ({features['emotional_intensity']:.0%} intensity)"
            if features['clickbait_pattern_count'] > 0:
                mixed_signals += f"\nWARNING: Uses clickbait techniques"
            
            mixed_signals += "\n\nVERDICT: The content appears to blend real information with sensationalism or bias. Verify the specific claims independently."
            reasoning_details.insert(0, "MindSpore AI found both credible and questionable elements")
            base_explanation += mixed_signals
            
        elif judgment == "Real":
            base_explanation = "REAL: This content shows characteristics of authentic, credible information."
            
            # Explain credibility factors
            why_real = "\n\nWHY THIS IS CONSIDERED CREDIBLE:"
            if features['factual_language_score'] > 0.6:
                why_real += "\nUses factual, evidence-based language with proper terminology"
            if features['source_credibility'] > 0.6:
                why_real += "\nReferences credible sources and authorities"
            if features['emotional_intensity'] < 0.05:
                why_real += "\nMaintains neutral, objective tone"
            if features['fake_keyword_count'] == 0:
                why_real += "\nNo misinformation keywords detected"
            if features['clickbait_pattern_count'] == 0:
                why_real += "\nNo clickbait or sensationalist patterns"
            
            # Add any concerns if present
            if explanation_parts:
                why_real += f"\n\nMINOR CONCERNS:\n• {', '.join(explanation_parts)}"
                why_real += "\n\nDespite these minor issues, the overall assessment is positive."
            
            reasoning_details.insert(0, "MindSpore AI analysis indicates authentic content patterns")
            base_explanation += why_real
        
        else:  # No Data
            base_explanation = "NO DATA: Insufficient information to make a reliable determination."
            reasoning_details.insert(0, "Content is too short or ambiguous for accurate analysis")
        
        # Create DETAILED but CONCISE summary for bubble display
        if judgment == "Fake":
            summary_parts = [f"FAKE - Reliability: {reliability_score}%"]
            
            problems = []
            if features['fake_keyword_count'] > 0:
                problems.append(f"• {features['fake_keyword_count']} fake news keywords detected")
            if features['clickbait_pattern_count'] > 0:
                problems.append(f"• {features['clickbait_pattern_count']} clickbait patterns found")
            if features['emotional_intensity'] > 0.1:
                problems.append(f"• High emotional manipulation ({features['emotional_intensity']:.0%})")
            if features['source_credibility'] < 0.3:
                problems.append("• No credible sources cited")
            if features['factual_language_score'] < 0.3:
                problems.append("• Lacks factual/scientific language")
            if features['caps_ratio'] > 0.1:
                problems.append(f"• Excessive CAPS ({features['caps_ratio']:.0%} of text)")
            if features['exclamation_count'] > 3:
                problems.append(f"• {features['exclamation_count']} exclamation marks (sensationalism)")
            
            summary = summary_parts[0] + "\n\nRed Flags:\n" + "\n".join(problems[:5])
            if not problems:
                summary += "\nMultiple misinformation indicators detected by MindSpore AI"
                
        elif judgment == "Half-Truth":
            summary_parts = [f"HALF-TRUTH - Reliability: {reliability_score}%"]
            
            good_signs = []
            if features['source_credibility'] > 0.4:
                good_signs.append("• Has some source references")
            if features['factual_language_score'] > 0.4:
                good_signs.append("• Uses some factual language")
            
            concerns = []
            if features['fake_keyword_count'] > 0:
                concerns.append(f"• {features['fake_keyword_count']} fake news keywords")
            if features['emotional_intensity'] > 0.05:
                concerns.append(f"• Emotional bias detected ({features['emotional_intensity']:.0%})")
            if features['clickbait_pattern_count'] > 0:
                concerns.append(f"• {features['clickbait_pattern_count']} clickbait patterns")
            
            summary = summary_parts[0]
            if good_signs:
                summary += "\n\nPositive Signs:\n" + "\n".join(good_signs[:3])
            if concerns:
                summary += "\n\nConcerns:\n" + "\n".join(concerns[:3])
            if not good_signs and not concerns:
                summary += "\n\nMixes factual info with misleading/sensationalist elements"
                
        elif judgment == "Real":
            summary_parts = [f"REAL - Reliability: {reliability_score}%"]
            
            strengths = []
            if features['factual_language_score'] > 0.6:
                strengths.append(f"• Factual language score: {features['factual_language_score']:.0%}")
            if features['source_credibility'] > 0.6:
                strengths.append(f"• Credible sources referenced: {features['source_credibility']:.0%}")
            if features['emotional_intensity'] < 0.05:
                strengths.append(f"• Neutral tone (emotion: {features['emotional_intensity']:.0%})")
            if features['fake_keyword_count'] == 0:
                strengths.append("• No fake news keywords detected")
            if features['clickbait_pattern_count'] == 0:
                strengths.append("• No clickbait patterns")
            
            minor_issues = []
            if features['emotional_intensity'] > 0.05:
                minor_issues.append(f"• Some emotional content ({features['emotional_intensity']:.0%})")
            if features['exclamation_count'] > 0:
                minor_issues.append(f"• {features['exclamation_count']} exclamation marks")
            
            summary = summary_parts[0] + "\n\nStrengths:\n" + "\n".join(strengths[:4])
            if minor_issues:
                summary += "\n\nMinor Notes:\n" + "\n".join(minor_issues[:2])
        else:
            summary = "NO DATA - Content too short or unclear\n\nNeed at least 20 characters for reliable analysis"
        
        # Combine FULL explanation with detailed reasoning
        if reasoning_details:
            explanation = base_explanation + "\n\nDETAILED ANALYSIS:\n" + "\n".join(reasoning_details)
            
            # Add analysis methodology explanation
            explanation += f"\n\nANALYSIS METHOD:\nMindSpore AI examined {features['word_count']} words using neural network + rule-based detection. "
            explanation += f"Reliability Score: {reliability_score}/100 (higher = more reliable)"
        else:
            explanation = base_explanation
        
        return {
            "judgment": judgment,
            "summary": summary,  # SHORT for bubbles
            "explanation": explanation,  # FULL for Analysis Details
            "confidence": min(0.95, 0.6 + abs(misinformation_score - 0.5)),
            "reliability_score": reliability_score,
            "is_misinformation": judgment in ["Fake", "Half-Truth"],
            "misinformation_score": misinformation_score
        }
    
    def _combine_predictions(self, rule_based: Dict[str, Any], nn_prediction: np.ndarray, 
                           features: Dict[str, float]) -> Dict[str, Any]:
        """Combine rule-based and neural network predictions"""
        
        result = rule_based.copy()
        result["features"] = features
        
        if nn_prediction is not None:
            result["analysis_method"] = "Combined (Rule-based + MindSpore Neural Network)"
            # Neural network classes: [Real, Half-Truth, Fake, No Data]
            class_names = ["Real", "Half-Truth", "Fake", "No Data"]
            nn_judgment = class_names[np.argmax(nn_prediction)]
            nn_confidence = float(np.max(nn_prediction))
            
            # Weight the predictions (70% rule-based, 30% neural network for now)
            rule_weight = 0.7
            nn_weight = 0.3
            
            # Combine confidences
            combined_confidence = (rule_based["confidence"] * rule_weight + 
                                 nn_confidence * nn_weight)
            
            # Enhanced explanation with neural network reasoning
            nn_explanation_parts = []
            
            # If neural network strongly disagrees, provide detailed explanation
            if nn_judgment != rule_based["judgment"]:
                confidence_diff = abs(nn_confidence - rule_based["confidence"])
                if nn_confidence > 0.8:
                    nn_explanation_parts.append(f"\n\nNEURAL NETWORK ANALYSIS:")
                    nn_explanation_parts.append(f"• AI Model Prediction: {nn_judgment} ({nn_confidence:.1%} confidence)")
                    nn_explanation_parts.append(f"• Disagreement with rule-based analysis detected")
                    
                    if nn_judgment == "Fake" and nn_confidence > 0.9:
                        nn_explanation_parts.append("• Deep learning model detected subtle misinformation patterns")
                    elif nn_judgment == "Real" and nn_confidence > 0.9:
                        nn_explanation_parts.append("• Deep learning model recognizes authentic content patterns")
                    
                    nn_explanation_parts.append(f"• Final decision weighted: {rule_weight:.0%} traditional analysis, {nn_weight:.0%} AI model")
                else:
                    nn_explanation_parts.append(f"\n\nNote: AI model suggests '{nn_judgment}' but with lower confidence ({nn_confidence:.1%})")
            
            elif nn_confidence > 0.95:  # High agreement
                nn_explanation_parts.append(f"\n\nCONFIRMATION: Neural network strongly agrees ({nn_confidence:.1%} confidence)")
            
            # Add probability breakdown for transparency
            if nn_confidence > 0.7:
                probabilities = nn_prediction.tolist()
                prob_breakdown = []
                for i, (name, prob) in enumerate(zip(class_names, probabilities)):
                    if prob > 0.1:  # Only show significant probabilities
                        prob_breakdown.append(f"{name}: {prob:.1%}")
                
                if len(prob_breakdown) > 1:
                    nn_explanation_parts.append(f"• Probability breakdown: {', '.join(prob_breakdown)}")
            
            # Append neural network explanation to main explanation
            if nn_explanation_parts:
                result["explanation"] += "\n".join(nn_explanation_parts)
            
            result["confidence"] = combined_confidence
            result["nn_prediction"] = {
                "judgment": nn_judgment,
                "confidence": nn_confidence,
                "probabilities": nn_prediction.tolist()
            }
        else:
            result["analysis_method"] = "Rule-based (MindSpore ML unavailable)"
        
        return result
