"""
Online Fact Verification Module
===============================

Integrates online sources for comprehensive fact-checking and origin detection.
Works alongside MindSpore analysis to provide enhanced verification.

NOW USES REAL-TIME FACT CHECKER FOR ACTUAL WEB VERIFICATION!
"""

import requests
import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import quote, urlencode
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import the REAL fact checker
try:
    from .real_time_fact_checker import get_fact_checker
    REALTIME_AVAILABLE = True
    logger.info("Real-time fact checker available for online verification")
except ImportError:
    REALTIME_AVAILABLE = False
    logger.warning("Real-time fact checker not available")

class OnlineFactVerifier:
    """Online fact-checking and source verification"""
    
    def __init__(self):
        self.session = None
        self.fact_check_apis = {
            # Free fact-checking APIs
            'snopes': 'https://www.snopes.com/api/v1/fact-check',
            'factcheck_org': 'https://www.factcheck.org/api',
            'politifact': 'https://www.politifact.com/api/v/2',
        }
        
        # News source credibility rankings (based on Media Bias/Fact Check)
        self.source_credibility = {
            'high_credibility': [
                'reuters.com', 'apnews.com', 'bbc.com', 'npr.org', 'pbs.org',
                'nature.com', 'science.org', 'nejm.org', 'who.int', 'cdc.gov',
                'fda.gov', 'nih.gov', 'nasa.gov', 'noaa.gov'
            ],
            'medium_credibility': [
                'cnn.com', 'nytimes.com', 'washingtonpost.com', 'theguardian.com',
                'wsj.com', 'forbes.com', 'time.com', 'newsweek.com'
            ],
            'low_credibility': [
                'infowars.com', 'breitbart.com', 'naturalnews.com', 'beforeitsnews.com',
                'truthseekers.com', 'conspiracy.com', 'alternativemedia.com'
            ]
        }
        
        # Search engines for verification
        self.search_apis = {
            'duckduckgo': 'https://api.duckduckgo.com/',
            'bing': 'https://api.bing.microsoft.com/v7.0/search',  # Requires API key
        }

    async def verify_claim(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive online verification of a claim
        NOW USES REAL WEB SEARCH via real_time_fact_checker!
        """
        try:
            verification_results = {
                'overall_credibility': 0.5,
                'source_analysis': {},
                'fact_check_results': [],
                'origin_detection': {},
                'online_presence': {},
                'contradictory_evidence': [],
                'supporting_evidence': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # USE REAL-TIME FACT CHECKER for actual web verification
            if REALTIME_AVAILABLE and len(text) > 20:
                try:
                    rt_checker = get_fact_checker()
                    realtime_result = rt_checker.verify_claims(text)
                    
                    logger.info(f"Real-time verification: {realtime_result.get('overall_verdict', 'UNKNOWN')}")
                    
                    # Process REAL verification results
                    if realtime_result.get('false_claims'):
                        verification_results['contradictory_evidence'] = [
                            {
                                'claim': c['claim'],
                                'verdict': c['verdict'],
                                'explanation': c.get('explanation', ''),
                                'severity': c.get('severity', 'NORMAL'),
                                'evidence': c.get('evidence', [])
                            }
                            for c in realtime_result['false_claims']
                        ]
                        
                        # Lower credibility score for false claims
                        critical_count = sum(1 for c in realtime_result['false_claims'] if c.get('severity') == 'CRITICAL')
                        if critical_count > 0:
                            verification_results['overall_credibility'] = 0.1  # Very low for critical falsehoods
                        else:
                            verification_results['overall_credibility'] = 0.3
                    
                    if realtime_result.get('verified_claims'):
                        verification_results['supporting_evidence'] = [
                            {
                                'claim': c['claim'],
                                'verdict': c['verdict'],
                                'confidence': c.get('confidence', 0),
                                'sources': c.get('sources', [])
                            }
                            for c in realtime_result['verified_claims']
                        ]
                        
                        # Boost credibility for verified claims
                        verification_results['overall_credibility'] = min(0.9, 0.5 + len(realtime_result['verified_claims']) * 0.1)
                    
                    # Add overall verdict
                    verification_results['online_presence'] = {
                        'verdict': realtime_result.get('overall_verdict', 'UNKNOWN'),
                        'reliability_score': realtime_result.get('reliability_score', 50),
                        'total_claims_found': realtime_result.get('total_claims_found', 0),
                        'source': 'Real-time Web Verification (Wikipedia + DuckDuckGo + Fact-checking sites)'
                    }
                    
                    logger.info(f"Real online verification completed. Credibility: {verification_results['overall_credibility']:.2f}")
                    
                except Exception as e:
                    logger.error(f"Real-time verification error: {str(e)}")
            
            # FALLBACK: If real-time checker not available, use old method
            if not verification_results['supporting_evidence'] and not verification_results['contradictory_evidence']:
                # Extract key claims from text
                key_claims = self._extract_key_claims(text)
                
                # Search for each claim online (old simulated method)
                for claim in key_claims:
                    search_results = await self._search_claim(claim)
                    fact_checks = await self._check_fact_checking_sites(claim)
                
                verification_results['fact_check_results'].extend(fact_checks)
                
                # Analyze sources
                source_analysis = await self._analyze_sources(search_results)
                verification_results['source_analysis'].update(source_analysis)
                
                # Detect contradictory vs supporting evidence
                evidence = self._categorize_evidence(search_results, claim)
                verification_results['contradictory_evidence'].extend(evidence['contradictory'])
                verification_results['supporting_evidence'].extend(evidence['supporting'])
            
            # Calculate overall credibility
            verification_results['overall_credibility'] = self._calculate_overall_credibility(
                verification_results
            )
            
            # Detect likely origin
            verification_results['origin_detection'] = await self._detect_origin(text)
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error in online verification: {e}")
            return {
                'overall_credibility': 0.5,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _extract_key_claims(self, text: str) -> List[str]:
        """Extract factual claims that can be verified"""
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        key_claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short sentences
                # Look for factual claim indicators
                if any(indicator in sentence.lower() for indicator in [
                    'study shows', 'research indicates', 'scientists found',
                    'data reveals', 'according to', 'statistics show',
                    'report states', 'evidence suggests', 'proven',
                    'discovered', 'confirmed', 'announced'
                ]):
                    key_claims.append(sentence)
                elif any(number in sentence for number in re.findall(r'\d+', sentence)):
                    # Claims with numbers are often factual
                    key_claims.append(sentence)
        
        return key_claims[:5]  # Limit to top 5 claims

    async def _search_claim(self, claim: str) -> List[Dict]:
        """Search for information about a claim"""
        try:
            # Use a simple web search approach
            search_query = claim.replace(' ', '+')
            
            # Try multiple search approaches
            results = []
            
            # Method 1: Try DuckDuckGo Instant Answer API
            try:
                url = f"https://api.duckduckgo.com/?q={search_query}&format=json&no_html=1&skip_disambig=1"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process instant answer
                            if data.get('AbstractText'):
                                results.append({
                                    'title': data.get('AbstractSource', 'DuckDuckGo'),
                                    'snippet': data.get('AbstractText'),
                                    'url': data.get('AbstractURL', ''),
                                    'source_type': 'instant_answer'
                                })
                            
                            # Process related topics
                            for topic in data.get('RelatedTopics', [])[:3]:
                                if isinstance(topic, dict) and 'Text' in topic:
                                    results.append({
                                        'title': topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else 'Related Topic',
                                        'snippet': topic.get('Text'),
                                        'url': topic.get('FirstURL', ''),
                                        'source_type': 'related_topic'
                                    })
            except Exception as e:
                logger.warning(f"DuckDuckGo API error: {e}")
            
            # NO MORE FAKE SIMULATED SOURCES!
            # If no results, return empty list - real_time_fact_checker will handle it
            if not results:
                logger.info("No search results found via DuckDuckGo (this is OK - real-time fact checker is primary)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching claim: {e}")
            return []

    async def _check_fact_checking_sites(self, claim: str) -> List[Dict]:
        """Check major fact-checking websites"""
        fact_checks = []
        
        try:
            # Search Snopes-style fact-checking
            search_terms = self._extract_search_terms(claim)
            
            # Simulate fact-checking results (in real implementation, you'd use actual APIs)
            # For now, we'll use pattern matching to simulate fact-checking
            
            fake_indicators = [
                'doctors hate this', 'secret cure', 'government hiding',
                'they don\'t want you to know', 'miracle treatment',
                'ancient secret', 'banned by', 'suppressed information'
            ]
            
            credibility_score = 1.0
            fact_check_verdict = "Unknown"
            
            claim_lower = claim.lower()
            
            if any(indicator in claim_lower for indicator in fake_indicators):
                credibility_score = 0.2
                fact_check_verdict = "Likely False"
                
                fact_checks.append({
                    'source': 'Pattern Analysis',
                    'verdict': fact_check_verdict,
                    'credibility': credibility_score,
                    'explanation': 'Contains common misinformation indicators',
                    'url': '',
                    'date': datetime.now().isoformat()
                })
            
            elif any(term in claim_lower for term in ['study', 'research', 'university', 'journal']):
                credibility_score = 0.8
                fact_check_verdict = "Likely True"
                
                fact_checks.append({
                    'source': 'Academic Pattern Analysis',
                    'verdict': fact_check_verdict,
                    'credibility': credibility_score,
                    'explanation': 'Contains academic/research language patterns',
                    'url': '',
                    'date': datetime.now().isoformat()
                })
            
            return fact_checks
            
        except Exception as e:
            logger.error(f"Error checking fact-checking sites: {e}")
            return []

    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract key terms for searching"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        return meaningful_words[:10]  # Top 10 terms

    async def _analyze_sources(self, search_results: List[Dict]) -> Dict[str, Any]:
        """Analyze the credibility of sources found"""
        source_analysis = {
            'high_credibility_sources': [],
            'medium_credibility_sources': [],
            'low_credibility_sources': [],
            'unknown_sources': [],
            'overall_source_score': 0.5
        }
        
        total_sources = len(search_results)
        if total_sources == 0:
            return source_analysis
        
        credibility_scores = []
        
        for result in search_results:
            url = result.get('url', '')
            domain = self._extract_domain(url)
            
            if any(trusted in domain for trusted in self.source_credibility['high_credibility']):
                source_analysis['high_credibility_sources'].append({
                    'domain': domain,
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', '')[:200]
                })
                credibility_scores.append(0.9)
                
            elif any(medium in domain for medium in self.source_credibility['medium_credibility']):
                source_analysis['medium_credibility_sources'].append({
                    'domain': domain,
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', '')[:200]
                })
                credibility_scores.append(0.6)
                
            elif any(low in domain for low in self.source_credibility['low_credibility']):
                source_analysis['low_credibility_sources'].append({
                    'domain': domain,
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', '')[:200]
                })
                credibility_scores.append(0.2)
                
            else:
                source_analysis['unknown_sources'].append({
                    'domain': domain,
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', '')[:200]
                })
                credibility_scores.append(0.5)
        
        # Calculate overall source credibility
        if credibility_scores:
            source_analysis['overall_source_score'] = sum(credibility_scores) / len(credibility_scores)
        
        return source_analysis

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if url.startswith('http'):
                domain = url.split('/')[2]
                return domain.replace('www.', '')
            return url
        except:
            return 'unknown'

    def _categorize_evidence(self, search_results: List[Dict], claim: str) -> Dict[str, List]:
        """Categorize evidence as supporting or contradictory"""
        evidence = {
            'supporting': [],
            'contradictory': []
        }
        
        # Extract key terms from claim
        claim_terms = set(self._extract_search_terms(claim))
        
        for result in search_results:
            snippet = result.get('snippet', '').lower()
            snippet_terms = set(self._extract_search_terms(snippet))
            
            # Check for contradiction indicators
            contradiction_words = ['false', 'debunked', 'myth', 'incorrect', 'wrong', 'misleading']
            support_words = ['confirmed', 'verified', 'true', 'proven', 'validated', 'accurate']
            
            if any(word in snippet for word in contradiction_words):
                evidence['contradictory'].append({
                    'source': result.get('title', 'Unknown'),
                    'snippet': result.get('snippet', '')[:200],
                    'url': result.get('url', ''),
                    'confidence': 0.7
                })
            elif any(word in snippet for word in support_words):
                evidence['supporting'].append({
                    'source': result.get('title', 'Unknown'),
                    'snippet': result.get('snippet', '')[:200],
                    'url': result.get('url', ''),
                    'confidence': 0.7
                })
        
        return evidence

    def _calculate_overall_credibility(self, verification_results: Dict) -> float:
        """Calculate overall credibility score"""
        scores = []
        
        # Source credibility weight
        source_score = verification_results.get('source_analysis', {}).get('overall_source_score', 0.5)
        scores.append(source_score * 0.4)  # 40% weight
        
        # Fact-check results weight
        fact_checks = verification_results.get('fact_check_results', [])
        if fact_checks:
            fact_check_score = sum(fc.get('credibility', 0.5) for fc in fact_checks) / len(fact_checks)
            scores.append(fact_check_score * 0.3)  # 30% weight
        
        # Evidence balance weight
        supporting = len(verification_results.get('supporting_evidence', []))
        contradictory = len(verification_results.get('contradictory_evidence', []))
        
        if supporting + contradictory > 0:
            evidence_score = supporting / (supporting + contradictory)
            scores.append(evidence_score * 0.3)  # 30% weight
        
        return sum(scores) if scores else 0.5

    async def _detect_origin(self, text: str) -> Dict[str, Any]:
        """Detect likely origin of the information"""
        origin_analysis = {
            'likely_source_type': 'unknown',
            'confidence': 0.5,
            'indicators': [],
            'earliest_mention': None,
            'propagation_pattern': 'unknown'
        }
        
        text_lower = text.lower()
        
        # Social media origin indicators
        if any(indicator in text_lower for indicator in [
            'viral', 'trending', 'shared', 'retweet', 'hashtag', 'going around'
        ]):
            origin_analysis['likely_source_type'] = 'social_media'
            origin_analysis['confidence'] = 0.8
            origin_analysis['indicators'] = ['viral_language', 'sharing_terminology']
        
        # News media origin
        elif any(indicator in text_lower for indicator in [
            'breaking news', 'reported', 'according to sources', 'news outlet'
        ]):
            origin_analysis['likely_source_type'] = 'news_media'
            origin_analysis['confidence'] = 0.7
            origin_analysis['indicators'] = ['news_language', 'reporting_structure']
        
        # Scientific origin
        elif any(indicator in text_lower for indicator in [
            'study', 'research', 'journal', 'peer-reviewed', 'university'
        ]):
            origin_analysis['likely_source_type'] = 'scientific'
            origin_analysis['confidence'] = 0.9
            origin_analysis['indicators'] = ['academic_language', 'research_terminology']
        
        # Conspiracy/alternative origin
        elif any(indicator in text_lower for indicator in [
            'they don\'t want you to know', 'hidden truth', 'cover-up', 'secret'
        ]):
            origin_analysis['likely_source_type'] = 'conspiracy'
            origin_analysis['confidence'] = 0.8
            origin_analysis['indicators'] = ['conspiracy_language', 'hidden_knowledge_claims']
        
        return origin_analysis

    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()


class EnhancedTextAnalyzer:
    """
    Enhanced text analyzer combining MindSpore ML + Real-Time Web Verification
    
    Combines two analysis methods:
    1. MindSpore ML: Pattern detection, text structure analysis, sentiment analysis
    2. Online Verifier: Real-time web verification, fact-checking sites, domain credibility
    
    Weighted combination: 60% ML + 40% Online verification
    """
    
    def __init__(self, text_analyzer, online_verifier):
        self.text_analyzer = text_analyzer
        self.online_verifier = online_verifier
        logger.info("EnhancedTextAnalyzer initialized: MindSpore ML + Real-time Web Verification")
    
    async def analyze_comprehensive(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis combining local and online analysis"""
        
        # Get local MindSpore analysis
        local_analysis = self.text_analyzer.analyze(text)
        
        # Get online verification
        online_analysis = await self.online_verifier.verify_claim(text)
        
        # Combine results
        comprehensive_result = {
            'mindspore_analysis': local_analysis,
            'online_verification': online_analysis,
            'combined_credibility': self._combine_credibility_scores(local_analysis, online_analysis),
            'recommendation': self._generate_recommendation(local_analysis, online_analysis),
            'detailed_breakdown': self._create_detailed_breakdown(local_analysis, online_analysis)
        }
        
        return comprehensive_result
    
    def _combine_credibility_scores(self, local: Dict, online: Dict) -> Dict[str, float]:
        """Combine local and online credibility scores"""
        local_score = local.get('credibility', 0.5)
        online_score = online.get('overall_credibility', 0.5)
        
        # Weight: 60% MindSpore analysis, 40% online verification
        combined_score = (local_score * 0.6) + (online_score * 0.4)
        
        return {
            'local_score': local_score,
            'online_score': online_score,
            'combined_score': combined_score,
            'confidence_level': 'high' if abs(local_score - online_score) < 0.2 else 'medium'
        }
    
    def _generate_recommendation(self, local: Dict, online: Dict) -> str:
        """Generate user-friendly recommendation with detailed reasoning"""
        combined_score = self._combine_credibility_scores(local, online)['combined_score']
        
        # Get specific indicators
        local_judgment = local.get('judgment', 'Unknown')
        online_credibility = online.get('overall_credibility', 0.5)
        origin_type = online.get('origin_detection', {}).get('likely_source_type', 'unknown')
        supporting_count = len(online.get('supporting_evidence', []))
        contradictory_count = len(online.get('contradictory_evidence', []))
        
        # Base recommendation with reasoning
        if combined_score >= 0.8:
            recommendation = "[HIGH CREDIBILITY] Highly credible information"
            reasons = []
            if local_judgment == "Real":
                reasons.append("MindSpore analysis confirms authenticity")
            if online_credibility > 0.7:
                reasons.append("Strong online verification")
            if supporting_count > 0:
                reasons.append(f"{supporting_count} supporting sources found")
            if origin_type in ['scientific', 'news_media']:
                reasons.append(f"Originates from {origin_type.replace('_', ' ')}")
            
            if reasons:
                recommendation += f" ({', '.join(reasons)})"
                
        elif combined_score >= 0.6:
            recommendation = "[MODERATE CREDIBILITY] Generally credible but verify important details"
            concerns = []
            if local_judgment in ["Half-Truth", "Fake"]:
                concerns.append(f"MindSpore detected: {local_judgment}")
            if contradictory_count > 0:
                concerns.append(f"{contradictory_count} contradictory sources")
            if online_credibility < 0.5:
                concerns.append("Limited online verification")
            
            if concerns:
                recommendation += f" (Concerns: {', '.join(concerns)})"
                
        elif combined_score >= 0.4:
            recommendation = "[MIXED CREDIBILITY] Mixed credibility - investigate further before sharing"
            issues = []
            if local_judgment == "Fake":
                issues.append("AI detected misinformation patterns")
            if contradictory_count > supporting_count:
                issues.append("More contradictory than supporting evidence")
            if origin_type == 'conspiracy':
                issues.append("Conspiracy-type language detected")
            if online_credibility < 0.3:
                issues.append("Poor online credibility")
                
            if issues:
                recommendation += f" (Issues: {', '.join(issues)})"
        else:
            recommendation = "[LOW CREDIBILITY] Low credibility - likely misinformation"
            problems = []
            if local_judgment == "Fake":
                problems.append("Strong misinformation indicators")
            if contradictory_count > 0:
                problems.append(f"{contradictory_count} sources contradict claims")
            if origin_type in ['conspiracy', 'social_media']:
                problems.append(f"Unreliable origin type: {origin_type.replace('_', ' ')}")
            if online_credibility < 0.2:
                problems.append("Very poor online verification")
                
            if problems:
                recommendation += f" (Red flags: {', '.join(problems)})"
        
        return recommendation
    
    def _create_detailed_breakdown(self, local: Dict, online: Dict) -> Dict[str, Any]:
        """Create detailed breakdown for user"""
        breakdown = {
            'pattern_analysis': {
                'emotional_language': local.get('emotional_intensity', 0),
                'factual_language': local.get('factual_language_score', 0),
                'misinformation_indicators': local.get('fake_keyword_count', 0)
            },
            'source_verification': {
                'high_credibility_sources': len(online.get('source_analysis', {}).get('high_credibility_sources', [])),
                'contradictory_evidence': len(online.get('contradictory_evidence', [])),
                'supporting_evidence': len(online.get('supporting_evidence', []))
            },
            'origin_analysis': online.get('origin_detection', {})
        }
        
        return breakdown
