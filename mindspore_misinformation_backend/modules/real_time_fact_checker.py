"""
Real-Time Fact Checker with Web Search Integration
Verifies factual claims against live web sources and knowledge bases
"""

import requests
import re
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RealTimeFactChecker:
    """
    Real-time fact checker that searches the web to verify claims
    Addresses: "WHAT IF SOMEONE MENTIONED SOMEONE AS DEAD BUT IN REAL LIFE HIS ALIVE?"
    """
    
    def __init__(self):
        self.cache = {}  # Simple cache to avoid repeated queries
        self.cache_ttl = 3600  # 1 hour cache
        self.headers = {
            'User-Agent': 'MindSpore Misinformation Detector/1.0 (Educational Project)'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Fact-checking sources
        self.fact_check_sources = {
            'Snopes': 'https://www.snopes.com',
            'FactCheck.org': 'https://www.factcheck.org',
            'PolitiFact': 'https://www.politifact.com',
            'Reuters Fact Check': 'https://www.reuters.com/fact-check'
        }
    
    def extract_verifiable_claims(self, text: str) -> List[Dict[str, str]]:
        """
        Extract specific verifiable claims from text
        Focus on: names, dates, events, statistics, statements
        """
        claims = []
        
        # 1. DETECT DEATH CLAIMS
        death_patterns = [
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:died|dead|deceased|killed|passed away)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:is|was|has been|got|became)\s+(?:dead|deceased|killed)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:death|funeral|obituary)',
            r'(?:death of|RIP|rest in peace)\s+([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)',
            r'Breaking(?:\s+news)?[:\s]+([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+died',
            r'([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*)\s+(?:DIED|DEAD|DECEASED|KILLED)',
        ]
        
        for pattern in death_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                person_name = match.group(1).strip()
                claims.append({
                    'type': 'death_claim',
                    'subject': person_name,
                    'claim': f"{person_name} is dead/deceased",
                    'context': match.group(0),
                    'verifiable': True,
                    'critical': True
                })
        
        # 1B. DETECT ALIVE CLAIMS (opposite of death claims)
        alive_patterns = [
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:is|are)\s+(?:still\s+)?(?:alive|living|not dead)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:is|are)\s+(?:alive|living)\s+(?:and|,)',
            r'(?:sure|certain|confirmed)\s+(?:that\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:is|are)\s+(?:alive|living|not dead)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:survived|is surviving|lives on|is living)',
        ]
        
        for pattern in alive_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                person_name = match.group(1).strip()
                claims.append({
                    'type': 'alive_claim',
                    'subject': person_name,
                    'claim': f"{person_name} is alive/living",
                    'context': match.group(0),
                    'verifiable': True,
                    'critical': True
                })
        
        # 2. DETECT SPECIFIC STATISTICS
        stat_patterns = [
            r'(\d+(?:\.\d+)?%)\s+of\s+([\w\s]+)',
            r'([\w\s]+)\s+(?:increased|decreased|rose|fell)\s+by\s+(\d+(?:\.\d+)?%)',
            r'(\$[\d,]+(?:\.\d+)?)\s+([\w\s]+)'
        ]
        
        for pattern in stat_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append({
                    'type': 'statistic',
                    'claim': match.group(0),
                    'context': match.group(0),
                    'verifiable': True,
                    'critical': False
                })
        
        # 3. DETECT DATE-SPECIFIC EVENTS
        event_date_pattern = r'(?:on|in)\s+(\w+\s+\d{1,2},?\s+\d{4})\s*[,.]?\s*([\w\s]+)'
        matches = re.finditer(event_date_pattern, text, re.IGNORECASE)
        for match in matches:
            date = match.group(1)
            event = match.group(2)[:100]
            claims.append({
                'type': 'dated_event',
                'date': date,
                'claim': f"Event on {date}: {event}",
                'context': match.group(0),
                'verifiable': True,
                'critical': False
            })
        
        # 4. DETECT ATTRIBUTION STATEMENTS ("X said Y")
        attribution_pattern = r'([\w\s]+?)\s+(?:said|stated|claimed|announced|declared)\s+(?:that\s+)?["\']?(.+?)["\']?(?:\.|,|$)'
        matches = re.finditer(attribution_pattern, text, re.IGNORECASE)
        for match in matches:
            speaker = match.group(1).strip()
            statement = match.group(2).strip()[:200]
            claims.append({
                'type': 'attribution',
                'speaker': speaker,
                'claim': f"{speaker} said: {statement}",
                'context': match.group(0),
                'verifiable': True,
                'critical': False
            })
        
        # 5. DETECT LOCATION-BASED CLAIMS
        location_pattern = r'(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[,\s]+([A-Z][a-z]+)'
        matches = re.finditer(location_pattern, text)
        for match in matches:
            claims.append({
                'type': 'location',
                'claim': match.group(0),
                'context': match.group(0),
                'verifiable': True,
                'critical': False
            })
        
        # 6. DETECT WAR/CONFLICT CLAIMS (CRITICAL)
        war_patterns = [
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:invaded|invading|will invade|attacked|bombing|declared war)',
            r'(?:war|conflict|invasion|attack)\s+(?:in|on|against)\s+([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)',
            r'([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*)\s+(?:INVADED|INVADING|WILL INVADE|ATTACKED|DECLARED WAR)',
            r'(?:military|army|troops)\s+(?:from|of)\s+([A-Z][A-Za-z]+)\s+(?:invaded|attacked|entered)'
        ]
        
        for pattern in war_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                country_or_entity = match.group(1).strip() if match.lastindex >= 1 else "Unknown"
                claims.append({
                    'type': 'war_claim',
                    'subject': country_or_entity,
                    'claim': f"War/conflict involving {country_or_entity}",
                    'context': match.group(0),
                    'verifiable': True,
                    'critical': True
                })
        
        # 7. DETECT TERRORIST ATTACK CLAIMS (CRITICAL)
        terrorist_patterns = [
            r'terrorist\s+attack\s+(?:in|at|on)\s+([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:terrorist|terror)\s+(?:attack|bombing|incident)',
            r'(?:bomb|bombing|explosion)\s+(?:in|at)\s+([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)\s+(?:carried out|conducted|claimed)\s+(?:terrorist|terror)\s+attack',
            r'TERRORIST\s+ATTACK\s+(?:IN|AT|ON)\s+([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*)'
        ]
        
        for pattern in terrorist_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                location_or_group = match.group(1).strip() if match.lastindex >= 1 else "Unknown"
                claims.append({
                    'type': 'terrorist_claim',
                    'subject': location_or_group,
                    'claim': f"Terrorist attack involving {location_or_group}",
                    'context': match.group(0),
                    'verifiable': True,
                    'critical': True
                })
        
        # 8. DETECT MEDICAL/HEALTH CLAIMS (HIGH PRIORITY)
        diseases = r'(?:cancer|COVID(?:-19)?|diabetes|heart disease|AIDS|HIV|stroke|alzheimer|parkinsons?|asthma|arthritis|tuberculosis|TB|malaria|dengue|hepatitis|influenza|flu|pneumonia|bronchitis|COPD|hypertension|high blood pressure|kidney disease|liver disease|brain tumor|leukemia|lymphoma|melanoma|skin cancer|lung cancer|breast cancer|prostate cancer|colon cancer|autism|ADHD|depression|anxiety|schizophrenia|bipolar|dementia|epilepsy|multiple sclerosis|MS|lupus|Crohn\'?s disease|ulcerative colitis|IBS|celiac disease|fibromyalgia|chronic fatigue|Lyme disease|West Nile|Ebola|Zika|SARS|MERS|bird flu|swine flu|smallpox|polio|measles|mumps|rubella|chickenpox|shingles|herpes|HPV|chlamydia|gonorrhea|syphilis|meningitis|encephalitis|sepsis|anemia|hemophilia|sickle cell|thalassemia|osteoporosis|gout|psoriasis|eczema|acne|rosacea|vitiligo|alopecia|[\w\s]+disease|[\w\s]+virus|[\w\s]+infection|[\w\s]+syndrome|[\w\s]+disorder)'
        
        medical_patterns = [
            rf'([\w\s]+)\s+(?:cures?|treats?|prevents?|causes?|stops?|reverses?|eliminates?)\s+{diseases}',
            rf'([\w\s]+)\s+(?:vaccine|drug|medication|treatment|remedy|therapy|pill|injection|supplement|herb|oil|extract)\s+(?:for|against|treats?|cures?|prevents?)\s+{diseases}',
            rf'(\d+(?:\.\d+)?%)\s+(?:of people|effective|success rate|cure rate|survival rate|recovery rate)\s+(?:with|for|against|from)\s+{diseases}',
            rf'(?:new|breakthrough|miracle|natural|alternative|home|herbal|ancient)\s+(?:cure|treatment|vaccine|remedy|therapy|solution)\s+(?:for|against)\s+{diseases}',
            rf'([\w\s]+)\s+(?:is safe|is dangerous|has side effects|is toxic|is harmful|causes)\s+(?:for|in|to)\s+(?:humans?|people|children|patients?|adults?|elderly|pregnant women)',
            rf'{diseases}\s+(?:can be cured|is curable|is incurable|is deadly|is fatal|kills|causes death|has no cure|has a cure)',
            rf'(?:eating|drinking|taking|consuming|using|applying)\s+([\w\s]+)\s+(?:cures?|prevents?|treats?|causes?|gives you|leads to)\s+{diseases}',
            rf'([\w\s]+)\s+(?:is approved by|was approved by|banned by|rejected by|endorsed by|recommended by|warned against by)\s+(?:FDA|WHO|CDC|NIH|health authorities|doctors?|scientists?|medical community)',
            rf'(?:vaccine|vaccination|drug|medication|medicine)\s+(?:causes?|leads to|results in|gives you|is linked to)\s+{diseases}',
            rf'{diseases}\s+(?:spreads?|is spread by|is transmitted by|is contagious|can be transmitted|is airborne|is communicable)',
            rf'(?:scientists?|doctors?|researchers?|study|studies)\s+(?:found|discovered|proved|confirmed|revealed)\s+(?:that\s+)?([\w\s]+)\s+(?:cures?|treats?|causes?|prevents?)\s+{diseases}',
            rf'{diseases}\s+(?:outbreak|epidemic|pandemic)\s+(?:in|at|across)\s+([A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+)*)',
            rf'([\w\s]+)\s+(?:kills|prevents|stops|blocks|fights|destroys|eliminates)\s+(?:cancer cells|tumors?|viruses?|bacteria|pathogens?|infections?)',
            rf'(?:blood pressure|cholesterol|sugar levels?|glucose)\s+(?:can be lowered|is reduced|drops|increases)\s+(?:by|with|using)\s+([\w\s]+)',
            rf'([\w\s]+)\s+(?:boosts?|strengthens?|improves?|enhances?|weakens?|damages?|destroys?)\s+(?:immune system|immunity|health)'
        ]
        
        for pattern in medical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                medical_claim = match.group(0)
                claims.append({
                    'type': 'medical_claim',
                    'claim': medical_claim,
                    'context': match.group(0),
                    'verifiable': True,
                    'critical': True
                })
        
        # Deduplicate claims by normalizing subject names
        seen_claims = {}
        deduplicated_claims = []
        
        for claim in claims:
            if claim['type'] in ['death_claim', 'alive_claim']:
                # Normalize the subject name by removing extra words like WAS, IS, etc.
                subject = claim['subject']
                normalized_subject = ' '.join([word for word in subject.upper().split() if word not in ['WAS', 'IS', 'HAS', 'BEEN', 'GOT', 'BECAME', 'STILL', 'ARE']])
                
                claim_key = f"{claim['type']}_{normalized_subject}"
                
                if claim_key not in seen_claims:
                    # Update the claim with the normalized subject
                    claim['subject'] = normalized_subject.title() if not normalized_subject.isupper() else normalized_subject
                    if claim['type'] == 'death_claim':
                        claim['claim'] = f"{claim['subject']} is dead/deceased"
                    else:
                        claim['claim'] = f"{claim['subject']} is alive/living"
                    seen_claims[claim_key] = True
                    deduplicated_claims.append(claim)
            else:
                # For non-death/alive claims, just add them
                deduplicated_claims.append(claim)
        
        return deduplicated_claims
    
    def search_web_for_claim(self, query: str) -> List[Dict]:
        """
        Search the web for information about a claim
        Using multiple sources: Wikipedia, DuckDuckGo
        """
        # Check cache first
        if query in self.cache:
            cached_time, cached_data = self.cache[query]
            if time.time() - cached_time < 3600:
                return cached_data
        
        results = []
        
        # Method 1: Wikipedia search
        try:
            # Search Wikipedia for the subject
            wiki_search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'utf8': 1,
                'srlimit': 3
            }
            
            response = requests.get(wiki_search_url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'query' in data and 'search' in data['query']:
                    for item in data['query']['search'][:2]:
                        title = item.get('title', '')
                        snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                        
                        results.append({
                            'type': 'wikipedia',
                            'source': 'Wikipedia',
                            'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            'text': f"{title}: {snippet}",
                            'title': title,
                            'snippet': snippet
                        })
        
        except Exception as e:
            print(f"[Wikipedia Search Error]: {str(e)}")
        
        # Method 2: DuckDuckGo search
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract abstract/summary
                if data.get('Abstract'):
                    results.append({
                        'type': 'abstract',
                        'source': data.get('AbstractSource', 'Unknown'),
                        'url': data.get('AbstractURL', ''),
                        'text': data['Abstract']
                    })
                
                # Extract related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'type': 'related',
                            'source': 'DuckDuckGo',
                            'url': topic.get('FirstURL', ''),
                            'text': topic['Text']
                        })
        
        except Exception as e:
            print(f"[DuckDuckGo Error]: {str(e)}")
        
        # Cache results
        self.cache[query] = (time.time(), results)
        return results
    
    def check_fact_checking_sites(self, claim: str) -> List[Dict]:
        """
        Check known fact-checking websites for the claim
        """
        fact_check_results = []
        
        for site_name, base_url in self.fact_check_sources.items():
            try:
                search_query = claim.replace(' ', '+')
                search_url = f"{base_url}/search?q={search_query}"
                
                response = self.session.get(search_url, timeout=3)
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    if any(word in content for word in ['false', 'mostly false', 'pants on fire']):
                        fact_check_results.append({
                            'source': site_name,
                            'verdict': 'FALSE',
                            'url': search_url,
                            'confidence': 0.7
                        })
                    elif any(word in content for word in ['true', 'mostly true', 'correct']):
                        fact_check_results.append({
                            'source': site_name,
                            'verdict': 'TRUE',
                            'url': search_url,
                            'confidence': 0.7
                        })
                    elif 'unproven' in content or 'unverified' in content:
                        fact_check_results.append({
                            'source': site_name,
                            'verdict': 'UNVERIFIED',
                            'url': search_url,
                            'confidence': 0.5
                        })
            
            except Exception as e:
                logger.debug(f"Fact-check site {site_name} query failed: {e}")
                continue
        
        return fact_check_results
    
    def verify_person_status(self, person_name: str) -> Dict:
        """
        Verify if a person is alive or dead
        """
        # Check cache first
        cache_key = f"person_status_{person_name.lower()}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result
        
        result = {
            'person': person_name,
            'status': 'UNKNOWN',
            'confidence': 0.0,
            'sources': [],
            'last_checked': datetime.now().isoformat()
        }
        
        # Search for recent news/information
        search_queries = [
            f"{person_name} death 2024 2025",
            f"{person_name} alive recent news",
            f"{person_name} latest update"
        ]
        
        alive_mentions = 0
        death_mentions = 0
        
        for query in search_queries:
            web_results = self.search_web_for_claim(query)
            
            for item in web_results:
                text = item.get('text', '').lower()
                
                # Check for death indicators
                if any(word in text for word in ['died', 'dead', 'death', 'passed away', 'obituary', 'funeral']):
                    death_mentions += 1
                    result['sources'].append({
                        'type': 'death_mention',
                        'source': item.get('source'),
                        'url': item.get('url'),
                        'excerpt': text[:200]
                    })
                
                # Check for alive indicators
                if any(word in text for word in ['alive', 'living', 'recent', 'currently', 'today', 'spoke', 'appeared']):
                    alive_mentions += 1
                    result['sources'].append({
                        'type': 'alive_mention',
                        'source': item.get('source'),
                        'url': item.get('url'),
                        'excerpt': text[:200]
                    })
                
                elif item.get('type') == 'wikipedia' and 'death' not in text and 'died' not in text:
                    result['sources'].append({
                        'type': 'general_info',
                        'source': item.get('source'),
                        'url': item.get('url'),
                        'excerpt': text[:200]
                    })
        
        total_sources = len(result['sources'])
        
        if death_mentions > 0 and death_mentions > alive_mentions:
            result['status'] = 'DECEASED'
            result['confidence'] = min(0.95, 0.65 + (death_mentions * 0.15))
        
        elif alive_mentions > death_mentions and alive_mentions > 0:
            result['status'] = 'ALIVE'
            result['confidence'] = min(0.90, 0.65 + (alive_mentions * 0.15))
        
        elif total_sources > 0 and death_mentions == 0:
            result['status'] = 'ALIVE'
            result['confidence'] = min(0.80, 0.60 + (min(total_sources, 5) * 0.04))
        
        elif total_sources == 0:
            result['status'] = 'UNKNOWN'
            result['confidence'] = 0.1
        
        else:
            result['status'] = 'UNVERIFIED'
            result['confidence'] = 0.40
        
        self.cache[cache_key] = (time.time(), result)
        
        return result
    
    def verify_claims(self, text: str) -> Dict:
        """
        Main function to verify all claims in text against real-time sources
        """
        # Extract claims
        claims = self.extract_verifiable_claims(text)
        
        verified_claims = []
        false_claims = []
        unverified_claims = []
        
        # Verify each claim
        for claim in claims:
            claim_text = claim['claim']
            
            # Special handling for death claims (CRITICAL)
            if claim['type'] == 'death_claim':
                person_status = self.verify_person_status(claim['subject'])
                
                if person_status['status'] == 'ALIVE' and person_status['confidence'] > 0.6:
                    false_claims.append({
                        'claim': claim_text,
                        'verdict': 'FALSE',
                        'explanation': f"{claim['subject']} is actually alive (confidence: {person_status['confidence']:.0%})",
                        'evidence': person_status['sources'][:3],
                        'severity': 'CRITICAL'
                    })
                elif person_status['status'] == 'DECEASED' and person_status['confidence'] > 0.6:
                    verified_claims.append({
                        'claim': claim_text,
                        'verdict': 'VERIFIED',
                        'confidence': person_status['confidence'],
                        'sources': person_status['sources'][:2]
                    })
                else:
                    unverified_claims.append({
                        'claim': claim_text,
                        'verdict': 'UNVERIFIED',
                        'reason': 'Insufficient evidence to confirm or deny'
                    })
                
                continue
            
            # Special handling for alive claims (CRITICAL) - opposite of death claims
            if claim['type'] == 'alive_claim':
                person_status = self.verify_person_status(claim['subject'])
                
                if person_status['status'] == 'DECEASED' and person_status['confidence'] > 0.6:
                    false_claims.append({
                        'claim': claim_text,
                        'verdict': 'FALSE',
                        'explanation': f"{claim['subject']} is actually deceased (confidence: {person_status['confidence']:.0%})",
                        'evidence': person_status['sources'][:3],
                        'severity': 'CRITICAL'
                    })
                elif person_status['status'] == 'ALIVE' and person_status['confidence'] > 0.6:
                    verified_claims.append({
                        'claim': claim_text,
                        'verdict': 'VERIFIED',
                        'confidence': person_status['confidence'],
                        'sources': person_status['sources'][:2]
                    })
                else:
                    unverified_claims.append({
                        'claim': claim_text,
                        'verdict': 'UNVERIFIED',
                        'reason': 'Insufficient evidence to confirm or deny'
                    })
                
                continue
            
            # Special handling for war/conflict claims (CRITICAL)
            if claim['type'] == 'war_claim':
                search_query = f"{claim['subject']} war invasion attack {time.strftime('%Y')}"
                web_results = self.search_web_for_claim(search_query)
                fact_checks = self.check_fact_checking_sites(claim['context'])
                
                if fact_checks:
                    false_count = sum(1 for fc in fact_checks if fc['verdict'] == 'FALSE')
                    if false_count > 0:
                        false_claims.append({
                            'claim': claim_text,
                            'verdict': 'FALSE',
                            'explanation': f"No credible evidence of war/invasion involving {claim['subject']}",
                            'evidence': fact_checks[:2],
                            'severity': 'CRITICAL'
                        })
                        continue
                
                if len(web_results) < 2:
                    unverified_claims.append({
                        'claim': claim_text,
                        'verdict': 'UNVERIFIED',
                        'reason': f"No credible news sources confirm war/invasion involving {claim['subject']}"
                    })
                    continue
            
            # Special handling for terrorist attack claims (CRITICAL)
            if claim['type'] == 'terrorist_claim':
                search_query = f"terrorist attack {claim['subject']} {time.strftime('%Y %B')}"
                web_results = self.search_web_for_claim(search_query)
                fact_checks = self.check_fact_checking_sites(claim['context'])
                
                if fact_checks:
                    false_count = sum(1 for fc in fact_checks if fc['verdict'] == 'FALSE')
                    if false_count > 0:
                        false_claims.append({
                            'claim': claim_text,
                            'verdict': 'FALSE',
                            'explanation': f"No credible evidence of terrorist attack in {claim['subject']}",
                            'evidence': fact_checks[:2],
                            'severity': 'CRITICAL'
                        })
                        continue
                
                if len(web_results) < 2:
                    unverified_claims.append({
                        'claim': claim_text,
                        'verdict': 'UNVERIFIED',
                        'reason': f"No major news sources confirm terrorist attack in {claim['subject']}"
                    })
                    continue
            
            # Special handling for medical claims (CRITICAL)
            if claim['type'] == 'medical_claim':
                search_query = f"{claim['claim']} medical study research"
                web_results = self.search_web_for_claim(search_query)
                fact_checks = self.check_fact_checking_sites(claim['context'])
                
                if fact_checks:
                    false_count = sum(1 for fc in fact_checks if fc['verdict'] == 'FALSE')
                    true_count = sum(1 for fc in fact_checks if fc['verdict'] == 'TRUE')
                    
                    if false_count > 0:
                        false_claims.append({
                            'claim': claim_text,
                            'verdict': 'FALSE',
                            'explanation': f"Medical claim debunked by {false_count} fact-checking source(s)",
                            'evidence': fact_checks[:2],
                            'severity': 'CRITICAL'
                        })
                        continue
                    elif true_count > 0:
                        verified_claims.append({
                            'claim': claim_text,
                            'verdict': 'VERIFIED',
                            'confidence': 0.75,
                            'sources': fact_checks[:2]
                        })
                        continue
                
                unverified_claims.append({
                    'claim': claim_text,
                    'verdict': 'UNVERIFIED',
                    'reason': 'Medical claim requires verification from scientific sources'
                })
                continue
            
            # For other claims, do web search + fact-check sites
            web_results = self.search_web_for_claim(claim_text)
            fact_checks = self.check_fact_checking_sites(claim_text)
            
            if fact_checks:
                false_count = sum(1 for fc in fact_checks if fc['verdict'] == 'FALSE')
                true_count = sum(1 for fc in fact_checks if fc['verdict'] == 'TRUE')
                
                if false_count > 0:
                    false_claims.append({
                        'claim': claim_text,
                        'verdict': 'FALSE',
                        'explanation': f"Rated false by {false_count} fact-checking source(s)",
                        'evidence': fact_checks[:2],
                        'severity': 'HIGH'
                    })
                elif true_count > 0:
                    verified_claims.append({
                        'claim': claim_text,
                        'verdict': 'VERIFIED',
                        'confidence': 0.8,
                        'sources': fact_checks[:2]
                    })
            elif web_results:
                verified_claims.append({
                    'claim': claim_text,
                    'verdict': 'PARTIALLY VERIFIED',
                    'confidence': 0.5,
                    'sources': web_results[:2]
                })
            else:
                unverified_claims.append({
                    'claim': claim_text,
                    'verdict': 'UNVERIFIED',
                    'reason': 'No corroborating sources found'
                })
        
        # Generate final assessment
        total_claims = len(claims)
        false_critical = len([c for c in false_claims if c.get('severity') == 'CRITICAL'])
        
        if false_critical > 0:
            overall_verdict = 'CONTAINS CRITICAL FALSEHOODS'
            reliability = 10
        elif len(false_claims) > total_claims * 0.3:
            overall_verdict = 'MOSTLY FALSE'
            reliability = 25
        elif len(verified_claims) > total_claims * 0.7:
            overall_verdict = 'MOSTLY VERIFIED'
            reliability = 80
        elif len(unverified_claims) > total_claims * 0.5:
            overall_verdict = 'LARGELY UNVERIFIED'
            reliability = 50
        else:
            overall_verdict = 'MIXED CLAIMS'
            reliability = 60
        
        return {
            'overall_verdict': overall_verdict,
            'reliability_score': reliability,
            'total_claims_found': total_claims,
            'verified_claims': verified_claims,
            'false_claims': false_claims,
            'unverified_claims': unverified_claims,
            'timestamp': datetime.now().isoformat()
        }

# Singleton instance
_fact_checker_instance = None

def get_fact_checker():
    """Get or create the real-time fact checker instance"""
    global _fact_checker_instance
    if _fact_checker_instance is None:
        _fact_checker_instance = RealTimeFactChecker()
    return _fact_checker_instance
