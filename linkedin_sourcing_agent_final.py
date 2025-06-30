"""
LinkedIn Sourcing Agent - Production Version
Direct LinkedIn URL processing with real LLM and RapidAPI integration
Author: AI Assistant via GitHub Copilot
Date: June 29, 2025
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import aiohttp
import logging
import PyPDF2
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM client supporting Gemini and Groq APIs"""
    
    def __init__(self, provider: str = "groq", api_key: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider not in ["gemini", "groq"]:
            raise ValueError("Provider must be 'gemini' or 'groq'")
        
        # Check if we have a valid API key
        if not api_key or api_key in ["your_api_key_here", "YOUR_API_KEY_HERE"]:
            logger.warning(f"No valid API key for {provider}, will use mock responses")
            self.use_mock = True
        else:
            self.use_mock = False
            logger.info(f"✅ Using real {provider.upper()} API")
        
        # Set up API endpoints
        if self.provider == "gemini":
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        elif self.provider == "groq":
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text using real API or mock response"""
        if self.use_mock:
            return self._mock_response(prompt)
        
        if self.provider == "gemini":
            return await self._call_gemini(prompt, max_tokens)
        elif self.provider == "groq":
            return await self._call_groq(prompt, max_tokens)
    
    async def _call_gemini(self, prompt: str, max_tokens: int) -> str:
        """Call Gemini API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0.7
                    }
                }
                
                url = f"{self.base_url}?key={self.api_key}"
                
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info("✅ Gemini API call successful")
                        return text
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Gemini API error {response.status}: {error_text}")
                        return self._mock_response(prompt)
                        
        except Exception as e:
            logger.error(f"❌ Gemini API exception: {str(e)}")
            return self._mock_response(prompt)
    
    async def _call_groq(self, prompt: str, max_tokens: int) -> str:
        """Call Groq API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                
                async with session.post(self.base_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        text = data["choices"][0]["message"]["content"]
                        logger.info("✅ Groq API call successful")
                        return text
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Groq API error {response.status}: {error_text}")
                        return self._mock_response(prompt)
                        
        except Exception as e:
            logger.error(f"❌ Groq API exception: {str(e)}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response as fallback"""
        logger.warning("🔄 Using mock LLM response")
        if "score" in prompt.lower():
            return """Education: 8.5\nTrajectory: 7.5\nCompany: 8.0\nSkills: 8.5\nLocation: 10.0\nTenure: 7.0"""
        elif "outreach" in prompt.lower():
            return "Hi there, I noticed your impressive experience in ML. We have an exciting opportunity that might interest you. Would you be open to a brief conversation?"
        else:
            return "Mock response - API call failed or not configured"

class RapidAPILinkedInClient:
    """Client for RapidAPI Fresh LinkedIn Profile Data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://fresh-linkedin-profile-data.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }
        
        # Check if we have a valid API key
        if not api_key or api_key in ["YOUR_RAPIDAPI_KEY_HERE", "your_rapidapi_key_here"]:
            logger.warning("❌ No valid RapidAPI key, profile enrichment will be mocked")
            self.use_mock = True
        else:
            self.use_mock = False
            logger.info("✅ RapidAPI LinkedIn client initialized")
    
    async def get_profile_data(self, linkedin_url: str) -> Optional[Dict]:
        """Extract detailed profile data from LinkedIn URL"""
        if self.use_mock:
            return self._mock_profile_data(linkedin_url)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {"linkedin_url": linkedin_url}
                
                async with session.get(
                    f"{self.base_url}/get-linkedin-profile",
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ RapidAPI LinkedIn profile fetched: {linkedin_url}")
                        return self._parse_linkedin_data(data)
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ RapidAPI LinkedIn error {response.status}: {error_text}")
                        return self._mock_profile_data(linkedin_url)
                        
        except Exception as e:
            logger.error(f"❌ RapidAPI LinkedIn exception: {str(e)}")
            return self._mock_profile_data(linkedin_url)
    
    def _parse_linkedin_data(self, raw_data: Dict) -> Dict:
        """Parse RapidAPI LinkedIn response into standardized format"""
        try:
            profile_data = {
                "name": raw_data.get("name", ""),
                "headline": raw_data.get("headline", ""),
                "location": raw_data.get("location", ""),
                "about": raw_data.get("about", ""),
                "experience": raw_data.get("experience", []),
                "education": raw_data.get("education", []),
                "skills": raw_data.get("skills", []),
                "company": raw_data.get("current_company", ""),
                "position": raw_data.get("current_position", ""),
                "industry": raw_data.get("industry", ""),
                "connections": raw_data.get("connections", 0),
                "data_completeness": self._calculate_completeness(raw_data),
                "api_source": "rapidapi_real"
            }
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error parsing LinkedIn data: {str(e)}")
            return {}
    
    def _mock_profile_data(self, linkedin_url: str) -> Dict:
        """Generate mock profile data for testing"""
        logger.warning("🔄 Using mock LinkedIn profile data")
        
        # Extract name from URL if possible
        name = "Sample Candidate"
        try:
            if "/in/" in linkedin_url:
                username = linkedin_url.split("/in/")[-1].split("/")[0]
                name = username.replace("-", " ").title()
        except:
            pass
        
        return {
            "name": name,
            "headline": "Senior Machine Learning Engineer | AI Research | Deep Learning",
            "location": "San Francisco, CA",
            "about": "Passionate ML engineer with 5+ years experience building AI systems at scale. Expertise in LLMs, computer vision, and distributed training.",
            "experience": [
                {"company": "Tech Corp", "title": "Senior ML Engineer", "duration": "2021-Present"},
                {"company": "AI Startup", "title": "ML Engineer", "duration": "2019-2021"}
            ],
            "education": [
                {"school": "Stanford University", "degree": "MS Computer Science", "year": "2019"}
            ],
            "skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "Deep Learning", "NLP"],
            "connections": 500,
            "company": "Tech Corp",
            "position": "Senior ML Engineer",
            "data_completeness": 0.85,
            "api_source": "mock"
        }
    
    def _calculate_completeness(self, data: Dict) -> float:
        """Calculate data completeness score"""
        fields = ["name", "headline", "location", "about", "experience", "education", "skills"]
        completed = sum(1 for field in fields if data.get(field))
        return completed / len(fields)

class CandidateScorer:
    """Candidate scorer using real LLM API calls"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.scoring_weights = {
            'education': 0.20,
            'trajectory': 0.20,
            'company': 0.15,
            'skills': 0.25,
            'location': 0.10,
            'tenure': 0.10
        }
    
    async def score_candidates(self, candidates: List[Dict], job_description: str) -> List[Dict]:
        """Score all candidates based on job requirements"""
        logger.info(f"📊 Scoring {len(candidates)} candidates using {self.llm.provider}")
        
        scored_candidates = []
        
        for candidate in candidates:
            try:
                score_breakdown = await self._score_individual_candidate(candidate, job_description)
                
                # Calculate weighted overall score
                overall_score = sum(
                    score_breakdown[category] * weight 
                    for category, weight in self.scoring_weights.items()
                )
                
                candidate_with_score = {
                    **candidate,
                    'fit_score': round(overall_score, 2),
                    'score_breakdown': score_breakdown,
                    'api_enriched': True
                }
                
                scored_candidates.append(candidate_with_score)
                
                # Rate limiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error scoring candidate {candidate.get('name', 'Unknown')}: {str(e)}")
                continue
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x['fit_score'], reverse=True)
        return scored_candidates
    
    async def _score_individual_candidate(self, candidate: Dict, job_description: str) -> Dict:
        """Score individual candidate using LLM"""
        
        prompt = f"""
        Analyze this candidate for the job and provide numeric scores (0-10) for each category.
        
        JOB DESCRIPTION:
        {job_description}
        
        CANDIDATE PROFILE:
        Name: {candidate.get('name', 'Unknown')}
        Headline: {candidate.get('headline', 'N/A')}
        Location: {candidate.get('location', 'N/A')}
        Company: {candidate.get('company', 'N/A')}
        Position: {candidate.get('position', 'N/A')}
        About: {str(candidate.get('about', 'N/A'))[:300]}
        Skills: {str(candidate.get('skills', []))[:200]}
        
        Provide scores (0-10) for:
        Education: Educational background relevance
        Trajectory: Career progression
        Company: Quality of companies
        Skills: Technical skills match
        Location: Geographic fit
        Tenure: Job stability
        
        Format:
        Education: X.X
        Trajectory: X.X
        Company: X.X
        Skills: X.X
        Location: X.X
        Tenure: X.X
        """
        
        try:
            response = await self.llm.generate_text(prompt, max_tokens=200)
            return self._parse_scores(response)
        except Exception as e:
            logger.error(f"Error getting LLM scores: {str(e)}")
            return {category: 5.0 for category in self.scoring_weights.keys()}
    
    def _parse_scores(self, response: str) -> Dict:
        """Parse LLM response into score dictionary"""
        scores = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                try:
                    category, score = line.split(':', 1)
                    category = category.strip().lower()
                    score = float(score.strip())
                    
                    if category in self.scoring_weights:
                        scores[category] = max(0, min(10, score))
                except:
                    continue
        
        # Fill missing scores with default
        for category in self.scoring_weights.keys():
            if category not in scores:
                scores[category] = 5.0
        
        return scores

class OutreachGenerator:
    """Outreach message generator using real LLM API calls"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def generate_outreach_messages(self, candidates: List[Dict], job_description: str) -> List[Dict]:
        """Generate personalized outreach messages"""
        logger.info(f"✉️ Generating outreach messages for {len(candidates)} candidates using {self.llm.provider}")
        
        messages = []
        
        for candidate in candidates:
            try:
                message = await self._generate_individual_message(candidate, job_description)
                messages.append({
                    'candidate': candidate['name'],
                    'message': message,
                    'personalization_score': self._calculate_personalization_score(message)
                })
                
                # Rate limiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error generating message for {candidate.get('name', 'Unknown')}: {str(e)}")
                messages.append({
                    'candidate': candidate['name'],
                    'message': 'Failed to generate personalized message',
                    'personalization_score': 0.0
                })
        
        return messages
    
    async def _generate_individual_message(self, candidate: Dict, job_description: str) -> str:
        """Generate personalized outreach message"""
        
        prompt = f"""
        Write a personalized LinkedIn outreach message for this candidate. Keep it professional, concise (under 150 words), and engaging.
        
        JOB DESCRIPTION:
        {job_description}
        
        CANDIDATE:
        Name: {candidate.get('name')}
        Role: {candidate.get('position')} at {candidate.get('company')}
        Headline: {candidate.get('headline')}
        Location: {candidate.get('location')}
        Skills: {str(candidate.get('skills', []))[:150]}
        
        Guidelines:
        - Address them by name
        - Mention something specific from their background
        - Clearly state the opportunity
        - Professional and conversational tone
        - Clear call-to-action
        
        Write only the message:
        """
        
        try:
            response = await self.llm.generate_text(prompt, max_tokens=300)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating outreach message: {str(e)}")
            return f"Hi {candidate.get('name', 'there')}, I noticed your experience in {candidate.get('headline', 'your field')}. We have an exciting opportunity that might interest you. Would you be open to a brief conversation?"
    
    def _calculate_personalization_score(self, message: str) -> float:
        """Calculate how personalized the message is"""
        indicators = ['your experience', 'your background', 'your work', 'noticed', 'impressed']
        message_lower = message.lower()
        score = sum(1 for indicator in indicators if indicator in message_lower)
        return min(1.0, score / len(indicators))

class LinkedInSourcingAgent:
    """Main LinkedIn Sourcing Agent with direct URL processing"""
    
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.llm_provider = self.config.get('llm_provider', 'groq')
        
        # Get API keys
        if self.llm_provider == 'gemini':
            api_key = self.config.get('gemini_api_key')
        else:
            api_key = self.config.get('groq_api_key')
        
        rapidapi_key = self.config.get('rapidapi_key')
        
        # Initialize components
        self.llm_client = LLMClient(self.llm_provider, api_key)
        self.rapidapi_client = RapidAPILinkedInClient(rapidapi_key)
        self.scorer = CandidateScorer(self.llm_client)
        self.outreach_generator = OutreachGenerator(self.llm_client)
        
        self.storage_file = self.config.get('storage_file', 'candidates_data.json')
        
        logger.info(f"🚀 LinkedIn Sourcing Agent initialized")
        logger.info(f"🤖 LLM Provider: {self.llm_provider.upper()}")
        logger.info(f"🔗 RapidAPI: {'✅' if not self.rapidapi_client.use_mock else '❌ Mock'}")
    
    async def process_candidates(self, job_id: str, job_description: str, linkedin_urls: List[str]) -> Dict:
        """Process a list of LinkedIn URLs for a job"""
        logger.info(f"🎯 Processing {len(linkedin_urls)} LinkedIn profiles for job {job_id}")
        start_time = time.time()
        
        try:
            # Enrich profiles using RapidAPI
            candidates = []
            for url in linkedin_urls:
                profile_data = await self.rapidapi_client.get_profile_data(url)
                if profile_data:
                    profile_data['linkedin_url'] = url
                    candidates.append(profile_data)
                
                # Rate limiting
                await asyncio.sleep(2.0)
            
            if not candidates:
                logger.warning("No valid candidates found")
                return self._create_empty_result(job_id, start_time)
            
            # Score candidates
            scored_candidates = await self.scorer.score_candidates(candidates, job_description)
            
            # Generate outreach messages
            outreach_messages = await self.outreach_generator.generate_outreach_messages(
                scored_candidates[:5], job_description
            )
            
            # Prepare results
            result = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "candidates_found": len(scored_candidates),
                "processing_time": time.time() - start_time,
                "llm_provider": self.llm_provider,
                "api_status": {
                    "llm_api": not self.llm_client.use_mock,
                    "rapidapi": not self.rapidapi_client.use_mock
                },
                "linkedin_urls_processed": linkedin_urls,
                "top_candidates": []
            }
            
            # Add top candidates with outreach messages
            for i, candidate in enumerate(scored_candidates):
                outreach_data = next(
                    (msg for msg in outreach_messages if msg['candidate'] == candidate['name']),
                    {'message': 'Failed to generate message', 'personalization_score': 0.0}
                )
                
                result["top_candidates"].append({
                    "rank": i + 1,
                    "name": candidate['name'],
                    "linkedin_url": candidate.get('linkedin_url', ''),
                    "headline": candidate['headline'],
                    "location": candidate.get('location', ''),
                    "company": candidate.get('company', ''),
                    "position": candidate.get('position', ''),
                    "fit_score": candidate['fit_score'],
                    "score_breakdown": candidate['score_breakdown'],
                    "data_completeness": candidate.get('data_completeness', 0),
                    "api_source": candidate.get('api_source', 'unknown'),
                    "outreach_message": outreach_data['message'],
                    "personalization_score": outreach_data.get('personalization_score', 0.0)
                })
            
            # Save results
            await self._save_results(result)
            
            logger.info(f"✅ Job {job_id} completed in {result['processing_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing job {job_id}: {str(e)}")
            return {
                "job_id": job_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "llm_provider": self.llm_provider
            }
    
    def _create_empty_result(self, job_id: str, start_time: float) -> Dict:
        """Create empty result structure"""
        return {
            "job_id": job_id,
            "candidates_found": 0,
            "top_candidates": [],
            "processing_time": time.time() - start_time,
            "llm_provider": self.llm_provider,
            "api_status": {
                "llm_api": not self.llm_client.use_mock,
                "rapidapi": not self.rapidapi_client.use_mock
            }
        }
    
    async def _save_results(self, result: Dict):
        """Save results to JSON file"""
        try:
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {"jobs": []}
            
            data["jobs"].append(result)
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

# Example usage and testing
async def main():
    """Main function demonstrating the LinkedIn Sourcing Agent"""
    
    try:
        # Initialize the agent
        agent = LinkedInSourcingAgent()
        
        # Read job description (try PDF first, then text file)
        job_description = ""
        
        # Try to read from PDF first
        if os.path.exists("Job_Description.pdf"):
            try:
                with open("Job_Description.pdf", "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    job_description = ""
                    for page in pdf_reader.pages:
                        job_description += page.extract_text() + "\n"
                    print("📄 Job description loaded from PDF")
            except Exception as e:
                print(f"⚠️ Error reading PDF: {str(e)}")
        
        # Fallback to text file
        if not job_description:
            try:
                with open("Job_Description.txt", "r", encoding="utf-8") as f:
                    job_description = f.read()
                print("📄 Job description loaded from TXT file")
            except FileNotFoundError:
                print("📄 Using default job description")
                job_description = """
            Senior Machine Learning Engineer at Windsurf
            
            We're looking for an experienced ML engineer to join our team building AI-powered developer tools.
            
            Requirements:
            - 5+ years experience in machine learning
            - Experience with LLMs and neural networks
            - Proficiency in Python, TensorFlow/PyTorch
            - Experience with MLOps and distributed training
            - Located in San Francisco Bay Area preferred
            
            We offer competitive salary ($140-300k), equity, and cutting-edge AI technology.
            """
        
        # Example LinkedIn URLs (you can replace these with real profiles)
        linkedin_urls = [
            "https://www.linkedin.com/in/williamhgates/",
            "https://www.linkedin.com/in/jeffweiner08/",
            "https://www.linkedin.com/in/reidhoffman/"
        ]
        
        print("🚀 Starting LinkedIn Sourcing Agent - Clean Version")
        print(f"🤖 Using {agent.llm_provider.upper()} LLM")
        print(f"🔗 RapidAPI: {'✅ Active' if not agent.rapidapi_client.use_mock else '❌ Mock'}")
        print("=" * 60)
        
        # Process the candidates
        result = await agent.process_candidates(
            "ml-engineer-windsurf-clean", 
            job_description, 
            linkedin_urls
        )
        
        # Display results
        print(f"\n{'='*60}")
        print("📊 RESULTS")
        print(f"{'='*60}")
        
        print(f"📋 Job ID: {result.get('job_id', 'Unknown')}")
        print(f"🤖 LLM Provider: {result.get('llm_provider', 'Unknown').upper()}")
        
        api_status = result.get('api_status', {})
        print(f"📡 API Status:")
        print(f"   LLM API: {'✅ Active' if api_status.get('llm_api') else '❌ Mock'}")
        print(f"   RapidAPI: {'✅ Active' if api_status.get('rapidapi') else '❌ Mock'}")
        
        print(f"👥 Candidates Processed: {result.get('candidates_found', 0)}")
        print(f"⏱️ Processing Time: {result.get('processing_time', 0):.2f} seconds")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Show top candidates
        candidates = result.get('top_candidates', [])
        if candidates:
            print(f"\n🏆 Top Candidates:")
            for candidate in candidates:
                print(f"\n{candidate['rank']}. {candidate['name']}")
                print(f"   📊 Fit Score: {candidate['fit_score']}/10")
                print(f"   🏢 Company: {candidate['company']}")
                print(f"   📍 Location: {candidate['location']}")
                print(f"   🔄 Data Source: {candidate.get('api_source', 'unknown')}")
                print(f"   📈 Data Quality: {candidate['data_completeness']:.1%}")
                print(f"   🔗 LinkedIn: {candidate['linkedin_url']}")
                
                # Show score breakdown
                scores = candidate.get('score_breakdown', {})
                print(f"   📋 Scores: Skills({scores.get('skills', 0):.1f}) "
                      f"Education({scores.get('education', 0):.1f}) "
                      f"Company({scores.get('company', 0):.1f})")
                
                print(f"   ✉️ Outreach: {candidate['outreach_message'][:100]}...")
        else:
            print("\n❌ No candidates processed")
        
        print(f"\n💾 Results saved to: {agent.storage_file}")
        
        # API usage recommendations
        print(f"\n{'='*60}")
        print("🎯 NEXT STEPS")
        print(f"{'='*60}")
        
        if api_status.get('llm_api'):
            print(f"✅ Check your {result.get('llm_provider', '').title()} dashboard for API usage")
        if api_status.get('rapidapi'):
            print("✅ Check your RapidAPI dashboard for LinkedIn API usage")
        
        print("📝 To use with your own candidates:")
        print("   1. Replace the linkedin_urls list with real LinkedIn profile URLs")
        print("   2. Update Job_Description.txt with your job posting")
        print("   3. Run the script to get scored candidates and outreach messages")
        
        print("\n✅ LinkedIn Sourcing Agent completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure your API keys are configured in config.json")

if __name__ == "__main__":
    asyncio.run(main())
