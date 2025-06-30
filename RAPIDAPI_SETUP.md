# RapidAPI LinkedIn Sourcing Agent Setup Guide

## Overview
This enhanced LinkedIn Sourcing Agent now includes **real API integration** with:
- **Gemini/Groq LLMs** for candidate scoring and outreach generation
- **RapidAPI Fresh LinkedIn Profile Data** for extracting detailed LinkedIn profiles
- **DuckDuckGo/Google Search** for finding LinkedIn profile URLs

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure API Keys
Update `config.json` with your API keys:

```json
{
  "llm_provider": "gemini",  // or "groq"
  "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE",
  "groq_api_key": "YOUR_GROQ_API_KEY_HERE",
  "rapidapi_key": "YOUR_RAPIDAPI_KEY_HERE",
  // ... other settings
}
```

### 3. Get Your API Keys

#### Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to `config.json`

#### Groq API Key
1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up/sign in
3. Create a new API key
4. Copy the key to `config.json`

#### RapidAPI Key for LinkedIn Data
1. Visit [RapidAPI](https://rapidapi.com/)
2. Sign up for a free account
3. Search for "Fresh LinkedIn Profile Data" API
4. Subscribe to the API (free tier available)
5. Copy your RapidAPI key to `config.json`

### 4. Run the Agent
```powershell
# Test the setup
python test_rapidapi_agent.py

# Run the full agent
python rapidapi_linkedin_sourcing_agent.py
```

## 🔧 Configuration Options

### LLM Provider Selection
Choose between Gemini and Groq in `config.json`:
```json
{
  "llm_provider": "gemini",  // or "groq"
  // ...
}
```

### Rate Limiting
Adjust API call delays:
```json
{
  "rate_limit_delay": 2.0,  // seconds between API calls
  // ...
}
```

### Search Configuration
Control candidate search behavior:
```json
{
  "max_candidates_per_job": 15,
  "max_search_queries": 3,
  // ...
}
```

## 📊 API Usage & Monitoring

### Expected API Calls Per Job
- **LLM API**: ~10-15 calls (scoring + outreach generation)
- **RapidAPI**: ~5-10 calls (LinkedIn profile enrichment)
- **Search APIs**: ~3-5 calls (finding LinkedIn URLs)

### Monitoring Your Usage
1. **Gemini**: Check [Google AI Studio Console](https://makersuite.google.com/)
2. **Groq**: Check [Groq Console](https://console.groq.com/)
3. **RapidAPI**: Check [RapidAPI Dashboard](https://rapidapi.com/developer/dashboard)

## 🎯 Features

### Real LinkedIn Profile Enrichment
- Extracts complete profile data (experience, education, skills)
- Calculates data completeness scores
- Provides structured candidate information

### Intelligent Candidate Scoring
- Uses LLMs to analyze job fit across 6 dimensions
- Weighted scoring system for overall fit
- Detailed score breakdowns

### Personalized Outreach Generation
- Creates customized LinkedIn messages
- Mentions specific candidate details
- Professional and engaging tone

### Multi-Source Search
- DuckDuckGo search for LinkedIn URLs
- Google search fallback
- Mock candidate generation for testing

## 🔍 Testing & Validation

### Quick Test
```powershell
python test_rapidapi_agent.py
```

This will:
- Check your API configuration
- Test individual components
- Process a sample job posting
- Show detailed results and API status

### Example Output
```
🚀 RapidAPI LinkedIn Sourcing Agent Test
==================================================
📡 Gemini API Key: ✅ Set
📡 Groq API Key: ✅ Set  
📡 RapidAPI Key: ✅ Set
🤖 Selected LLM: GEMINI

📊 TEST RESULTS
============================================================
📋 Job ID: test-ml-engineer-rapidapi
🤖 LLM Provider: GEMINI
📡 API Status:
   LLM API: ✅ Active
   RapidAPI: ✅ Active
👥 Candidates Found: 5
⏱️ Processing Time: 12.34 seconds

🏆 Top 3 Candidates:
1. Alice Johnson
   📊 Fit Score: 8.7/10
   🏢 Company: Google
   📍 Location: San Francisco, CA
   🔄 Data Source: rapidapi
   📈 Data Quality: 85%
   ✉️ Outreach: Hi Alice, I noticed your impressive experience with ML at Google...
```

## 🛠️ Troubleshooting

### Common Issues

#### "No candidates found"
- Check your internet connection
- Verify search engines aren't blocking requests
- Try different search terms in job description

#### "Mock responses being used"
- Verify API keys are correctly set in `config.json`
- Check API key validity on respective platforms
- Ensure you have sufficient API credits

#### "RapidAPI errors"
- Confirm you've subscribed to the LinkedIn API
- Check your RapidAPI quota/limits
- Verify the API endpoint is accessible

### API Rate Limits
- **Gemini**: 60 requests/minute (free tier)
- **Groq**: Varies by plan
- **RapidAPI**: Depends on subscription

Adjust `rate_limit_delay` in config if you hit limits.

## 📁 File Structure
```
├── rapidapi_linkedin_sourcing_agent.py  # Main agent with RapidAPI
├── test_rapidapi_agent.py              # Test runner
├── config.json                         # API keys and settings
├── requirements.txt                    # Python dependencies
├── Job_Description.txt                 # Sample job posting
└── candidates_data.json                # Output results
```

## 🔐 Security Notes
- Never commit API keys to version control
- Use environment variables for production
- Regularly rotate your API keys
- Monitor API usage for unexpected charges

## 💰 Cost Estimation
- **Gemini**: Free tier (60 RPM), paid tiers available
- **Groq**: Free tier available, pay-per-use
- **RapidAPI LinkedIn**: ~$0.01-0.05 per profile (varies by plan)

For 10 candidates per job: ~$0.10-0.50 per job in API costs.

## 🚀 Next Steps
1. Test with your API keys
2. Run on real job postings
3. Monitor API usage and costs
4. Customize scoring weights if needed
5. Add additional search sources
6. Integrate with your recruitment workflow

## 📞 Support
If you encounter issues:
1. Check the troubleshooting section
2. Verify API key setup
3. Test individual components
4. Check API provider documentation
