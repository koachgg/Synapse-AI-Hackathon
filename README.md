# Enhanced LinkedIn Sourcing Agent

A comprehensive Python tool for automating multi-source candidate sourcing, AI-powered scoring, and personalized outreach generation using Google Search, GitHub, Company Websites, Academic Sites, and flexible LLM integration (Gemini or Grok).

## 🚀 Key Enhancements

### **Multi-Source Candidate Discovery**
- � **LinkedIn Profiles**: Traditional LinkedIn profile search via Google
- 👨‍💻 **GitHub Profiles**: Technical candidates with public repositories
- 🏢 **Company Websites**: Current employees from career pages and team pages
- 🎓 **Academic Sites**: Researchers from Google Scholar, ResearchGate, ArXiv
- 🌐 **General Web**: Comprehensive web search for additional profiles

### **Flexible LLM Integration**
- 🤖 **Gemini Support**: Google's Gemini for detailed analysis
- ⚡ **Grok Support**: xAI's Grok for alternative AI insights
- 🔄 **Easy Switching**: Change LLM providers with a single parameter
- 📊 **Performance Comparison**: Built-in A/B testing between providers

### **Enhanced Scoring System**
- 📈 **Multi-Source Bonus**: Extra points for candidates found across platforms
- 🧠 **Intelligent Analysis**: LLM-powered evaluation of technical skills
- 📊 **Data Completeness**: Scoring based on available information richness
- 🎯 **Context-Aware**: Enhanced prompts with multi-source context

## � **Multi-Source Search Strategy**

| Source | Weight | Purpose | Example Finds |
|--------|--------|---------|---------------|
| LinkedIn | 40% | Primary profiles | Professional experience, headlines |
| GitHub | 20% | Technical validation | Code repositories, contribution activity |
| Company Sites | 20% | Current employment | Team pages, career announcements |
| Academic Sites | 10% | Research credentials | Published papers, academic projects |
| General Web | 10% | Additional context | Blog posts, conference talks |

## 📊 **Enhanced Scoring Criteria**

### **Base Scoring (1-10 scale)**
1. **Education (20%)**: University prestige and CS background
2. **Career Trajectory (20%)**: Career progression and promotions  
3. **Company Relevance (15%)**: Previous company experience quality
4. **Experience Match (25%)**: Skills alignment with job requirements
5. **Location Match (10%)**: Geographic compatibility
6. **Tenure (10%)**: Job stability and tenure patterns

### **Multi-Source Bonuses (up to +0.5 points)**
- **GitHub Profile**: +0.2 (technical credibility)
- **Academic Publications**: +0.15 (research depth)
- **Company Website Presence**: +0.1 (employment validation)
- **High Data Completeness**: +0.05 (comprehensive profile)

## 🛠 **Setup Instructions**

### **1. Install Dependencies**
```powershell
pip install -r requirements.txt
```

### **2. Configure API Keys**

#### **For Gemini (Google AI Studio)**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create new API key
3. Update `config.json`:
```json
{
  "llm_provider": "gemini",
  "gemini_api_key": "your_actual_api_key_here"
}
```

#### **For Grok (xAI)**
1. Visit [xAI Console](https://console.x.ai/)
2. Generate API key
3. Update `config.json`:
```json
{
  "llm_provider": "grok",
  "grok_api_key": "your_actual_api_key_here"
}
```

### **3. Run Setup Script**
```powershell
python setup.py
```

## 🚀 **Usage Options**

### **Quick Test (Recommended First Run)**
```powershell
# Test with Gemini
python test_runner.py quick

# Test with Grok
python test_runner.py quick grok
```

### **Full Multi-Source Pipeline**
```powershell
# Full pipeline with Gemini
python test_runner.py

# Full pipeline with Grok
python test_runner.py grok
```

### **Compare LLM Providers**
```powershell
python test_runner.py compare
```

### **Custom Implementation**
```python
import asyncio
from linkedin_sourcing_agent import EnhancedLinkedInSourcingAgent

async def main():
    # Initialize with preferred LLM
    agent = EnhancedLinkedInSourcingAgent(
        llm_provider="gemini",  # or "grok"
        api_key="your_api_key"
    )
    
    jobs = [{
        "job_id": "your-job-id",
        "job_description": "Your job description..."
    }]
    
    results = await agent.process_multiple_jobs(jobs)
    
    # Results include multi-source statistics
    for result in results:
        print(f"Sources found: {result['multi_source_stats']['sources']}")
        print(f"Multi-source candidates: {result['multi_source_stats']['multi_source_candidates']}")

asyncio.run(main())
```

## 📈 **Enhanced Output Format**

```json
{
  "job_id": "ml-research-windsurf-enhanced",
  "llm_provider": "gemini",
  "candidates_found": 15,
  "processing_time": 45.2,
  "multi_source_stats": {
    "total_candidates": 15,
    "sources": {
      "linkedin": 6,
      "github": 4,
      "company_website": 3,
      "academic": 2
    },
    "multi_source_candidates": 5,
    "avg_data_completeness": 0.73
  },
  "top_candidates": [
    {
      "rank": 1,
      "name": "Jane Smith",
      "fit_score": 9.2,
      "base_score": 8.7,
      "multi_source_bonus": 0.5,
      "linkedin_url": "linkedin.com/in/janesmith",
      "github_url": "github.com/janesmith",
      "academic_url": "scholar.google.com/citations?user=...",
      "source": "linkedin",
      "multi_source": true,
      "data_completeness": 0.85,
      "score_breakdown": {
        "education": 9.0,
        "trajectory": 8.5,
        "company": 8.8,
        "skills": 9.3,
        "location": 10.0,
        "tenure": 8.0
      },
      "outreach_message": "Hi Jane, I noticed your impressive ML research publications and active GitHub contributions..."
    }
  ]
}
```

## 🎯 **Why Multi-Source is Better**

### **Higher Quality Candidates**
- **Technical Validation**: GitHub profiles verify coding skills
- **Research Depth**: Academic profiles show intellectual capability  
- **Employment Verification**: Company sites confirm current roles
- **Comprehensive View**: Multiple data points reduce false positives

### **Better Outreach**
- **Personalized Messages**: Reference specific GitHub projects or research
- **Higher Response Rates**: More context leads to relevant outreach
- **Technical Credibility**: Mention concrete technical achievements
- **Multi-Platform Approach**: Connect via preferred platform

### **Improved Scoring Accuracy**
- **Cross-Validation**: Multiple sources confirm candidate quality
- **Bonus System**: Rewards candidates with verified technical presence
- **Context Enhancement**: LLM analysis with richer data inputs
- **Reduced Bias**: Less reliance on LinkedIn headline optimization

## ⚡ **Performance Comparison: Gemini vs Grok**

| Aspect | Gemini | Grok | Recommendation |
|--------|--------|------|----------------|
| **Speed** | Fast (~1-2s per candidate) | Very Fast (~0.5-1s) | Grok for speed |
| **Accuracy** | High technical analysis | High reasoning | Both excellent |
| **Cost** | Free tier available | Pay-per-use | Gemini for budget |
| **Availability** | Global | Limited regions | Check availability |
| **Context Window** | Large | Very Large | Grok for complex jobs |

## 🔧 **Configuration Options**

### **Source Weights** (Customize in `config.json`)
```json
{
  "search_sources": {
    "linkedin": {"enabled": true, "weight": 0.4},
    "github": {"enabled": true, "weight": 0.2},
    "company_websites": {"enabled": true, "weight": 0.2},
    "academic_sites": {"enabled": true, "weight": 0.1},
    "general_web": {"enabled": true, "weight": 0.1}
  }
}
```

### **LLM Provider Switching**
```json
{
  "llm_provider": "gemini",  // or "grok"
  "gemini_api_key": "your_gemini_key",
  "grok_api_key": "your_grok_key"
}
```

## 🚨 **Important Considerations**

### **API Usage & Costs**
- **Gemini**: Free tier available, paid plans for scale
- **Grok**: Pay-per-use model, check current pricing
- **Rate Limits**: Built-in delays respect API boundaries
- **Monitoring**: Track usage in your provider dashboard

### **Data Sources & Ethics**
- **Public Information**: Only searches publicly available data
- **No Direct Scraping**: Uses Google search, not direct platform scraping
- **Rate Limiting**: Respects robots.txt and platform terms
- **Privacy Compliant**: Follows GDPR and privacy best practices

### **GitHub Considerations**
- **Public Profiles Only**: Only finds public GitHub profiles
- **Repository Context**: Analyzes public repository activity
- **Contribution History**: Considers commit frequency and project quality
- **Language Detection**: Identifies primary programming languages

## 🎓 **Academic Integration**

### **Supported Academic Sources**
- **Google Scholar**: Published papers and citations
- **ResearchGate**: Research profiles and publications
- **ArXiv**: Preprint publications in CS/ML
- **University Pages**: Faculty and researcher profiles

### **Research Scoring Bonuses**
- **Publication Count**: More papers = higher education score
- **Citation Impact**: H-index and citation analysis
- **Recent Work**: Recent publications in relevant fields
- **Collaboration**: Co-authorship with known researchers

## 🏢 **Company Website Integration**

### **Supported Company Sites**
- **Google Careers**: Current Google employees
- **Microsoft Careers**: Current Microsoft employees  
- **Meta Careers**: Current Meta employees
- **OpenAI**: Current OpenAI team members
- **NVIDIA**: Current NVIDIA employees
- **Custom Sites**: Configurable for specific companies

### **Employment Verification**
- **Current Roles**: Validates current employment
- **Team Pages**: Finds team member profiles
- **Career Announcements**: Recent hiring announcements
- **Role Descriptions**: Matches against job requirements

## 🎯 **Best Practices**

### **For Best Results**
1. **Use Both LLMs**: Compare Gemini vs Grok for different jobs
2. **Enable All Sources**: Maximum candidate discovery
3. **Customize Weights**: Adjust based on role requirements
4. **Regular Updates**: Keep API keys and dependencies current
5. **Monitor Usage**: Track API costs and rate limits

### **Optimization Tips**
- **Technical Roles**: Increase GitHub weight to 0.3
- **Research Roles**: Increase academic weight to 0.2
- **Senior Roles**: Focus on LinkedIn and company sites
- **Startup Roles**: Enable general web search for broader reach

## 🚀 **Future Enhancements**

- [ ] **Real-time Integration**: Live API connections to LinkedIn/GitHub
- [ ] **ML Model Training**: Custom scoring models based on hiring success
- [ ] **Integration APIs**: Connect with ATS systems (Greenhouse, Lever)
- [ ] **Advanced Analytics**: Candidate pipeline analytics and insights
- [ ] **Video Analysis**: Parse conference talks and technical presentations
- [ ] **Social Media**: Twitter/X integration for thought leadership analysis

## 🤝 **Contributing**

This enhanced multi-source approach represents a significant advancement in automated technical recruiting. The combination of diverse data sources with flexible LLM integration provides unprecedented candidate discovery and analysis capabilities.

Feel free to submit issues, enhancement requests, or contribute to the growing ecosystem of AI-powered recruiting tools!

---

**Note**: This tool is designed for ethical, responsible recruiting practices. Always respect platform terms of service, candidate privacy, and applicable employment laws.
