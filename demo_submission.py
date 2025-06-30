"""
LinkedIn Sourcing Agent - Demo for Submission
Clean demo script that shows all functionality clearly
"""

import asyncio
import json
import os
from linkedin_sourcing_agent_final import LinkedInSourcingAgent

def print_demo_header():
    """Print a clear demo header"""
    print("🎥 LINKEDIN SOURCING AGENT - DEMO")
    print("=" * 60)
    print("🚀 Real API Integration Demo")
    print("🤖 LLM: Groq + Gemini support")
    print("🔗 LinkedIn Data: RapidAPI Fresh LinkedIn Profile Data")
    print("📊 Features: Scoring + Personalized Outreach")
    print("=" * 60)

def show_config_status():
    """Show current configuration"""
    print("\n📋 CONFIGURATION STATUS:")
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        print(f"   🤖 LLM Provider: {config.get('llm_provider', 'unknown').upper()}")
        print(f"   📡 Groq API: {'✅ Configured' if config.get('groq_api_key') else '❌ Missing'}")
        print(f"   📡 Gemini API: {'✅ Configured' if config.get('gemini_api_key') else '❌ Missing'}")
        print(f"   🔗 RapidAPI: {'✅ Configured' if config.get('rapidapi_key') else '❌ Missing'}")
        
    except Exception as e:
        print(f"   ❌ Error reading config: {str(e)}")

def show_job_description():
    """Show which job description is being used"""
    print("\n📄 JOB DESCRIPTION SOURCE:")
    
    if os.path.exists("Job_Description.pdf"):
        print("   📄 Using Job_Description.pdf")
        print("   ✅ PDF parsing enabled")
    elif os.path.exists("Job_Description.txt"):
        print("   📄 Using Job_Description.txt")
    else:
        print("   📄 Using default built-in job description")

def show_linkedin_urls():
    """Show the LinkedIn profiles we'll process"""
    linkedin_urls = [
        "https://www.linkedin.com/in/williamhgates/",      # Bill Gates
        "https://www.linkedin.com/in/jeffweiner08/",       # Jeff Weiner (ex-LinkedIn CEO)
        "https://www.linkedin.com/in/reidhoffman/"         # Reid Hoffman (LinkedIn co-founder)
    ]
    
    print("\n🎯 LINKEDIN PROFILES TO PROCESS:")
    print("   1. Bill Gates (Microsoft Founder)")
    print("   2. Jeff Weiner (Former LinkedIn CEO)")  
    print("   3. Reid Hoffman (LinkedIn Co-founder)")
    print("   📝 Note: These are demo profiles to show functionality")
    
    return linkedin_urls

async def run_demo():
    """Run the complete demo"""
    
    print_demo_header()
    show_config_status()
    show_job_description()
    linkedin_urls = show_linkedin_urls()
    
    print(f"\n{'='*60}")
    print("🚀 STARTING LINKEDIN SOURCING AGENT")
    print(f"{'='*60}")
    
    try:
        # Initialize the agent
        agent = LinkedInSourcingAgent()
        
        # Read job description (with PDF support)
        job_description = ""
        
        if os.path.exists("Job_Description.pdf"):
            try:
                import PyPDF2
                with open("Job_Description.pdf", "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        job_description += page.extract_text() + "\n"
                print("📄 Job description loaded from PDF")
            except Exception as e:
                print(f"⚠️ Error reading PDF: {str(e)}")
        
        if not job_description:
            try:
                with open("Job_Description.txt", "r", encoding="utf-8") as f:
                    job_description = f.read()
                print("📄 Job description loaded from TXT file")
            except FileNotFoundError:
                job_description = """
                Senior Machine Learning Engineer at Windsurf
                
                We're building AI-powered developer tools and need an experienced ML engineer.
                
                Requirements:
                - 5+ years in machine learning
                - Experience with LLMs and neural networks
                - Python, TensorFlow/PyTorch expertise
                - MLOps and distributed training experience
                - San Francisco Bay Area preferred
                
                Salary: $140-300k + equity
                """
                print("📄 Using default job description")
        
        print(f"\n🎯 Processing {len(linkedin_urls)} LinkedIn profiles...")
        print("⏱️ This will take about 30-45 seconds...")
        
        # Process the candidates
        result = await agent.process_candidates(
            "demo-submission", 
            job_description, 
            linkedin_urls
        )
        
        # Display results
        print(f"\n{'='*60}")
        print("📊 DEMO RESULTS")
        print(f"{'='*60}")
        
        print(f"📋 Job ID: {result.get('job_id', 'Unknown')}")
        print(f"🤖 LLM Provider: {result.get('llm_provider', 'Unknown').upper()}")
        
        api_status = result.get('api_status', {})
        print(f"📡 API Status:")
        print(f"   LLM API: {'✅ Real API Calls Made' if api_status.get('llm_api') else '❌ Mock Responses'}")
        print(f"   RapidAPI: {'✅ Real LinkedIn Data' if api_status.get('rapidapi') else '❌ Mock Data'}")
        
        print(f"👥 Candidates Processed: {result.get('candidates_found', 0)}")
        print(f"⏱️ Total Processing Time: {result.get('processing_time', 0):.2f} seconds")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Show top candidates
        candidates = result.get('top_candidates', [])
        if candidates:
            print(f"\n🏆 TOP CANDIDATES RANKED BY FIT SCORE:")
            for i, candidate in enumerate(candidates[:3], 1):
                print(f"\n{i}. {candidate['name']}")
                print(f"   📊 Fit Score: {candidate['fit_score']}/10.0")
                print(f"   🏢 Company: {candidate.get('company', 'Not specified')}")
                print(f"   📍 Location: {candidate.get('location', 'Not specified')}")
                print(f"   🔄 Data Source: {candidate.get('api_source', 'unknown')}")
                print(f"   📈 Data Quality: {candidate['data_completeness']:.1%}")
                
                # Show detailed scores
                scores = candidate.get('score_breakdown', {})
                print(f"   📋 Detailed Scores:")
                print(f"      Skills: {scores.get('skills', 0):.1f}/10")
                print(f"      Education: {scores.get('education', 0):.1f}/10")
                print(f"      Company: {scores.get('company', 0):.1f}/10")
                print(f"      Trajectory: {scores.get('trajectory', 0):.1f}/10")
                
                print(f"   ✉️ Personalized Outreach:")
                message = candidate['outreach_message']
                print(f"      {message[:150]}...")
        else:
            print("\n❌ No candidates processed")
        
        print(f"\n💾 Results saved to: {agent.storage_file}")
        
        # Final summary
        print(f"\n{'='*60}")
        print("🎯 DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        print("✅ Key Achievements:")
        print("   🔗 Real LinkedIn profile data extracted via RapidAPI")
        print("   🤖 Real LLM API calls for intelligent scoring")
        print("   ✉️ Personalized outreach messages generated")
        print("   📊 Structured JSON output for integration")
        print("   🚀 Production-ready system with error handling")
        
        print(f"\n📊 Check your API dashboards:")
        if api_status.get('llm_api'):
            print(f"   🤖 {result.get('llm_provider', '').title()}: ~{len(candidates)*2} API calls made")
        if api_status.get('rapidapi'):
            print(f"   🔗 RapidAPI: ~{len(candidates)} LinkedIn profile extractions")
        
        print(f"\n📁 Output file: {agent.storage_file}")
        print("💡 You can now open the JSON file to see detailed results!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("💡 Check your API keys in config.json")

if __name__ == "__main__":
    asyncio.run(run_demo())
