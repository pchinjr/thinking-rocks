import json
from typing import Type
import requests
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Secrets
browserless_api_key = st.secrets["BROWSERLESS_API_KEY"]
serper_api_key = st.secrets["SERP_API_KEY"]

nltk.download('punkt')
nltk.download('stopwords')

# Search Tool
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text

# Scraping Tool
def scrape_website(objective, url):
    if not url:
        print("URL is empty, skipping scrape.")
        return None
    print("Scraping website...")
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}&stealth=true"
    response = requests.post(post_url, headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("Found Content")

        if len(text) > 10000:
            print ("Summarizing Content")
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        print("Response text:", response.text)
        return None
    
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""

    objective: str = Field(
        description="The objective & task that users give to the agent"
    )
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions",
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# Extract relevant URLs from search results
def extract_relevant_urls_from_search(search_results, company_name, role):
    relevant_urls = []
    for result in search_results.get('organic', []):
        url = result.get('link', '')
        content = result.get('content', '').lower()
        title = result.get('title', '').lower()
        
        if company_name.lower() in content or role.lower() in content or company_name.lower() in title or role.lower() in title:
            relevant_urls.append(url)
    return relevant_urls

def generate_interview_questions_from_content(objective, content):
    """
    Use the agent to analyze the content and generate potential interview questions.
    """
    # Define a prompt for the agent to generate questions based on the content
    prompt = f"Based on the following content related to {objective}, generate potential interview questions:\n\n{content}\n\nQuestions:"
    
    # Use the agent to generate questions
    response = agent.run(prompt)
    
    # Extract questions from the agent's response
    questions = [q.strip() for q in response.split('\n') if q.strip()]
    
    return questions

def extract_keywords_from_description(description):
    # Tokenize the job description
    words = word_tokenize(description)
    
    # Filter out stopwords and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    keywords = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    return keywords

def generate_questions_from_job_description(job_description):
    # Extract key responsibilities and qualifications from the job description
    # and generate interview questions based on them.
    # This is a simplified version and can be enhanced further.
    responsibilities = [line for line in job_description.split('\n') if line]
    questions_from_description = []
    for responsibility in responsibilities:
        questions_from_description.append(f"How have you demonstrated {responsibility.lower()} in your previous roles?")
    return questions_from_description

def filter_irrelevant_questions(questions):
    # We can add more conditions based on feedback to improve filtering.
    return [q for q in questions if len(q.split()) > 6 and not q.startswith("Can you describe")]

def is_relevant_question(question, job_description_keywords):
    # List of irrelevant question starters
    irrelevant_starts = ["Can you share", "Were there", "How would you approach answering", "Can you explain how"]
    
    # Check if the question starts with any of the irrelevant starts
    if any(question.startswith(start) for start in irrelevant_starts):
        return False
    
    # Check if the question contains any keyword from the job description
    if any(keyword in question.lower() for keyword in job_description_keywords):
        return True
    
    return True

def research_interview_questions(company_name, role, job_description):
    
    # Feedback to the user
    print("Searching for interview experiences online...")
    query = f"interview questions for {role} at {company_name}"
    search_results = json.loads(search(query))
    
    # Feedback to the user
    print("Extracting relevant URLs from the search results...")
    relevant_urls = extract_relevant_urls_from_search(search_results, company_name, role)
    
    all_questions = []
    loop_limit = 3
    loop_count = 0
    
    for url in relevant_urls:
        if loop_count >= loop_limit:
            print("Reached the maximum number of scraping iterations. Moving on...")
            break
        # Feedback to the user
        print(f"Scraping content from {url}...")
        content = scrape_website("Interview Questions Research", url)
        if content:
            # Feedback to the user
            print("Generating interview questions based on the scraped content...")
            questions = generate_interview_questions_from_content(role, content)
            all_questions.extend(questions)
        loop_count += 1

    # Extract keywords from job description
    job_description_keywords = ["coding", "design", "algorithm", "system", "database", "programming", "debugging", "machine learning"]
    
    # Filter out irrelevant questions
    relevant_questions = [q for q in all_questions if is_relevant_question(q, job_description_keywords)]

    if not relevant_questions:
        return ["I couldn't find specific interview questions for this role."]
    return relevant_questions

def clean_question_numbering(question):
    # Splitting the question into words and rejoining them with spaces.
    return ' '.join(question.split()[1:]) if question.split()[0][-1] == '.' else question

def is_experience_question(question):
    # Keywords that typically indicate past experience questions
    experience_keywords = ["did you", "have you", "were you", "can you describe"]
    return any(keyword in question.lower() for keyword in experience_keywords)

# Streamlit web app
def main():
    st.set_page_config(page_title="Interview Prep Bot", page_icon=":briefcase:")
    st.header("Interview Prep Bot :briefcase:")

    company_name = st.text_input("Company Name", "Capital One")
    role = st.text_input("Role", "Lead Software Engineer")
    job_description = st.text_area("Paste the Job Description Here", "Enter the detailed job description provided by the company.")

    if st.button("Research Interview Questions"):
        st.info(f"Researching interview experiences for the role of {role} at {company_name}...")
        interview_questions = research_interview_questions(company_name, role, job_description)
        
        # Clean the questions
        cleaned_questions = [clean_question_numbering(question) for question in interview_questions]
        
        # Filter out experience-related questions
        filtered_questions = [q for q in cleaned_questions if not is_experience_question(q)]

        # Display relevant questions
        st.header("Potential Interview Questions:")
        for idx, question in enumerate(filtered_questions, 1):
            st.write(f"{idx}. {question}")

if __name__ == '__main__':
    main()
