import json
import requests
import streamlit as st
import nltk
from typing import Type
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

# Download the Punkt tokenizer models used by NLTK for sentence splitting.
nltk.download('punkt')
# Download the list of stopwords from NLTK, used to filter out common words.
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

# Summary Tool
def summary(objective, content):
    # Initialize the ChatOpenAI model with specific parameters
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    
    # Create an instance of RecursiveCharacterTextSplitter, which breaks the content 
    # into smaller chunks suitable for processing, based on the provided separators
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    
    # Split the content into smaller documents using the text_splitter
    docs = text_splitter.create_documents([content])
    
    # Define the prompt template for summarization, where the model is asked
    # to summarize the provided text for a specific objective
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )
    
    # Load the summarization chain with the defined prompt and the ChatOpenAI model
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )
    
    # Run the summary chain to generate the summary of the content based on the objective
    output = summary_chain.run(input_documents=docs, objective=objective)
    return output

# Define a data model for the input of the scrape_website function
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    
    # Field to specify the objective or task for the scraping
    objective: str = Field(
        description="The objective & task that users give to the agent"
    )
    
    # Field to specify the URL of the website to be scraped
    url: str = Field(description="The url of the website to be scraped")

# Define a class to handle website scraping operations
class ScrapeWebsiteTool(BaseTool):
    # Name of the tool
    name = "scrape_website"
    
    # A brief description of the tool
    description = ("useful when you need to get data from a website url, "
                   "passing both url and objective to the function; DO NOT "
                   "make up any url, the url should only be from the search results")
    
    # Define the expected input schema for the tool
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    # Define the main function that runs the tool
    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    # Define an asynchronous function, which is currently not implemented
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

# Define additional arguments for initializing the ChatGPT agent
agent_kwargs = {
    # Include extra prompt messages, specifically a placeholder for 'memory' which stores past interactions
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    
    # Define a system message that may be used to set the behavior or context for the agent
    "system_message": system_message,
}

# Initialize the ChatOpenAI (language model) instance with specific parameters
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# Set up a buffer memory to store a summary of the conversation, which helps in retaining context across interactions
memory = ConversationSummaryBufferMemory(
    memory_key="memory", # Key to identify the memory buffer
    return_messages=True, # Indicate whether to return past messages
    llm=llm, # Associate the language model instance
    max_token_limit=1000 # Set a maximum token limit for the stored messages
)

# Initialize the agent using specified tools, language model, and other configurations
agent = initialize_agent(
    tools, # List of tools or functions available for the agent to use
    llm, # Language model instance
    agent=AgentType.OPENAI_FUNCTIONS, # Type of agent being initialized
    verbose=True, # Set to True to enable detailed logging
    agent_kwargs=agent_kwargs, # Additional agent-specific arguments
    memory=memory, # The buffer memory instance
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

def generate_interview_questions_from_content(role, content, job_description):

    # Define a prompt for the agent to generate questions based on the content
    prompt = f"Based on the following content related to {role} and job description of: {job_description}, what questions are likely to be asked during an interview:\n\n{content}\n\nQuestions:"
    
    # Use the agent to generate questions
    response = agent({"input": prompt})
    output = response["output"]
    # Extract questions from the agent's response
    questions = [q.strip() for q in output.split('\n') if q.strip()]
    
    return questions

def extract_keywords_from_description(description):
    # Tokenize the job description
    words = word_tokenize(description)
    
    # Filter out stopwords and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    keywords = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    return keywords

def is_relevant_question(question, job_description_keywords):
    # List of irrelevant question starters
    irrelevant_starts = ["Can you share", "Were there", "How would you approach answering", "Can you explain how", "Based on"]
    
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

    # Extract keywords from job description
    job_description_keywords = extract_keywords_from_description(job_description)
    
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
            questions = generate_interview_questions_from_content(role, content, job_description_keywords)
            all_questions.extend(questions)
        loop_count += 1

    # Filter out irrelevant questions
    relevant_questions = [q for q in all_questions if is_relevant_question(q, job_description_keywords)]

    if not relevant_questions:
        return ["I couldn't find specific interview questions for this role."]
    return relevant_questions

def clean_question_numbering(question):
    # Splitting the question into words and rejoining them with spaces.
    cleaned = ' '.join(question.split()[1:]) if question.split()[0][-1] == '.' else question
    
    # Splitting the cleaned question by line, converting to a set to remove duplicates, and rejoining.
    cleaned = '\n'.join(set(cleaned.split('\n')))
    
    return cleaned

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