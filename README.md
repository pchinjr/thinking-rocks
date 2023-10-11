# thinking-rocks

Thinking Rocks

## External APIs

https://platform.openai.com/docs/introduction

https://serper.dev/

https://www.browserless.io/

# Interview Prep Bot

## Overview

Interview Prep Bot is a tool designed to assist job seekers in preparing for interviews. By providing the company name, desired role, and a detailed job description, users can receive potential interview questions tailored to their specific situation.

## Features

- **User-Friendly Interface**: Built with Streamlit, the app offers a simple and intuitive UI for users to input necessary details.
- **Heuristic-Based Question Generation**: Generates relevant interview questions based on the provided job description.
- **Support for Multiple Roles and Companies**: Versatile enough to generate questions for a wide range of roles and companies.

## Getting Started

### Prerequisites

- Python 3.x
- Pipenv

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/thinking-rocks.git
```

2. Navigate to the project directory:
```bash
cd thinking-rocks
```

3. Install the required packages using Pipenv:
```bash
pipenv install
```

### Usage

1. Activate the virtual environment:
```bash
pipenv shell
```

2. Run the application with Streamlit:
```bash
streamlit run app.py
```

3. Navigate to the displayed URL in your web browser to access the Interview Prep Bot.

Certainly! Here's how you can append to the existing README.md to include the process of creating environment secrets in `.streamlit/secrets.toml`:

---

## Environment Secrets with Streamlit

Streamlit offers a way to manage secrets which might be needed by your app, such as API keys. These secrets can be stored in `.streamlit/secrets.toml`, ensuring they're not exposed in the application code.

### Setting up Secrets in Streamlit

1. **Create a Secrets File**: In your project directory, create a folder named `.streamlit` if it doesn't exist. Inside `.streamlit`, create a file named `secrets.toml`.

2. **Add Your Secrets**: Open `secrets.toml` with your favorite text editor and add your secrets in TOML format:

```toml
# .streamlit/secrets.toml

OPENAI_API_KEY = "XXXXXXXXXXX"
SERP_API_KEY = "XXXXXXXXXXX"
BROWSERLESS_API_KEY = "XXXXXXXXXXX"
```

3. **Accessing Secrets in Your App**: In your Streamlit app, you can access these secrets using:

```python
import streamlit as st

api_key = st.secrets["api_key"]
db_username = st.secrets["db_username"]
db_password = st.secrets["db_password"]
```

4. **Ensure File is Git Ignored**: To make sure your secrets file isn't checked into Git, add `.streamlit/secrets.toml` to your `.gitignore` file:

```
# .gitignore

.streamlit/secrets.toml
```

### Caution

Always ensure that your secrets file is not exposed or shared. This method keeps your secrets out of your application code, but you should always ensure the security of any environment in which you're deploying your app.

## How It Works

1. **User Input**: The user provides the company name, desired role, and a detailed job description.
2. **Research and Generation**: The bot searches for relevant content online, extracts information, and uses heuristic-based logic to generate potential interview questions.
3. **Display Results**: The generated questions are displayed to the user, providing insights into what they might be asked during the interview.

## Challenges and Solutions

During the development of Interview Prep Bot, we encountered several challenges:

1. **Relevance of Generated Questions**: The initial version often produced questions focused on the interview experience rather than the job role. This was addressed by implementing heuristic-based logic, leveraging keywords from the provided job description.
  
2. **Text Formatting Issues**: Some questions were concatenated without spaces or had duplicate numbering. We refined our text processing logic to handle and correct these formatting anomalies.

3. **Dependency Management**: To ensure smooth deployment and collaboration, we utilized Pipenv for dependency management and created a `requirements.txt` for easy setup.

## Contributing

If you'd like to contribute, please fork the repository and make changes as you'd like. Pull requests are warmly welcome.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- **OpenAI's ChatGPT**: For guidance and assistance in refining the bot's logic.
- **Streamlit**: For providing a platform to turn the bot into a web application, making it more accessible to users.
- **NLTK library**: For NLP tools and utilities.
- **BeautifulSoup**: For its invaluable web scraping capabilities, enabling the extraction of relevant content.
- **LangChain**: For their advanced language processing capabilities which significantly enhanced the quality and relevance of our application's outputs.

---

# Architectural Decision Record (ADR)

## Title:
Employing Heuristics to Improve Question Relevance in Interview Prep Bot

## Status:
Accepted

## Context:
The initial version of the Interview Prep Bot used a relatively straightforward approach to generate interview questions based on the role and company name provided by the user. However, the generated questions often focused on the interview experience itself rather than being directly relevant to the job role. To improve the relevance of these questions, a more sophisticated approach was needed.

## Decision:
To address this issue, we decided to implement a heuristic-based approach that leveraged the job description provided by the user. This approach involved:

1. **Keyword Extraction from Job Description**:
    - Extracted important keywords and phrases from the user-provided job description using the NLTK library.
    - Utilized these keywords to guide the generation of questions, ensuring that they are more aligned with the job role's specifics.

2. **Filtering Heuristics**:
    - Implemented filters to eliminate questions that focused primarily on the interview experience.
    - Prioritized questions that seemed more directly related to the actual job role and responsibilities, as well as the company culture.

3. **Question Formatting**:
    - Added logic to handle and correct text formatting issues, ensuring that the generated questions are well-phrased and readable.

By employing these heuristics, the aim was to create a list of questions that a potential candidate might realistically be asked during an interview for the provided job role.

## Consequences:
1. **Improved Relevance**: The generated questions are now more directly relevant to the user's input job role and description, providing better utility and value to the user.
2. **Increased Complexity**: The added heuristics and keyword extraction logic increased the complexity of the code. However, the benefits in terms of question quality outweigh the drawbacks.
3. **Dependency on Job Description Quality**: The effectiveness of the heuristic approach is somewhat dependent on the quality and detail of the provided job description. A generic or vague job description might still result in less relevant questions.
4. **Flexibility for Future Enhancements**: The heuristic-based approach provides a foundation upon which further refinements and enhancements can be built, such as incorporating more advanced NLP techniques or feedback mechanisms.

## Date:
October 12, 2023

---

This ADR documents the rationale behind the decision to employ a heuristic-based approach to improve the relevance of generated interview questions in the Interview Prep Bot. Future refinements and adjustments to this approach can be documented in subsequent ADRs.