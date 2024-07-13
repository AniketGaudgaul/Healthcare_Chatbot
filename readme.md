# Website QA Application

A short description about the application.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10+
- pip (Python package installer)

## Installation

### Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/your-username/website_qa.git
cd website_qa
```

Install Required Libraries
Install all dependencies listed in the requirements.txt file using pip:
```bash
pip install -r requirements.txt
```


Set Up Environment Variables
You need to set up the OpenAI API key as an environment variable. Replace your_openai_api_key_here with your actual API key.

Open the .env file and add the following line:

```bash
OPENAI_API_KEY="<your_openai_api_key_here>"
```

## Usage
1. User Interface (UI)
To use the UI, run the following command in the project folder:

```bash
streamlit run ui.py
```

This will open the UI in your default web browser.

### Instructions:

Provide a sample URL, e.g., https://www.drugs.com/paracetamol.html, or upload a sample PDF from the sample_pdfs folder.

Click on the "Process URL" or "Process PDF" button to create the embeddings.

Once the embeddings are created, type in the sample questions provided in the sample_questions and click on "Get Answer".

The answer will be displayed in the left tab.

2. Command-Line Interface (CLI)
The CLI currently supports querying web pages. PDF support can be added later.

To use the CLI, run:

```bash
python main.py
```

### Instructions:

Enter the web URL when prompted.

Enter your question and hit enter.

The answer will be displayed in the CLI.
