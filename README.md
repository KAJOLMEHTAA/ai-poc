# AI Proposal Generator

A Flask-based web application that generates customized business proposals using GPT-4 and document retrieval.

## Features

- Generate detailed business proposals based on client name, industry, and requirements
- Uses GPT-4 for high-quality proposal generation
- Retrieves relevant information from existing proposals and case studies
- Dual interface:
  - REST API endpoint for programmatic access
  - Gradio web UI for interactive use

## Setup
- Clone this repository
- Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
- Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
- Copy `.env.example` to `.env` and add your OpenAI API key:
    ```bash
    OPENAI_API_KEY="your-api-key-here"
    ```