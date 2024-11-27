# Opey Agent

## (WIP)

An agentic version of the Opey chatbot for Open Bank Project that uses the [LangGraph](https://www.langchain.com/langgraph) framework

### Installing Locally
### 1. Installing the dependencies
The easiest way to do this is using _poetry_. Install using the [reccomended method](https://python-poetry.org/docs/) rather than trying to manually install.

If you havent got a venv already then set that up, if you don't do this you will have to prepend `poetry run` before running the main python scripts that you run here as poetry will create a new venv for you if it does not detect one.

Then you can run `poetry install` in the venv to install dependencies.
### 2. Creating the vector database
Create the 'data' folder by running 
```
cd src
mkdir data
``` 
Obtain or set up the ChromaDB database within this folder. A script to process OBP swagger documentation for endpoints and glossary and add it to a vector database will be released later.
### 3. Setting up the environemnet 
First you will need to rename the `.env.example` file to `.env` and change several parameters. You have options on which LLM provider you decide to use for the backend agent system. 
#### OpenAI
Obtain an OpenAI API key and set `OPENAI_API_KEY="sk-proj-..."` 

Then set:
```
MODEL_PROVIDER='openai'

OPENAI_SMALL_MODEL="gpt-4o-mini"
OPENAI_MEDIUM_MODEL="gpt-4o"
```

#### Anthropic
Obtain an Anthropic API key and set `ANTHROPIC_API_KEY="sk-ant-..."`

Then set:
```
MODEL_PROVIDER='anthropic'

ANTHROPIC_SMALL_MODEL="claude-3-haiku-20240307"
ANTHROPIC_MEDIUM_MODEL="claude-3-sonnet-20240229"
```
#### Ollama (Run models locally)
This is only reccomended if you can run models on a decent size GPU. Trying to run on CPU will take ages, not run properly or even crash your computer.

[Install](https://ollama.com/download) Ollama on your machine. I.e. for linux:

`curl -fsSL https://ollama.com/install.sh | sh` 

Pull a model that you want (and that supports [tool calling](https://ollama.com/search?&c=tools)) from ollama using `ollama pull <model name>` we reccomend the latest llama model from Meta: `ollama pull llama3.2`

Then set
```
MODEL_PROVIDER='anthropic'

OLLAMA_SMALL_MODEL="llama3.2"
OLLAMA_MEDIUM_MODEL="llama3.2"
```
Note that the small and medium models are set as the same here, but you can pull a different model and set that.

### 4. Open Bank Project (OBP) credentials
In order for the agent to communicate with the Open Bank Project API, we need to set credentials in the env. First sign up and get an API key on your specific instance of OBP i.e. https://apisandbox.openbankproject.com/ (this should match the `OBP_BASE_URL` in the env). Then set:
```
OBP_USERNAME="your-obp-username"
OBP_PASSWORD="your-obp-password"
OBP_CONSUMER_KEY="your-obp-consumer-key"
```

## Running
Run the backend agent with `python src/run_service.py` or `poetry run python src/run_service.py` if you did not create a venv at the start

Run the frontend streamlit app with `streamlit run src/streamlit_app.py` or same as above if you did not create the venv.

The best way to interact with the agent is through the streamlit app, but it also functions as a rest API whose docs can be found at `http://127.0.0.1:8000/docs`

## Langchain Tracing with Langsmith
If you want to have metrics and tracing for the agent from LangSmith. Obtain a [Langchain tracing API key](https://smith.langchain.com/) and set:
```
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="lsv2_pt_..."
LANGCHAIN_PROJECT="langchain-opey" # or whatever name you want
```