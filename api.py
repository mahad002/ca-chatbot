from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup, SoupStrainer
from dotenv import load_dotenv
import os
from langsmith import utils  # For LangSmith tracing
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure LangSmith environment variables for tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] ="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "pr-large-cluster-52"

# Initialize LangSmith utilities to confirm tracing is active
utils.tracing_is_enabled()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model with the OpenAI API key
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# Initialize memory for agent checkpointing
memory = MemorySaver()

# Function to retrieve all valid links from a given URL
def get_all_links(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip anchor links and invalid links
                if href.startswith("#") or not href.startswith(("http://", "https://", "www")):
                    continue

                # Add valid links
                if href.startswith("http") or href.startswith("www"):
                    links.add(href)
                elif href.startswith("/"):
                    links.add(requests.compat.urljoin(url, href))
            return list(links)
        else:
            print(f"Error: Received response code {response.status_code} for URL: {url}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to {url}, Exception: {str(e)}")
        return []

# Load links and documents from a specified URL
base_url = "https://certifiedaustralia.com.au"
all_links = get_all_links(base_url)

# Initialize a document loader
loader = WebBaseLoader(
    web_paths=tuple(all_links),
    bs_kwargs=dict(
        parse_only=SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

# Split documents for vector storage
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create an in-memory vector store
vectorstore = InMemoryVectorStore.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Build a retriever tool for retrieving course information
tool = create_retriever_tool(
    retriever,
    "course_info_retriever",
    "Retrieve course, certification, and training details."
)
tools = [tool]

# Initialize the agent with LangSmith tracing and tools
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Define the query endpoint for handling user requests
@app.route('/query', methods=['POST'])
def query_handler():
    # Extract query and user_id from the request JSON
    data = request.get_json()
    query_text = data.get("query", "I want to know about diploma relating to real estate")
    user_id = data.get("user_id", "default_user")
    
    # Set up a unique thread ID for each user
    thread_id = f"thread_{user_id}"

    # Execute the query using the LangChain agent
    response_messages = []
    for event in agent_executor.stream(
        {"messages": [HumanMessage(content=query_text)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        response_messages.append(event["messages"][-1].content)
    
    # Return the concatenated response as JSON
    return jsonify({"response": " ".join(response_messages)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
