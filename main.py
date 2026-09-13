import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from colorama import Fore
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
system_template = """You are a customer support specialist.
You assist users with general inquiries and technical issues based on the following context:
{context}"""
human_template = "{question}"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# I used OpenRouter instead of OpenAI - no billing needed you can login and can generate a api key for this
model = ChatOpenAI(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# we need to format this becuse the retriever returns a list of documents and we need to convert it into a string to pass it to the model an llm can't take a list of documents as input so we need to format it into a string and pass it to the model
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def load_documents():
    """Load a file from path and split it into chunks."""
    raw_documents = TextLoader("./docs/user-manual.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    return text_splitter.split_documents(raw_documents)


def load_embeddings(documents):
    #Here i have used sentence-transformers/all-MiniLM-L6-v2 model for embeddings, you can change it to any other model as per your requirement
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents, embeddings)
    return db.as_retriever()

#RunnablePassthrough is used to pass the query to the model without any changes, we can also use other runnables to modify the query before passing it to the model
def generate_response(retriever, query):
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_prompt_template
        | model
        | StrOutputParser()
    )
    return chain.invoke(query)


def start():
    documents = load_documents()
    retriever = load_embeddings(documents)

    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask(retriever)
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask(retriever):
    while True:
        user_input = input("Q: ")
        if user_input == "x":
            start()
        else:
            response = generate_response(retriever, user_input)
            print(Fore.BLUE + f"A: " + response + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()