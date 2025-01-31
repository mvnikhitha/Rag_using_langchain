# README

## LangChain RAG Implementation

This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI, and Chroma for efficient information retrieval and generation.

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required dependencies:
  ```bash
  pip install langchain langchain-openai langchain_chroma langchainhub bs4 getpass
  ```

### Project Setup

1. **Set up API Keys**
   - Export your API keys for LangChain and OpenAI:
   ```python
   import os
   import getpass
   os.environ['LANGCHAIN_TRACING_V2'] = 'true'
   os.environ['LANGCHAIN_API_KEY'] = '<your_langchain_api_key>'
   os.environ['OPENAI_API_KEY'] = getpass.getpass()
   ```

2. **Initialize LangChain Components**
   - Load the necessary modules and models:
   ```python
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
   ```

3. **Load and Process Documents**
   - Fetch and split a blog post for indexing:
   ```python
   from langchain_community.document_loaders import WebBaseLoader
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langchain_chroma import Chroma
   from langchain_openai import OpenAIEmbeddings
   import bs4

   loader = WebBaseLoader(
       web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
       bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
   )
   docs = loader.load()
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   splits = text_splitter.split_documents(docs)
   vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
   ```

4. **Create a RAG Pipeline**
   - Retrieve and generate responses:
   ```python
   from langchain import hub
   from langchain_core.output_parsers import StrOutputParser
   from langchain_core.runnables import RunnablePassthrough

   retriever = vectorstore.as_retriever()
   prompt = hub.pull("rlm/rag-prompt")

   def format_docs(docs):
       return "\n\n".join(doc.page_content for doc in docs)

   rag_chain = (
       {"context": retriever | format_docs, "question": RunnablePassthrough()}
       | prompt
       | llm
       | StrOutputParser()
   )
   ```

5. **Invoke the Model**
   ```python
   response = rag_chain.invoke("What is Task Decomposition?")
   print(response)
   ```

### Output
Example output:
```
Task decomposition is a technique that breaks down complex tasks into smaller and simpler steps. It allows agents to better understand and plan for each component of a task. Methods like Chain of Thought and Tree of Thoughts help in decomposing tasks for easier management and interpretation.
```

### Conclusion
This project showcases a simple yet powerful RAG setup using LangChain and OpenAI for information retrieval and response generation. You can extend it by integrating additional data sources or optimizing retrieval methods for better accuracy.

---
