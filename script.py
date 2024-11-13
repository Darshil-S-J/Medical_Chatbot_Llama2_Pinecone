from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import CTransformers
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore as Pine

from pinecone import ServerlessSpec, Pinecone

import os
import timeit
import sys
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "index-proj" # Change this as u like.

if index_name not in [i.name for i in Pinecone.list_indexes(pc)]:
    pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model vector dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        )
    )
    
index = Pinecone.Index(pc,name=index_name)


#***Extract Data From the PDF File***
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

extracted_data=load_pdf_file(data='data/')

#print(data)


#***Split the Data into Text Chunks****
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))

#***Download the Embeddings from Hugging Face***
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))



#Creating Embeddings for Each of The Text Chunks
# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
#If we already have an index we can load it like this
docsearch=Pine.from_documents(text_chunks,
                              index_name=index_name,
                              embedding=embeddings)

# query = "What are Allergies"
# docs=docsearch.similarity_search(query, k=3)
# print("Result", docs)



prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="models\\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
# llm=ChatOpenAI(model_name="gpt-3.5-turbo")

qa=RetrievalQA.from_chain_type(llm=llm,
                               chain_type="stuff",
                               retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
                               return_source_documents=True,
                               chain_type_kwargs=chain_type_kwargs)

while True:
    user_input=input(f"Input Prompt:")
    if user_input=='exit':
        print('Exiting')
        sys.exit()
    if user_input=='':
        continue
    result=qa.invoke({"query": user_input})
    print("Response : ", result["result"])
    print("Source Documents : ", result["source_documents"])


end=timeit.default_timer()
print(f"Time to retrieve response: {end-start}")