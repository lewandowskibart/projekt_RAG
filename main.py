import customtkinter as ctk
import threading
import queue as Queue
import re
from tkinter import Tk, font

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,

)

from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,

)
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone, ServerlessSpec

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import getpass
import os

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("1280x720")

class App:
    def __init__(self, master, queue):
        self.queue = queue
        self.master = master
        self.master.title("RAG IRNIP APP")

        self.texts = []
        self.max_texts = 6
        
        self.text_font = font.Font(family="Calibri", size=12)
        
        self.frame = ctk.CTkFrame(self.master)
        self.frame.pack(padx=20, pady=20)
       
        self.label = ctk.CTkLabel(self.frame, text="Enter prompt:")
        self.label.pack()

        self.input = ctk.CTkEntry(self.frame,width = 800)
        self.input.pack()

        self.button = ctk.CTkButton(self.frame, text="Send message", command=self.send_text)
        self.button.pack()
        

    def update_texts(self,text):
        width = 800
        text_width = self.text_font.measure(text)
        text_height = text_width//(width/2) + 2
        print(f"Wysokosc ramki: {text_height} ")
        new_text = ctk.CTkTextbox(self.frame,width,height=10*text_height)
        new_text.insert('1.0', text)
        new_text.configure(state='disabled')
               
        new_text.pack(pady=(0,10), fill='x')

        self.texts.append(new_text)
   
        if len(self.texts) > self.max_texts or sum([text.winfo_height() for text in self.texts]) > 800: 
            old_text = self.texts.pop(0)
            old_text.destroy()
        
    def send_text(self):
        message = self.input.get()
        if message:
            self.queue.put(message)
            self.input.delete(0, 'end')

        else:

            pass
    
print("Input your openAI key: ")
####POTRZEBNY KLUCZ, ZALATWIC
KLUCZ_OPENAI = ""
os.environ['KLUCZ_OPENAI'] = KLUCZ_OPENAI

print("\nPINEAPI_KEY")
####POTRZEBNY KLUCZ DO PINECONEA
KLUCZ_PINECONE = ""
os.environ['KLUCZ_PINECONE'] = KLUCZ_PINECONE

index_name = "RAG_IRNIP"
try:
  cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
  region = os.environ.get('PINECONE_REGION') or 'us-east-1'

  spec = ServerlessSpec(cloud=cloud, region=region)
  pc = Pinecone()
  pc.create_index(
          index_name,
          dimension=3072,
          metric='euclidean',
          spec=spec
      )
except:
  print("The index already exists!!!!!!")

loader = PyPDFLoader(("https://www.warta.pl/documents/UFK/Warunki_ubezpieczen/WN2/WN2_-_OWU_WARTA_NIERUCHOMOSCI.pdf"))

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=10)
docs = text_splitter.split_documents(docs)

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
embeddings = OpenAIEmbeddings()

docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name, namespace=f"ns1")

retriever = docsearch.as_retriever()

#propmpt

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user", "Given the above conversation and the documetn given, answer the following query"),
])


llm = ChatOpenAI(
    KLUCZ_OPENAI="",
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

contextualize_q_chain = prompt | llm | StrOutputParser()

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, say that you don't and explain why. \

{context}"""    
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

chat_history = {}
app = None
       
def main():
    global app
    
    message_queue = Queue.Queue()
    app = App(root, message_queue)
    
    ai_thread = threading.Thread(target=run_querie, args=(message_queue, app)) 
    ai_thread.daemon = True
    ai_thread.start()
     
    root.mainloop()
    
def run_querie(queue, app):
    while True:
        try:
            query = queue.get(block=True)
            print(f'Question: {query}\n')
            queue.queue.clear()
            x = conversational_rag_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": "w33d"}
            },  # constructs a key "abc123" in `store`.
            )['chat_history']
            
            if not x:
                pass
            else:
                answer = x[-1]
                print(f'Response: {answer}\n')
                app.master.after(0, app.update_texts, answer.content)
        except Queue.Empty:
            pass
        
        #print  answer generated by AI
if __name__ == "__main__":
    main()   


            
        
    
    
    

        
        