# imports

import os
from dotenv import load_dotenv
import gradio as gr

# imports for langchain, plotly and Chroma

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from random import randint
import shutil

MODEL = "gpt-4o-mini"
db_name = "vector_db"

# Load environment variables in a file called .env

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

folder = "my-knowledge-base/"
db_name = "vectorstore_db"

def process_files(files):
    os.makedirs(folder, exist_ok=True)

    processed_files = []
    for file in files:
        file_path = os.path.join(folder, os.path.basename(file))  # Get filename
        shutil.copy(file, file_path)
        processed_files.append(os.path.basename(file))

    # Load documents using LangChain's DirectoryLoader
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()

    # Assign filenames as metadata
    for doc in folder_docs:
        filename_md = os.path.basename(doc.metadata["source"])
        filename, _ = os.path.splitext(filename_md)
        doc.metadata["filename"] = filename

    documents = folder_docs 

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Delete previous vectorstore
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

    # Retrieve results
    collection = vectorstore._collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])

    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 35})
    global conversation_chain
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    processed_text = "**Processed Files:**\n\n" + "\n".join(f"- {file}" for file in processed_files)
    return result, processed_text

def random_color():
        return f"rgb({randint(0,255)},{randint(0,255)},{randint(0,255)})"

def show_embeddings_2d(result):
    vectors = np.array(result['embeddings'])  
    documents = result['documents']
    metadatas = result['metadatas']
    filenames = [metadata['filename'] for metadata in metadatas]
    filenames_unique = sorted(set(filenames))

    # color assignment
    color_map = {name: random_color() for name in filenames_unique}
    colors = [color_map[name] for name in filenames]

    tsne = TSNE(n_components=2, random_state=42,perplexity=4)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5,color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(filenames, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='2D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x',yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig

def show_embeddings_3d(result):
    vectors = np.array(result['embeddings'])  
    documents = result['documents']
    metadatas = result['metadatas']
    filenames = [metadata['filename'] for metadata in metadatas]
    filenames_unique = sorted(set(filenames))

    # color assignment
    color_map = {name: random_color() for name in filenames_unique}
    colors = [color_map[name] for name in filenames]

    tsne = TSNE(n_components=3, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(filenames, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

def visualise_data(result):
    fig_2d = show_embeddings_2d(result)
    fig_3d = show_embeddings_3d(result)
    return fig_2d,fig_3d

css = """
.btn {background-color: #1d53d1;}
"""

with gr.Blocks(css=css) as ui:
    gr.Markdown("# Markdown-Based Q&A with Visualization")
    with gr.Row():
        file_input = gr.Files(file_types=[".md"], label="Upload Markdown Files")
        with gr.Column(scale=1):
            processed_output = gr.Markdown("Progress")
    with gr.Row():
        process_btn = gr.Button("Process Files",elem_classes=["btn"])
    with gr.Row():
        question = gr.Textbox(label="Chat ", lines=10)
        answer = gr.Markdown(label= "Response")
    with gr.Row():
        question_btn = gr.Button("Ask a Question",elem_classes=["btn"])
        clear_btn = gr.Button("Clear Output",elem_classes=["btn"])
    with gr.Row():
        plot_2d = gr.Plot(label="2D Visualization")
        plot_3d = gr.Plot(label="3D Visualization")
    with gr.Row():
        visualise_btn = gr.Button("Visualise Data",elem_classes=["btn"])

    result = gr.State([])
    # Action: When button is clicked, process files and update visualization
    clear_btn.click(fn=lambda:("", ""), inputs=[],outputs=[question, answer])
    process_btn.click(process_files, inputs=[file_input], outputs=[result,processed_output])
    question_btn.click(chat, inputs=[question], outputs= [answer])
    visualise_btn.click(visualise_data, inputs=[result], outputs=[plot_2d,plot_3d])

# Launch Gradio app
ui.launch(inbrowser=True)