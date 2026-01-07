import streamlit as st
import os
import subprocess
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="YouTube Transcript QA", layout="centered")
st.title("🎥 YouTube Transcript QA with LLM")
st.write("Ask questions based on a YouTube video's transcript.")

# ----------------------------
# OpenAI API Key Input
# ----------------------------
openai_api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-xxxxxxxxxxxxxxxxxxxx"
)

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# ----------------------------
# YouTube URL Input
# ----------------------------
video_url = st.text_input("YouTube Video URL")

if not video_url:
    st.stop()

# ----------------------------
# Download Captions
# ----------------------------
st.info("Downloading auto-generated captions...")

try:
    subprocess.run(
        [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "-o", "captions",
            video_url
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
except subprocess.CalledProcessError:
    st.error("❌ Failed to download captions. Make sure yt-dlp is installed.")
    st.stop()

# ----------------------------
# Parse VTT File
# ----------------------------
def parse_vtt(file_path):
    text = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (
                line
                and "-->" not in line
                and not line.isdigit()
                and not line.startswith("WEBVTT")
            ):
                text.append(line)
    return " ".join(text)


transcript_file = "captions.en.vtt"

try:
    transcript = parse_vtt(transcript_file)
    st.success("✅ Transcript loaded successfully!")
except FileNotFoundError:
    st.error("❌ Transcript file not found.")
    st.stop()

# ----------------------------
# Split Transcript
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = splitter.create_documents([transcript])

# ----------------------------
# Create Embeddings + Vector Store
# ----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.from_documents(documents, embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# ----------------------------
# Prompt
# ----------------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer question with the help for content you provide .
."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# ----------------------------
# Question Input
# ----------------------------
question = st.text_input("Ask a question about the video")

if not question:
    st.stop()

# ----------------------------
# RAG Chain
# ----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
)

chain = parallel_chain | prompt | llm | StrOutputParser()

# ----------------------------
# Run Chain
# ----------------------------
with st.spinner("Generating answer..."):
    answer = chain.invoke(question)

# ----------------------------
# Output
# ----------------------------
st.subheader("📌 Answer")
st.write(answer)
