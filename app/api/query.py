import logging
import sys
import dotenv
import os

dotenv.load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import torch
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === Step 1: Load Documents ===
data_dir = "app/data"
persist_dir = "app/index_storage"

documents = None 
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir, exist_ok=True)
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"✅ Loaded {len(documents)} documents from {data_dir}")
else:
    print(f"📁 Found existing index in '{persist_dir}', skipping document load.")

# === Step 2: Setup LLM ===
system_prompt = """<|SYSTEM|> #
You are my assistant who gives me details about my courses which I have provided in the form of .xlsx documents.
"""

query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.7, "do_sample": False, "pad_token_id": 0},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    # tokenizer_kwargs={"max_length": 2048},
)

Settings.llm = llm
Settings.chunk_size = 1024

# === Step 3: Embedding model ===
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# === Step 4: Build or Load Index ===
if os.path.exists(os.path.join(persist_dir, "docstore.json")):
    print("📦 Loading existing index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
else:
    print("🧠 Building index from documents...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    print("✅ Index built and saved.")

# === Step 5: Define Query Function ===
def query_engine_response(query: str):
    query_engine = index.as_query_engine(streaming=False, similarity_top_k=1)
    response = query_engine.query(query)
    return str(response)

