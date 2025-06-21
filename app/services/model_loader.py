from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


system_prompt = """<|SYSTEM|> #
    You are my assistant who gives me details about my courses which i have given in the form of documents of .xlsx format
"""

query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=128,  
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 2048},  # Reduced to fit inside memory
)

Settings.llm = llm
Settings.chunk_size = 1024
llm.generate_kwargs["pad_token_id"] = 0  # Or the ID for <|PAD|> if available
# Use a local embedding model instead of OpenAI
local_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set the embedding model in the Settings
Settings.embed_model = local_embed_model
