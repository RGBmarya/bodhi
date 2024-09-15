import os
import json
import networkx as nx
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from sentence_transformers import SentenceTransformer, util


load_dotenv(".env")
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
retrieval_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # See https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html


# Extract named entities from query with LLM (query named entities)
# Encode named entities with retrieval encoder
# Select nodes from graph that are most similar to query named entities (query nodes)
# Run PPR algorithm over graph (P(query nodes) = 1, P(others) = 0)
def online_retrieval(graph, unique_named_entities, query):
    query_named_entities = extract_query_named_entities(query)
    query_nodes = get_query_nodes(query_named_entities["named_entities"], unique_named_entities)
    return ppr(graph, query_nodes)

def extract_query_named_entities(query):
    print("Extracting named entities from prompt...")
    prompt = get_query_named_entities_prompt(query)
    resp = llm.complete(prompt)
    print("query named entities:", resp.text)
    resp = json.loads(resp.text)
    return resp

def get_query_nodes(query_named_entities, stored_named_entities):
    # TODO: Consider approximate nearest neighbor or retrieve & re-rank for scalability
    query_named_entities_embeddings = retrieval_encoder.encode(query_named_entities)
    stored_named_entities_embeddings = retrieval_encoder.encode(stored_named_entities)
    hits = util.semantic_search(query_named_entities_embeddings, stored_named_entities_embeddings, top_k=1)
    return [stored_named_entities[hit[0]["corpus_id"]] for hit in hits]
    

def ppr(graph, query_nodes):
    personalization = {node:1 for node in query_nodes}        
    return nx.pagerank(graph, personalization=personalization)

def get_query_named_entities_prompt(query):
    return (
        f"""
        Instruction:
        You're a very effective entity extraction system. Please extract all named entities that are important
        for solving the questions below. Place the named entities in JSON format.
        The resluting JSON dict, represented as a string, should be immediately parsable using json.loads().
        Add no extra formatting around this JSON dict.
        
        One-Shot Demonstration:
        Question: Which magazine was started first Arthur's Magazine or First for Women?
        {{"named_entities": ["First for Women", "Arthur's Magazine"]}}
        
        Input:
        Question: {query}
        """
    )

if __name__ == "__main__":
    query_named_entities = ["Stanford University", "Alzheimer's disease"]
    stored_named_entities = ["MIT", "Harvard University", "Stanford", "Dementia", "Alzheimer's"]
    print(get_query_nodes(query_named_entities, stored_named_entities))
