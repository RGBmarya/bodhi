import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from sentence_transformers import SentenceTransformer
from matplotlib import colormaps


load_dotenv(".env")
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
retrieval_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # See https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def offline_index(filepath, debug=False):
    with open(filepath, 'r') as file:
        data = json.load(file)

    print(f"Extracting named entities from \'{filepath}\'")
    if os.path.exists("out/named_entities.json"):
        with open("out/named_entities.json", "r") as file:
            named_entities = json.load(file)
    else:
        named_entities = extract_named_entities(data)
        with open("out/named_entities.json", "w") as file:
            json.dump(named_entities, file)
    
    print(f"Extracting triples from named entities")
    if os.path.exists("out/named_entities_and_triples.json"):
        with open("out/named_entities_and_triples.json", "r") as file:
            named_entities_and_triples = json.load(file)
    else:
        named_entities_and_triples = extract_triples(data, named_entities)
        with open("out/named_entities_and_triples.json", "w") as file:  # temporary
            json.dump(named_entities_and_triples, file)

    print(f"Extracting synonymy relations")
    unique_entities = set()
    for entry in named_entities:
        for ne in entry["named_entities"]:
            unique_entities.add(ne)
    unique_entities = list(unique_entities)
    if os.path.exists("ou/knowledge_graph.json"):
        with open("out/knowledge_graph.json", "w") as file:
            knowledge_graph = json.load(file)
    else:
        synonymy_relations = extract_synonymy_relations(unique_entities)
        knowledge_graph = {
            "named_entities_and_triples": named_entities_and_triples,
            "synonymy_relations": synonymy_relations
        }
        print(f"Saving knowledge graph")
        with open("out/knowledge_graph.json", "w") as file:
            json.dump(knowledge_graph, file)

    print("Constructing entity-passage matrix")
    entity_passage_matrix = construct_matrix(unique_entities, knowledge_graph)
    entity_passage_df = pd.DataFrame(entity_passage_matrix, index=unique_entities)
    entity_passage_df.to_csv("out/entity_passage_matrix.csv")

    print("Building graph")
    graph, adj_matrix, node_colors = construct_graph(unique_entities, knowledge_graph)
    if debug:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(graph)  # Positions for all nodes

        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color=node_colors)
        nx.draw_networkx_edges(graph, pos, width=[graph[u][v]['weight'] for u, v in graph.edges()])
        nx.draw_networkx_labels(graph, pos, font_size=7, font_family="sans-serif")
        plt.title("Knowledge Graph")
        plt.show()

    return graph, entity_passage_matrix, unique_entities

def extract_named_entities(data):
    return [get_named_entity(e["title"], e["text"], e["idx"]) for e in data]

def extract_triples(data, named_entities):
    res = []
    for i in range(len(named_entities)):
        res.append(
            get_triple(data[i]["title"], data[i]["text"], data[i]["idx"], named_entities[i]["named_entities"])
        )
    return res

def extract_synonymy_relations(named_entities, threshold=0.8):
    synonymy_relations = []
    embeddings = retrieval_encoder.encode(named_entities)
    similarities = retrieval_encoder.similarity(embeddings, embeddings)
    for i, entity1 in enumerate(named_entities):
        for j, entity2 in enumerate(named_entities):
            if i != j and similarities[i][j] > threshold:
                synonymy_relations.append((entity1, entity2, similarities[i][j].item()))
    return synonymy_relations

def get_named_entity(title, text, id):
    print(f"Getting named entities for passage with title \'{title}\'")
    prompt = get_ner_prompt(title, text)
    resp = llm.complete(prompt)
    #  TODO: Validate resp format
    resp = json.loads(resp.text)
    resp["id"] = id
    return resp

def get_triple(title, text, id, named_entites, retry=False):
    try:
        print(f"Getting triples for passage with title \'{title}\'", f"retry={retry}")
        prompt = get_openie_prompt(title, text, named_entites)
        resp = llm.complete(prompt)
        #  TODO: Validate resp format
        resp = json.loads(resp.text)
        resp["id"] = id
        return resp
    except:
        get_triple(title, text, id, named_entites, retry=True)

def get_ner_prompt(title, text):
    return (
        f"""
            Instruction:
            Your task is to extract named entities from the given paragraph.
            Respond with a JSON list of entities.
            The resluting JSON list, represented as a string, should be immediately parsable using json.loads().
            Add no extra formatting around this JSON dict.

            
            One-Shot Demonstration:
            Paragraph:
            ```
            Radio City
            Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English
            and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
            portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
            features.
            ```
            {{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi","English", "May 2008",
            "PlanetRadiocity.com"]}}


            Input:
            Paragraph:
            ```
            {title}
            {text}
            ```
        """
    )

def get_openie_prompt(title, text, named_entities_list):
    return (
        f"""
        Instruction:
        Your task is to construct an RDF (Resource Description Framework) graph from the given passages and
        named entity lists.
        Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.
        Pay attention to the following requirements:
        - Each triple should contain at least one, but preferably two, of the named entities in the list for each
        passage.
        - Clearly resolve pronouns to their specific names to maintain clarity.
        Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
        The resluting JSON dict, represented as a string, should be immediately parsable using json.loads().
        Add no extra formatting around this JSON dict.

        
        One-Shot Demonstration:
        Paragraph:
        ```
        Radio City
        Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English
        and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
        portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
        features.
        ```
        {{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi","English", "May 2008",
        "PlanetRadiocity.com"]}}

        {{"triples":
            [
                ["Radio City", "located in", "India"],
                ["Radio City", "is", "private FM radio station"],
                ["Radio City", "started on", "3 July 2001"],
                ["Radio City", "plays songs in", "Hindi"],
                ["Radio City", "plays songs in", "English"],
                ["Radio City", "forayed into", "New Media"],
                ["Radio City", "launched", "PlanetRadiocity.com"],
                ["PlanetRadiocity.com", "launched in", "May 2008"],
                ["PlanetRadiocity.com", "is", "music portal"],
                ["PlanetRadiocity.com", "offers", "news"],
                ["PlanetRadiocity.com", "offers", "videos"],
                ["PlanetRadiocity.com", "offers", "songs"]
            ]
        }}

        
        Input:
        Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
        Paragraph:
        ```
        {title}
        {text}
        ```
        {{"named_entities": {named_entities_list}}}
        """
    )

def construct_matrix(unique_entities, knowledge_graph):
    entity_to_idx = {entity: idx for idx, entity in enumerate(unique_entities)}
    num_entities = len(unique_entities)
    num_passages = len(knowledge_graph["named_entities_and_triples"])
    matrix = np.zeros((num_entities, num_passages))
    for i, passage in enumerate(knowledge_graph["named_entities_and_triples"]):
        try:
            for named_entity in passage["named_entities"]:
                matrix[entity_to_idx[named_entity], passage["id"]] += 1
        except:
            pass
    return matrix

def construct_graph(unique_entities, knowledge_graph):
    # Extract named entities and synonymy relations
    named_entities_and_triples = knowledge_graph['named_entities_and_triples']
    synonymy_relations = knowledge_graph['synonymy_relations']

    # Step 1: Extract all unique named entities
    passage_colors = {}
    for idx, entity in enumerate(unique_entities):
        passage_colors[entity] = idx
    # for idx, entry in enumerate(named_entities_and_triples):
    #     if entry is not None:
    #         for entity in entry['named_entities']:
    #             passage_colors[entity] = idx  # Using passage index as the color code

    entity_to_idx = {entity: idx for idx, entity in enumerate(unique_entities)}

    # Step 2: Initialize an N x N adjacency matrix where N is the number of unique entities
    N = len(unique_entities)
    adj_matrix = np.zeros((N, N))

    # Step 3: Populate the adjacency matrix based on co-occurrence in triples
    for entry in named_entities_and_triples:
        if entry is not None:
            triples = entry["triples"]
            for triple in triples:
                head, _, tail = triple
                if head in entity_to_idx and tail in entity_to_idx:
                    head_idx = entity_to_idx[head]
                    tail_idx = entity_to_idx[tail]
                    # Increment weight by 1 for both (head, tail) and (tail, head)
                    adj_matrix[head_idx, tail_idx] += 1
                    adj_matrix[tail_idx, head_idx] += 1

    # Step 4: Account for synonymy relations by incrementing the weight between synonym pairs
    for synonymy_pair in synonymy_relations:
        entity_a, entity_b, _ = synonymy_pair
        if entity_a in entity_to_idx and entity_b in entity_to_idx:
            idx_a = entity_to_idx[entity_a]
            idx_b = entity_to_idx[entity_b]
            # Increment weight for both (a, b) and (b, a)
            adj_matrix[idx_a, idx_b] += 1
            adj_matrix[idx_b, idx_a] += 1

    # Step 5: Create the graph using networkx
    G = nx.Graph()

    # Add nodes (entities) to the graph
    for entity in unique_entities:
        G.add_node(entity)

    # Add weighted edges based on the adjacency matrix
    for i in range(N):
        for j in range(i + 1, N):  # Only upper triangular to avoid duplicates
            if adj_matrix[i, j] > 0:
                G.add_edge(unique_entities[i], unique_entities[j], weight=adj_matrix[i, j])

    cmap = colormaps.get_cmap('tab10')
    node_colors = [cmap(passage_colors[entity] % 10) for entity in unique_entities]

    np.save("out/adjacency_matrix.npy", adj_matrix)
    return G, adj_matrix, node_colors

if __name__ == "__main__":
    offline_index("data/sample.json")