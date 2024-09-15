import numpy as np
import json
from utils.offline_index import offline_index
from utils.online_retrieval import online_retrieval

def main():
    query = input("What's your research question?: ")
    graph, entity_passage_matrix, unique_named_entities = offline_index("data/sample.json")
    entity_to_score = online_retrieval(graph, unique_named_entities, query)
    probability_distribution = np.array(list(entity_to_score.values()))
    passage_scores = np.matmul(probability_distribution, entity_passage_matrix)
    sorted_indices = np.argsort(passage_scores)[::-1]

    with open("data/sample.json", 'r') as file:
        passages = json.load(file)

    k = 10
    print("To answer that question, you should take a look at the following papers: ")
    print(f"Top {k} Passages in order of decreasing passage score:")
    for i in sorted_indices[:k]:
        print(f"{i + 1}. {passages[i]["title"]}")

if __name__ == "__main__":
    main()
