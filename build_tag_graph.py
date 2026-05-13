import requests
import json
import networkx as nx
import time
import os
from collections import defaultdict
from itertools import combinations

def fetch_all_problems_with_tags():
    url = "https://leetcode.com/graphql"
    query = """
    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
      problemsetQuestionList: questionList(
        categorySlug: $categorySlug
        limit: $limit
        skip: $skip
        filters: $filters
      ) {
        total: totalNum
        questions: data {
          frontendQuestionId: questionFrontendId
          title
          titleSlug
          topicTags {
            name
            slug
          }
        }
      }
    }
    """
    
    all_questions = []
    skip = 0
    limit = 100
    total = None
    
    while True:
        variables = {
            "categorySlug": "",
            "skip": skip,
            "limit": limit,
            "filters": {}
        }
        
        response = requests.post(url, json={'query': query, 'variables': variables})
        if response.status_code != 200:
            print(f"Failed to fetch data at skip {skip}: {response.status_code}")
            break
            
        data = response.json()
        if 'data' not in data:
            print(f"Error in response at skip {skip}: {data}")
            break
            
        q_list = data['data']['problemsetQuestionList']
        
        if total is None:
            total = q_list['total']
            print(f"Total problems to fetch: {total}")
            
        questions = q_list['questions']
        if not questions:
            break
            
        all_questions.extend(questions)
        print(f"Fetched {len(all_questions)}/{total} problems...")
        
        skip += limit
        time.sleep(0.5) # Be polite to the API
        
        if skip >= total:
            break
            
    return all_questions

def build_bipartite_graph(questions):
    """Builds a graph where problems and tags are both nodes."""
    G = nx.Graph()
    for q in questions:
        q_id = q['frontendQuestionId']
        G.add_node(q_id, title=q['title'], titleSlug=q['titleSlug'], type='problem')
        
        for tag in q['topicTags']:
            tag_id = f"tag_{tag['slug']}"
            if not G.has_node(tag_id):
                G.add_node(tag_id, title=tag['name'], type='tag')
            G.add_edge(q_id, tag_id)
            
    return G

def build_problem_to_problem_graph(questions, min_shared_tags=1):
    """Builds a graph where problems are connected if they share tags."""
    G = nx.Graph()
    
    # Add all nodes
    for q in questions:
        q_id = q['frontendQuestionId']
        G.add_node(q_id, title=q['title'], titleSlug=q['titleSlug'])
        
    # Map tag -> list of problem IDs
    tag_to_probs = defaultdict(list)
    for q in questions:
        q_id = q['frontendQuestionId']
        for tag in q['topicTags']:
            tag_to_probs[tag['slug']].append(q_id)
            
    # Count shared tags between any two problems
    edges_weight = defaultdict(int)
    for tag, probs in tag_to_probs.items():
        # Iterate over every pair of problems that share this tag
        for p1, p2 in combinations(probs, 2):
            # Sort to avoid (p1, p2) and (p2, p1) being treated differently
            if p1 > p2:
                p1, p2 = p2, p1
            edges_weight[(p1, p2)] += 1
            
    # Add edges to the graph
    for (p1, p2), weight in edges_weight.items():
        if weight >= min_shared_tags:
            G.add_edge(p1, p2, weight=weight)
            
    return G

if __name__ == "__main__":
    print("Fetching problems with tags from LeetCode...")
    questions = fetch_all_problems_with_tags()
    
    os.makedirs("data", exist_ok=True)
    
    print("\nBuilding Bipartite Graph (Problems connected to Tags)...")
    G_bipartite = build_bipartite_graph(questions)
    print(f"Bipartite Graph created with {G_bipartite.number_of_nodes()} nodes and {G_bipartite.number_of_edges()} edges.")
    nx.write_gml(G_bipartite, "data/leetcode_tags_bipartite_graph.gml")
    
    print("\nBuilding Problem-to-Problem Graph (Edges = Shared Tags)...")
    # Using min_shared_tags=1 will create a massive graph. 
    # Let's generate it, and if it's too big we can filter it.
    G_prob_to_prob = build_problem_to_problem_graph(questions, min_shared_tags=1)
    print(f"Full Prob-to-Prob Graph created with {G_prob_to_prob.number_of_nodes()} nodes and {G_prob_to_prob.number_of_edges()} edges.")
    
    # Save the full graph (might be large)
    nx.write_gml(G_prob_to_prob, "data/leetcode_problems_shared_tags_graph.gml")
    
    # Also save a filtered version (only connect if they share at least 2 tags)
    print("\nBuilding Filtered Problem-to-Problem Graph (>= 2 Shared Tags)...")
    G_prob_to_prob_filtered = build_problem_to_problem_graph(questions, min_shared_tags=2)
    print(f"Filtered Graph created with {G_prob_to_prob_filtered.number_of_nodes()} nodes and {G_prob_to_prob_filtered.number_of_edges()} edges.")
    nx.write_gml(G_prob_to_prob_filtered, "data/leetcode_problems_shared_tags_filtered_graph.gml")
    
    print("\nDone! All files saved to data/ directory.")
