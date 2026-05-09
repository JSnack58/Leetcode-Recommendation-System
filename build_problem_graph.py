import requests
import json
import networkx as nx
import time
import os

def fetch_all_problems():
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
          similarQuestions
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

def build_graph(questions):
    G = nx.Graph()
    
    # First pass: Create nodes and build slug-to-id mapping
    slug_to_id = {}
    for q in questions:
        q_id = q['frontendQuestionId']
        title = q['title']
        slug = q['titleSlug']
        
        slug_to_id[slug] = q_id
        G.add_node(q_id, title=title, titleSlug=slug)
        
    # Second pass: Add edges for similar questions
    for q in questions:
        q_id = q['frontendQuestionId']
        similar_qs = json.loads(q['similarQuestions'])
        
        for sim_q in similar_qs:
            sim_slug = sim_q['titleSlug']
            
            # Look up the ID of the similar question
            if sim_slug in slug_to_id:
                sim_id = slug_to_id[sim_slug]
                G.add_edge(q_id, sim_id)
            else:
                # Some similar questions might be hidden or locked, so they weren't in the main list
                # We can add them as nodes using the title slug as ID temporarily
                G.add_node(sim_slug, title=sim_q.get('title', sim_slug), titleSlug=sim_slug)
                G.add_edge(q_id, sim_slug)
                
    return G

if __name__ == "__main__":
    print("Fetching problems from LeetCode...")
    questions = fetch_all_problems()
    
    print("\nBuilding undirected graph...")
    G = build_graph(questions)
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    os.makedirs("data", exist_ok=True)
    
    # Save as JSON
    json_output = "data/leetcode_problems_graph.json"
    print(f"\nSaving graph to {json_output}...")
    from networkx.readwrite import json_graph
    json_data = json_graph.node_link_data(G)
    with open(json_output, "w") as f:
        json.dump(json_data, f)
        
    # Save as GML (Graph Modeling Language - good for Gephi, Cytoscape)
    gml_output = "data/leetcode_problems_graph.gml"
    print(f"Saving graph to {gml_output}...")
    nx.write_gml(G, gml_output)
    
    print("Done!")
