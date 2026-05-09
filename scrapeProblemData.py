import requests
import json

url = "https://leetcode.com/graphql"

# 1. Added 'similarQuestions' to the GraphQL query
graphql_query = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    title
    difficulty
    stats
    topicTags {
      name
    }
    similarQuestions
  }
}
"""

variables = {
    "titleSlug": "minimum-path-sum"
}

payload = {
    "query": graphql_query,
    "variables": variables
}

headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    data = response.json()
    question_data = data['data']['question']
    
    print(f"Title: {question_data['title']}")
    print(f"ID: {question_data['questionId']}")
    print(f"Difficulty: {question_data['difficulty']}\n")
    
    # Parse Stats
    stats = json.loads(question_data['stats'])
    print(f"Total Accepted: {stats['totalAccepted']}")
    print(f"Acceptance Rate: {stats['acRate']}\n")
    
    # Parse Tags
    tags = [tag['name'] for tag in question_data['topicTags']]
    print(f"Tags: {', '.join(tags)}\n")
    
    # 2. Parse Similar Questions
    print("Similar Questions:")
    similar_qs = json.loads(question_data['similarQuestions'])
    
    # Loop through the array and print the title and difficulty
    for q in similar_qs:
        print(f"- {q['title']} ({q['difficulty']})")
        
else:
    print(f"Failed to fetch data: {response.status_code}")