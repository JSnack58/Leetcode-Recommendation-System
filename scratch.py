import requests

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
      titleSlug
      similarQuestions
    }
  }
}
"""
variables = {
    "categorySlug": "",
    "skip": 0,
    "limit": 4000,
    "filters": {}
}

response = requests.post(url, json={'query': query, 'variables': variables})
res_json = response.json()
if 'data' in res_json:
    questions = res_json['data']['problemsetQuestionList']['questions']
    print(f"Successfully fetched {len(questions)} questions.")
else:
    print(res_json)
