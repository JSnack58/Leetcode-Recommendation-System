import json
from collections import defaultdict

def run_score_trend_analysis(file_path):
    # Mapping user_slug -> list of (finish_time, score)
    user_records = defaultdict(list)
    
    print(f"Loading data from {file_path}...")
    total_records = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    user_slug = data.get("user_slug")
                    score = data.get("score")
                    finish_time = data.get("finish_time", 0)
                    
                    if user_slug is not None and score is not None:
                        user_records[user_slug].append((finish_time, score))
                    
                    total_records += 1
                    if total_records % 500000 == 0:
                        print(f"Processed {total_records} records...")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Data loaded. Analyzing trends...")
    
    # Trend 1: Does average score correlate with TOTAL number of contests participated?
    # e.g. users who do 1 contest average X, users who do 10 contests average Y
    score_by_total_participation = defaultdict(list)
    
    # Trend 2: How does score change over time for users?
    # e.g. average score on a user's 1st contest, 2nd contest, etc.
    score_by_contest_nth = defaultdict(list)
    
    for slug, records in user_records.items():
        # Sort by finish_time to get chronological order
        records.sort(key=lambda x: x[0])
        
        total_contests = len(records)
        
        # Calculate average score for this user
        avg_score = sum(score for _, score in records) / total_contests
        
        # Group users into buckets of total participation
        if total_contests >= 20:
            score_by_total_participation["20+"].append(avg_score)
        else:
            score_by_total_participation[str(total_contests)].append(avg_score)
            
        # Track score on their Nth contest
        for i, (_, score) in enumerate(records):
            nth = i + 1
            if nth <= 20: # cap at 20th contest for reporting
                score_by_contest_nth[nth].append(score)
                
    print("\n" + "="*50)
    print("TREND 1: Average Score by Total Number of Contests")
    print("Do more active users generally score higher?")
    print("="*50)
    
    keys = [str(i) for i in range(1, 20)] + ["20+"]
    for k in keys:
        if k in score_by_total_participation:
            scores = score_by_total_participation[k]
            avg = sum(scores) / len(scores)
            print(f"Users with {k:>2} total contests (N={len(scores):>7}): Avg Score = {avg:.2f}")

    print("\n" + "="*50)
    print("TREND 2: Average Score on User's N-th Contest")
    print("Does a user's score improve as they do more contests?")
    print("="*50)
    
    for nth in range(1, 21):
        if nth in score_by_contest_nth:
            scores = score_by_contest_nth[nth]
            avg = sum(scores) / len(scores)
            print(f"Score on contest #{nth:>2} (N={len(scores):>7}): Avg Score = {avg:.2f}")

if __name__ == "__main__":
    run_score_trend_analysis("combined_contest_data.jsonl")
