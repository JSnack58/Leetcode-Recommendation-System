import json
from collections import Counter
import statistics

def run_eda(file_path):
    total_records = 0
    unique_contests = set()
    user_counter = Counter()
    
    country_counter = Counter()
    region_counter = Counter()
    language_counter = Counter()
    
    scores = []
    
    print(f"Starting EDA on {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    total_records += 1
                    
                    contest_id = data.get("contest_id")
                    if contest_id is not None:
                        unique_contests.add(contest_id)
                        
                    user_slug = data.get("user_slug")
                    if user_slug is not None:
                        user_counter[user_slug] += 1
                        
                    country = data.get("country_name") or "Unknown/None"
                    country_counter[country] += 1
                    
                    region = data.get("data_region") or "Unknown"
                    region_counter[region] += 1
                    
                    score = data.get("score")
                    if score is not None:
                        scores.append(score)
                        
                    submissions = data.get("submissions", {})
                    for sub_id, sub_data in submissions.items():
                        lang = sub_data.get("lang", "Unknown")
                        language_counter[lang] += 1
                        
                except json.JSONDecodeError:
                    print(f"Error parsing line {line_num + 1}")
                    continue
                    
                if total_records % 500000 == 0:
                    print(f"Processed {total_records} records...")
                    
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print("\n" + "="*40)
    print("EDA RESULTS")
    print("="*40)
    
    unique_users_count = len(user_counter)
    users_more_than_once = sum(1 for count in user_counter.values() if count > 1)
    total_possible_interactions = unique_users_count * len(unique_contests)
    sparsity = 1.0 - (total_records / total_possible_interactions) if total_possible_interactions > 0 else 0.0

    print(f"Total Records: {total_records:,}")
    print(f"Unique Contests: {len(unique_contests):,}")
    print(f"Unique Users (by slug): {unique_users_count:,}")
    print(f"Users appearing > 1 time: {users_more_than_once:,} ({(users_more_than_once/unique_users_count)*100:.2f}%)")
    print(f"Data Sparsity (1 - records / (users * contests)): {sparsity*100:.2f}%")
    
    # Let's also print the distribution of user participation
    participation_counts = Counter(user_counter.values())
    print("\nUser Participation Distribution (Top 10 frequencies):")
    for times, num_users in participation_counts.most_common(10):
        print(f"  Participated in {times} contests: {num_users:,} users")
    
    if scores:
        print(f"\nScore Statistics:")
        print(f"  Average Score: {sum(scores)/len(scores):.2f}")
        print(f"  Max Score: {max(scores)}")
    
    print("\nTop 10 Countries:")
    for country, count in country_counter.most_common(10):
        print(f"  {country}: {count:,}")
        
    print("\nData Regions:")
    for region, count in region_counter.most_common():
        print(f"  {region}: {count:,}")
        
    print("\nTop 10 Languages Used in Submissions:")
    for lang, count in language_counter.most_common(10):
        print(f"  {lang}: {count:,}")

if __name__ == "__main__":
    run_eda("combined_contest_data.jsonl")
