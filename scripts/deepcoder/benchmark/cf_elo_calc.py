#!/usr/bin/env python3
# https://github.com/QwenLM/CodeElo/blob/main/calc_rating.py

import os
import requests
import bisect
import json
import re
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm

def get_percentile(rating: float, sorted_ratings: List[float]) -> float:
    """Calculate the percentile of a given rating."""
    idx = bisect.bisect_left(sorted_ratings, float(rating))
    return round(idx / len(sorted_ratings) * 100, 1)

def read_ratings(file_path: str) -> List[float]:
    """Read sorted ratings from a file."""
    with open(file_path, "r") as f:
        ratings_dict = json.load(f)  # dict with rating as key (str) and count as value (int)
    
    sorted_ratings = []
    for rating, count in ratings_dict.items():
        sorted_ratings.extend([float(rating)] * count)

    return sorted(sorted_ratings)

def get_json_with_retry(url, timeout=10, sleep_time=4, max_retries=5):
    """Fetch JSON data from a URL with retries."""
    tries = 0
    while tries < max_retries:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raises HTTPError for bad responses
            return response.json()
        except Exception as e:
            print(f"Request to {url} failed with error: {e}. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            tries += 1

def calc_elo_rating(contest_id: int, problem_status: Dict[str, List[bool]], sorted_ratings: List[float], pass_n=None) -> Optional[Tuple[int, float]]:
    """Calculate the Elo rating for a given contest based on problem status."""
    try:
        # Fetch contest data from Codeforces API
        standings = get_json_with_retry(f"https://codeforces.com/api/contest.standings?contestId={contest_id}&showUnofficial=false")
        
        rating_changes = get_json_with_retry(f"https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}")
        
        # Process and validate data
        handle_set: Set[str] = set()
        try:
            handle_set_standings = set(
                standings["result"]["rows"][i]["party"]["members"][0]["handle"] 
                for i in range(len(standings["result"]["rows"]))
            )
            
            handle_set_ratings = set(
                rating_changes["result"][i]["handle"] 
                for i in range(len(rating_changes["result"]))
            )
            
            handle_set = handle_set_standings.intersection(handle_set_ratings)
            
            standings["result"]["rows"] = [
                row for row in standings["result"]["rows"] 
                if row["party"]["members"][0]["handle"] in handle_set
            ]
            
            rating_changes["result"] = [
                change for change in rating_changes["result"] 
                if change["handle"] in handle_set
            ]
            
            assert len(standings["result"]["rows"]) == len(rating_changes["result"]) and len(standings["result"]["rows"]) > 200
        except Exception:
            return None
        
        # Validate results
        if ("result" not in standings or 
            "result" not in rating_changes or 
            len(standings["result"]["rows"]) != len(rating_changes["result"]) or 
            len(standings["result"]["rows"]) <= 200):
            return None
        
        # Find maximum rating
        max_rating = max(change["oldRating"] for change in rating_changes["result"])
        
        # Calculate score and penalty
        score = 0
        penalty = 0
        
        for problem in standings["result"]["problems"]:
            prob = f"{problem['contestId']}{problem['index']}"
            if prob in problem_status:
                if pass_n is None:
                    pass_n = len(problem_status[prob])
                for ith, status in enumerate(problem_status[prob][:pass_n]):
                    if status == 1.0:
                        if "points" in problem:
                            score += max(0, problem["points"] - 50 * ith)
                        else:
                            score += 1
                            penalty += ith * 10
                        break
        
        # Calculate rank
        n = len(standings["result"]["rows"])
        
        rank = n
        for i in range(n):
            if (standings["result"]["rows"][i]["points"] < score or 
                (standings["result"]["rows"][i]["points"] == score and 
                 standings["result"]["rows"][i]["penalty"] > penalty)):
                rank = i
                break
        
        # Binary search for rating
        l, r = 0, max_rating + 100
        while r - l > 1:
            mid = (l + r) // 2
            new_seed = 1
            for i in range(n):
                new_seed += 1 / (1 + 10 ** ((mid - rating_changes["result"][i]["oldRating"]) / 400))
            if new_seed < rank:
                r = mid
            else:
                l = mid
        
        percentile = get_percentile(l, sorted_ratings)
        return l, percentile
    
    except Exception as e:
        print(f"Error fetching data for contest ID {contest_id}: {e}")
        return None

def format_grouped_contest_data(submissions: List[List[bool]], problem_ids: List[str]) -> List[Tuple[int, Dict[str, List[bool]]]]:
    """
    Groups problems by contest ID (including problem letters like A1) into a list of tuples.
    """
    if len(submissions) != len(problem_ids):
        raise ValueError("Length of submissions and problem_ids must be the same.")
    
    grouped_data = defaultdict(dict)
    
    for problem_id, submission in zip(problem_ids, submissions):
        # Extract contest ID using regex to capture leading digits
        match = re.match(r'(\d+)([A-Z].*)', problem_id)
        if not match:
            raise ValueError(f"Invalid problem ID format: {problem_id}")
        
        contest_id = int(match.group(1))  # Numeric part as contest ID
        problem_letter = match.group(0)   # Full problem ID (contest ID + letter part)
        
        # Group problems under their corresponding contest ID
        grouped_data[contest_id][problem_letter] = submission
    
    # Convert to the required list of tuples format
    combined_data = [(contest_id, problems) for contest_id, problems in grouped_data.items()]
    
    return combined_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate Codeforces percentile based on problem submissions')
    parser.add_argument('--results_path', required=True, help='Path to the results JSON file')
    parser.add_argument('--pass_n', type=int, default=1, help='Number of passes to consider for each problem')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, "./codeforces/metadata_cf.json")
    results_path = os.path.abspath(args.results_path)
    ratings_path = os.path.join(current_dir, "./codeforces/ratings_2024.json")
    
    # Load required files
    try:
        # Load sorted ratings
        sorted_ratings = read_ratings(ratings_path)

        # Load results
        with open(results_path, 'r') as file:
            results = json.load(file)
        
        # Load metadata
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return
    
    # Process the data
    try:
        # Format the data
        model_results = format_grouped_contest_data(results, metadata)
        
        # Calculate Elo ratings for each contest with progress bar
        contest_elos = []
        for contest_id, problems in tqdm(model_results, desc="Processing contests"):
            elo_result = calc_elo_rating(contest_id, problems, sorted_ratings, args.pass_n)
            if elo_result is not None:
                contest_elos.append((contest_id, elo_result))
        
        # Calculate average percentile
        percentiles = [elo[1][1] for elo in contest_elos if elo[1] is not None]

        # Calculate estimated rating
        ratings = [elo[1][0] for elo in contest_elos if elo[1] is not None]
        
        if not percentiles:
            print("No valid percentiles calculated.")
            return
        
        estimated_rating = sum(ratings) / len(ratings)
        est_percentile = get_percentile(estimated_rating, sorted_ratings)
        
        # Display results
        print("\n" + "="*50)
        print("CODEFORCES PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Estimated Percentile: {est_percentile:.1f}%")
        print(f"Estimated Codeforces Rating: {estimated_rating}")
        print(f"Contests Processed: {len(contest_elos)}")
        print("="*50)
        
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()