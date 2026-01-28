# Standard library imports
import argparse
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from projects.finqa.constants import SCRAPED_DATA_DIR, SCRAPED_DATA_URLS_PATH


def load_sec_content(url, user_agent):
    # Use the Chrome webdriver to pull content
    chrome_options = Options()
    chrome_options.add_argument(f"user-agent={user_agent}") # This follows the SEC's required format
    chrome_options.add_argument("--headless=new")  # Run in background
    chrome_options.add_argument("--no-sandbox")  # Required for headless
    chrome_options.add_argument("--disable-dev-shm-usage")  # Prevents crashes

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    try:
        # Wait for the iframe (which loads the viewer) to be present
        iframe = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "ixvFrame"))
        )

        # Forced wait so that the viewer has time to load all the facts
        time.sleep(10)

        # Switch to the iframe so we can access its content
        driver.switch_to.frame(iframe)

        # Optionally, Wait until the XBRL content loads inside the iframe
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "html"))
        )

        # Get the fully rendered HTML
        rendered_html = driver.page_source
        return rendered_html
    finally:
        driver.quit() # Cleanup


def remove_attributes(element):
    new_element = deepcopy(element)

    attributes_to_remove = ["hover-fact", "style",
                            "continued-fact", "enabled-fact",
                            "selected-fact", "hover-fact"]

    for tag in new_element.find_all(True):
        for attr in attributes_to_remove:
            if tag.has_attr(attr):
                del tag[attr]

    return new_element


def process_company(row, output_base_dir, user_agent, index, total):
    """Process a single company (for parallel execution)"""
    url = row["url"]
    output_dir = row["output_directory"]
    
    try:
        rendered_html = load_sec_content(url, user_agent)
        soup = BeautifulSoup(rendered_html, 'html.parser')

        # We are only interested in the table block
        pattern = re.compile(r"TableTextBlock$")
        
        # Try up to 5 times to find tables
        for attempt in range(5):
            all_tags = soup.find_all(attrs={"name": pattern})
            valid_tag_names = [tag["name"] for tag in all_tags if tag.get("name")]
            
            if len(valid_tag_names) > 0:
                break  # Found tables, stop trying
            else : # Error, retrying
                print(f"[{index+1}/{total}] {output_dir}: Found 0 tables, retrying...")
                time.sleep(1) # can be removed 
                rendered_html = load_sec_content(url, user_agent)
                soup = BeautifulSoup(rendered_html, 'html.parser')
        
        print(f"[{index+1}/{total}] {output_dir}: Found {len(valid_tag_names)} tables")

        for tag_name in valid_tag_names:
            element = remove_attributes(soup.find('ix:nonnumeric', attrs={'name': tag_name}))
            filename = tag_name.replace(":", "_") + ".txt"
            final_dir = os.path.join(output_base_dir, output_dir)
            os.makedirs(final_dir, exist_ok=True)
            final_filepath = os.path.join(final_dir, filename)
            Path(final_filepath).write_text(str(element))
        
        print(f"[{index+1}/{total}] {output_dir}: ✓ Saved {len(valid_tag_names)} tables")
        return {"company": output_dir, "tables": len(valid_tag_names), "status": "success"}
    except Exception as e:
        print(f"[{index+1}/{total}] {output_dir}: ✗ Error: {e}")
        return {"company": output_dir, "status": "error", "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GENERATE 10K DATA FROM SEED URLS")
    parser.add_argument("--user_agent", type=str, help="The user agent to use for the browser, this should be an email address.")
    parser.add_argument("--seed_urls_path", type=str, default=SCRAPED_DATA_URLS_PATH, help="The path to the CSV file that contains the seed URLs. The CSV should have two columns: url and output_directory. Default is the seed URLs file in the scraped data directory.")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Output directory. If not provided, the output will be saved in the scraped data directory with the date as the subdirectory.")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel downloads (default: 10)")
    args = vars(parser.parse_args())

    # Create the output directory path
    output_base_dir = None
    if not args["output_base_dir"]:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_base_dir = os.path.join(SCRAPED_DATA_DIR, date_str)
    else:
        output_base_dir = args["output_base_dir"]
    
    os.makedirs(output_base_dir, exist_ok=True)

    # Input is a CSV file
    df = pd.read_csv(args["seed_urls_path"])
    print(f"Processing {len(df)} companies with {args['max_workers']} parallel workers...\n")

    # Process companies in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args['max_workers']) as executor:
        futures = {
            executor.submit(process_company, row, output_base_dir, args["user_agent"], idx, len(df)): idx
            for idx, row in df.iterrows()
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    total_tables = sum(r.get("tables", 0) for r in results if r["status"] == "success")
    print(f"\n✓ Completed: {successful}/{len(results)} companies, {total_tables} tables")
