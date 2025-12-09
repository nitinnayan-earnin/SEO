"""
Process SEO data with OpenAI API integration.

This script:
1. Reads SEO - Sheet2.csv and extracts employer data
2. Parses hourly rates from regular_hourly_rate_parsed_counts field
3. Fetches state occupation data using BLSOESDataExtractor
4. Formats prompts with employer data and state table
5. Calls OpenAI API to predict job titles
6. Saves results to output CSV
"""

import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple
from state_links import BLSOESDataExtractor
from prompt import ROLE_DETERMINATION_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import time
import os
from io import StringIO


def calculate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate OpenAI API cost based on model and token usage.
    
    Args:
        model: Model name (gpt-4o or gpt-4.1)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
        
    Raises:
        ValueError: If model is not supported
    """
    # Pricing per million tokens (as of 2024)
    pricing = {
        'gpt-4o': {'input': 2.50, 'output': 10.00},      # $2.50/$10.00 per million tokens
        'gpt-4.1': {'input': 2.00, 'output': 8.00},      # $2.00/$8.00 per million tokens
    }
    
    # Normalize model name (handle variations like "gpt-4.1", "gpt-4o", etc.)
    model_lower = model.lower()
    if 'gpt-4o' in model_lower or model_lower == 'gpt-4o':
        model_key = 'gpt-4o'
    elif 'gpt-4.1' in model_lower or 'gpt-41' in model_lower:
        model_key = 'gpt-4.1'
    else:
        # If model not in pricing, return 0 (don't calculate cost)
        return 0.0
    
    if model_key not in pricing:
        return 0.0
    
    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * pricing[model_key]['input']
    output_cost = (output_tokens / 1_000_000) * pricing[model_key]['output']
    total_cost = input_cost + output_cost
    
    return total_cost


def parse_hourly_rates(rate_str: str) -> List[str]:
    """
    Parse hourly rates from JSON-like string.
    
    Args:
        rate_str: String like '{"25.72": 1}' or '{"17.25": 1, "23.01": 1}'
                 The CSV has escaped quotes like '{""25.72"": 1}'
        
    Returns:
        List of hourly rate strings, or empty list if empty/invalid
    """
    if pd.isna(rate_str) or rate_str == '' or rate_str == '{}':
        return []
    
    try:
        # The CSV has double quotes escaped as '""', so we need to handle that
        # First, replace '""' with '"' to unescape
        cleaned = rate_str.replace('""', '"')
        # Parse as JSON
        rate_dict = json.loads(cleaned)
        # Extract keys (the hourly rates)
        rates = list(rate_dict.keys())
        return rates
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # Try regex fallback for malformed JSON
        # Extract numbers that look like hourly rates (handles both '""' and '"')
        # Pattern matches: "number" or ""number""
        rates = re.findall(r'"{1,2}([0-9]+\.?[0-9]*)"{1,2}', rate_str)
        if rates:
            return rates
        # If still no match, try simpler pattern
        rates = re.findall(r'([0-9]+\.[0-9]+)', rate_str)
        return rates


def parse_rate_counts(rate_str: str) -> Dict[str, int]:
    """
    Parse hourly rates and their counts from JSON-like string.
    
    Args:
        rate_str: String like '{"25.72": 1}' or '{"17.25": 1, "23.01": 1}'
                 The CSV has escaped quotes like '{""25.72"": 1}'
        
    Returns:
        Dictionary mapping rate strings to counts
    """
    if pd.isna(rate_str) or rate_str == '' or rate_str == '{}':
        return {}
    
    try:
        # The CSV has double quotes escaped as '""', so we need to handle that
        # First, replace '""' with '"' to unescape
        cleaned = rate_str.replace('""', '"')
        # Parse as JSON
        rate_dict = json.loads(cleaned)
        return rate_dict
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Try regex fallback for malformed JSON
        # Extract numbers and their counts
        result = {}
        # Pattern to match: "number": count or ""number"": count
        pattern = r'"{1,2}([0-9]+\.?[0-9]*)"{1,2}:\s*([0-9]+)'
        matches = re.findall(pattern, rate_str)
        for rate, count in matches:
            result[rate] = int(count)
        return result


def get_state_table_csv(extractor: BLSOESDataExtractor, state_code: str, 
                       state_cache: Dict[str, Optional[pd.DataFrame]], 
                       max_rows: int = 100, cache_dir: str = "bls_cache") -> str:
    """
    Get state occupation table as CSV string, using disk cache and in-memory cache.
    Limits the number of rows to avoid exceeding token limits.
    
    Args:
        extractor: BLSOESDataExtractor instance
        state_code: Two-letter state code
        state_cache: Dictionary to cache state data in memory
        max_rows: Maximum number of rows to include (default: 100)
        cache_dir: Directory to store cached state data files (default: "bls_cache")
        
    Returns:
        CSV string of state occupation data, or empty string if unavailable
    """
    # Handle NaN/missing values (handle NaN, None, empty string, or whitespace-only)
    if pd.isna(state_code) or state_code is None or str(state_code).strip() == '':
        return ""
    
    # Convert to string, strip whitespace, and uppercase
    state_code = str(state_code).strip().upper()
    
    # Create cache directory if it doesn't exist
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{state_code}.csv")
        print(f"[DEBUG] Cache directory: {os.path.abspath(cache_dir)}, Cache file: {cache_file}")
    except Exception as e:
        print(f"[ERROR] Failed to create cache directory {cache_dir}: {e}")
        cache_file = os.path.join(cache_dir, f"{state_code}.csv")
    
    # Check in-memory cache first
    if state_code in state_cache:
        df = state_cache[state_code]
    # Check disk cache
    elif os.path.exists(cache_file):
        print(f"[INFO] Loading cached data for state: {state_code} from disk...")
        try:
            df = pd.read_csv(cache_file)
            state_cache[state_code] = df  # Also store in memory cache
            print(f"[INFO] Successfully loaded {len(df)} rows for {state_code} from cache")
        except Exception as e:
            print(f"[WARN] Failed to load cache file for {state_code}: {e}, will download...")
            df = None
    else:
        df = None
    
    # Download if not in cache
    if df is None:
        print(f"[INFO] Downloading occupation data for state: {state_code}...")
        try:
            df = extractor.get_state_data(state_code, clean_data=True)
            print(f"[DEBUG] get_state_data returned: type={type(df)}, is None={df is None}, empty={df.empty if df is not None else 'N/A'}, rows={len(df) if df is not None else 0}")
            if df is not None and not df.empty:
                # Save FULL dataframe to disk cache BEFORE truncation
                try:
                    df.to_csv(cache_file, index=False)
                    # Verify file was created
                    if os.path.exists(cache_file):
                        file_size = os.path.getsize(cache_file)
                        print(f"[INFO] Successfully downloaded and cached {len(df)} rows for {state_code} to {cache_file} ({file_size} bytes)")
                    else:
                        print(f"[WARN] Cache file was not created: {cache_file}")
                except Exception as save_error:
                    print(f"[WARN] Failed to save cache file {cache_file}: {save_error}")
                    import traceback
                    traceback.print_exc()
                
                # Also store in memory cache
                state_cache[state_code] = df
            else:
                print(f"[WARN] No data returned for {state_code} (df is None or empty)")
                state_cache[state_code] = None
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {state_code}: {e}")
            import traceback
            traceback.print_exc()
            state_cache[state_code] = None
            df = None
    
    if df is None or df.empty:
        return ""
    
    # Create a copy for truncation (don't modify cached dataframe)
    df_limited = df.copy()
    # Limit rows to avoid exceeding token limits
    if len(df_limited) > max_rows:
        df_limited = df_limited.head(max_rows)
    
    # Convert limited DataFrame to CSV string
    csv_buffer = StringIO()
    df_limited.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def format_prompt(employer_name: str, employer_city: str, employer_state: str,
                 hourly_rate: str, state_table_csv: str) -> str:
    """
    Format the prompt template with employer data and state table.
    
    Args:
        employer_name: Employer name
        employer_city: Employee city
        employer_state: Employee state
        hourly_rate: Hourly rate string
        state_table_csv: CSV string of state occupation data
        
    Returns:
        Formatted prompt string
    """
    # Replace template variables
    prompt = ROLE_DETERMINATION_PROMPT
    prompt = prompt.replace("{{employer_name}}", str(employer_name))
    prompt = prompt.replace("{{employer_city}}", str(employer_city))
    prompt = prompt.replace("{{employer_state}}", str(employer_state))
    prompt = prompt.replace("{{hourly_rate}}", str(hourly_rate))
    
    # Append state table CSV data
    if state_table_csv:
        prompt += "\n\n# Occupational Data CSV:\n"
        prompt += state_table_csv
    
    return prompt


def call_openai_api(llm: ChatOpenAI, prompt_text: str, max_retries: int = 3) -> Tuple[Optional[dict], Dict[str, int]]:
    """
    Call OpenAI API using LangChain ChatOpenAI with the formatted prompt.
    
    Args:
        llm: LangChain ChatOpenAI instance
        prompt_text: Formatted prompt text (includes state table CSV)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (parsed JSON response as dict, token usage dict with 'input_tokens' and 'output_tokens')
        Returns (None, {}) if failed
    """
    for attempt in range(max_retries):
        try:
            # Use LangChain ChatOpenAI to invoke the API
            message = HumanMessage(content=prompt_text)
            response = llm.invoke([message])
            
            # Extract token usage from response
            token_usage = {'input_tokens': 0, 'output_tokens': 0}
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    token_usage['input_tokens'] = usage.get('prompt_tokens', 0)
                    token_usage['output_tokens'] = usage.get('completion_tokens', 0)
            
            # Extract content from LangChain response
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                content = str(response)
            
            # Parse JSON response
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            result = json.loads(content)
            return result, token_usage
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"[WARN] JSON decode error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"[ERROR] JSON decode error after {max_retries} attempts: {e}")
                if 'content' in locals():
                    print(f"[ERROR] Response content: {content[:200]}...")
                return None, {}
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            if attempt < max_retries - 1:
                print(f"[WARN] API error (attempt {attempt + 1}/{max_retries}): {error_type}, retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"[ERROR] API error after {max_retries} attempts: {error_type}: {error_msg}")
                return None, {}
    
    return None, {}


def parse_api_response(response: Optional[dict]) -> Tuple[List[str], List[Optional[int]], str, str]:
    """
    Parse API response to extract fields.
    
    Args:
        response: API response dictionary
        
    Returns:
        Tuple of (predicted_titles, confidence, source, reasoning)
        Always returns 3 titles and 3 confidences (padded with None)
    """
    if response is None:
        return [None, None, None], [None, None, None], "", ""
    
    predicted_titles = response.get("predicted_titles", [])
    confidence = response.get("confidence", [])
    source = response.get("source", "")
    reasoning = response.get("reasoning", "")
    
    # Pad to 3 elements
    while len(predicted_titles) < 3:
        predicted_titles.append(None)
    while len(confidence) < 3:
        confidence.append(None)
    
    # Truncate to 3 elements if more
    predicted_titles = predicted_titles[:3]
    confidence = confidence[:3]
    
    return predicted_titles, confidence, source, reasoning


def process_seo_data(input_csv: str, output_csv: str, model: str = "gpt-4o", 
                    temperature: float = 0.0, verbose: bool = True):
    """
    Main function to process SEO data.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        model: OpenAI model name
        temperature: Temperature setting
        verbose: Whether to print progress
    """
    # Load input CSV
    print(f"[STEP 1/5] Loading input CSV: {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"[STEP 1/5] ✓ Loaded {len(df)} rows")
    
    # Initialize state extractor
    print(f"[STEP 2/5] Initializing state data extractor...")
    extractor = BLSOESDataExtractor(headless=True, verbose=False)  # Disable verbose from extractor
    state_cache: Dict[str, Optional[pd.DataFrame]] = {}
    print(f"[STEP 2/5] ✓ State extractor ready")
    
    # Initialize LangChain ChatOpenAI
    print(f"[STEP 3/5] Initializing OpenAI API client...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Get model name from environment or use default
    model_name = os.getenv("OPENAI_MODEL", model)
    
    # Create ChatOpenAI instance
    # Handle SSL certificate issues by disabling verification (testing only)
    import httpx
    import warnings
    
    # Disable SSL warnings for testing
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    # Create client with SSL verification disabled due to certificate issues
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=60.0,
        http_client=httpx.Client(verify=False, timeout=60.0)
    )
    print(f"[STEP 3/5] ✓ OpenAI client initialized (model: {model_name})")
    
    # Process each row
    print(f"[STEP 4/5] Processing {len(df)} rows...")
    results = []
    total_rows = len(df)
    processed_count = 0
    skipped_count = 0
    
    # Track token usage for cost calculation
    total_input_tokens = 0
    total_output_tokens = 0
    
    for idx, row in df.iterrows():
        print(f"[PROGRESS] Row {idx + 1}/{total_rows}: {row['employer_name']} - {row['employee_city']}, {row['employee_state']}")
        
        # Extract basic info
        employer_name = row['employer_name']
        employee_city = row['employee_city']
        employee_state = row['employee_state']
        rate_str = row['regular_hourly_rate_parsed_counts']
        
        # Skip if missing state (handle NaN, None, empty string, or whitespace-only)
        if pd.isna(employee_state) or employee_state is None or str(employee_state).strip() == '':
            print(f"  → Skipped: Missing state information")
            skipped_count += 1
            continue
        
        # Parse hourly rates
        hourly_rates = parse_hourly_rates(rate_str)
        
        # Parse rate counts to get paystub counts for each rate
        rate_counts = parse_rate_counts(rate_str)
        
        # Skip if no hourly rates
        if not hourly_rates:
            print(f"  → Skipped: No hourly rates found")
            skipped_count += 1
            continue
        
        # Get state table CSV (cached, limited to 100 rows to avoid token limit)
        state_table_csv = get_state_table_csv(extractor, employee_state, state_cache, max_rows=100)
        
        # Process each hourly rate
        for rate_idx, hourly_rate in enumerate(hourly_rates):
            print(f"  → Processing rate {rate_idx + 1}/{len(hourly_rates)}: ${hourly_rate}/hr")
            
            # Format prompt
            prompt_text = format_prompt(
                employer_name, employee_city, employee_state, 
                hourly_rate, state_table_csv
            )
            
            # Call OpenAI API
            print(f"    → Calling OpenAI API...", end=" ", flush=True)
            response, token_usage = call_openai_api(llm, prompt_text)
            
            # Extract token counts
            input_tokens = token_usage.get('input_tokens', 0)
            output_tokens = token_usage.get('output_tokens', 0)
            
            # Track token usage
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            if response:
                print("✓ Success")
                # Parse response
                predicted_titles, confidence, source, reasoning = parse_api_response(response)
                
                # Create result row
                result_row = row.to_dict()
                
                # Remove unwanted columns (handle various naming variations)
                columns_to_remove = ['paystub_count', 'Unnamed: 5', 'unnamed:5', 'Unnamed:5']
                for col in columns_to_remove:
                    if col in result_row:
                        del result_row[col]
                
                # Update hourly rate field to only contain the corresponding rate
                result_row['regular_hourly_rate_parsed_counts'] = hourly_rate
                
                # Add paystub count for this specific rate
                paystub_count = rate_counts.get(hourly_rate, 0)
                result_row['paystub_count_by_rate'] = paystub_count
                
                # Add API response fields (always 3 titles and 3 confidences)
                result_row['predicted_title_1'] = predicted_titles[0]
                result_row['confidence_1'] = confidence[0]
                result_row['predicted_title_2'] = predicted_titles[1]
                result_row['confidence_2'] = confidence[1]
                result_row['predicted_title_3'] = predicted_titles[2]
                result_row['confidence_3'] = confidence[2]
                result_row['source'] = source
                result_row['reasoning'] = reasoning
                
                # Add token usage and cost for this API call
                result_row['input_tokens'] = input_tokens
                result_row['output_tokens'] = output_tokens
                result_row['total_tokens'] = input_tokens + output_tokens
                result_row['cost_usd'] = calculate_openai_cost(model_name, input_tokens, output_tokens)
                
                results.append(result_row)
                processed_count += 1
            else:
                print("✗ Failed")
            
            # Rate limiting - small delay between API calls
            time.sleep(0.5)
    
    print(f"[STEP 4/5] ✓ Processing complete: {processed_count} processed, {skipped_count} skipped")
    
    # Calculate and display API costs
    total_cost = calculate_openai_cost(model_name, total_input_tokens, total_output_tokens)
    if total_cost > 0:
        print(f"\n[COST SUMMARY]")
        print(f"  Model: {model_name}")
        print(f"  Total Input Tokens: {total_input_tokens:,}")
        print(f"  Total Output Tokens: {total_output_tokens:,}")
        print(f"  Total Tokens: {total_input_tokens + total_output_tokens:,}")
        print(f"  Estimated Cost: ${total_cost:.4f}")
    else:
        print(f"\n[COST SUMMARY] Cost calculation not available for model: {model_name}")
    
    # Create output DataFrame
    print(f"[STEP 5/5] Saving results to: {output_csv}...")
    output_df = pd.DataFrame(results)
    
    # Add summary row with totals if we have results
    if len(output_df) > 0 and total_input_tokens > 0:
        # Create a summary row with totals
        summary_row = {}
        # Fill all columns with empty values except the summary columns
        for col in output_df.columns:
            if col not in ['input_tokens', 'output_tokens', 'total_tokens', 'cost_usd']:
                summary_row[col] = None
        
        # Add summary values
        summary_row['employer_name'] = 'TOTAL SUMMARY'
        summary_row['input_tokens'] = total_input_tokens
        summary_row['output_tokens'] = total_output_tokens
        summary_row['total_tokens'] = total_input_tokens + total_output_tokens
        summary_row['cost_usd'] = total_cost
        
        # Append summary row
        summary_df = pd.DataFrame([summary_row])
        output_df = pd.concat([output_df, summary_df], ignore_index=True)
    
    output_df.to_csv(output_csv, index=False)
    print(f"[STEP 5/5] ✓ Saved {len(output_df)} rows to {output_csv}")
    
    return output_df


if __name__ == "__main__":
    import sys
    
    # Default file paths
    input_csv = "Untitled spreadsheet - Sheet1 (1).csv"
    output_csv = "SEO_test_output.csv"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    # Model name - user specified gpt-4.1, but that might not exist
    # Try gpt-4.1 first, fall back to gpt-4o if not available
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")  # Changed default to gpt-4o since gpt-4.1 doesn't exist
    # If OPENAI_MODEL env var is set to gpt-4.1, we'll try it and fall back to gpt-4o
    
    # Process all entries
    print("=" * 60)
    print("PROCESSING ALL ENTRIES")
    print("=" * 60)
    
    try:
        process_seo_data(input_csv, output_csv, model=model, temperature=0.0, verbose=True)
        print("\n" + "=" * 60)
        print("✓ ALL PROCESSING COMPLETE!")
        print(f"✓ Results saved to: {output_csv}")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

