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
from tqdm import tqdm


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
        List of hourly rate strings (excluding NaN), or empty list if empty/invalid
    """
    if pd.isna(rate_str) or rate_str == '' or rate_str == '{}':
        return []
    
    try:
        # The CSV has double quotes escaped as '""', so we need to handle that
        # First, replace '""' with '"' to unescape
        cleaned = rate_str.replace('""', '"')
        # Parse as JSON
        rate_dict = json.loads(cleaned)
        # Extract keys (the hourly rates) and filter out NaN values
        rates = [rate for rate in rate_dict.keys() if str(rate).upper() != 'NAN']
        return rates
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # Try regex fallback for malformed JSON
        # Extract numbers that look like hourly rates (handles both '""' and '"')
        # Pattern matches: "number" or ""number""
        rates = re.findall(r'"{1,2}([0-9]+\.?[0-9]*)"{1,2}', rate_str)
        if rates:
            # Filter out NaN values
            rates = [rate for rate in rates if str(rate).upper() != 'NAN']
            return rates
        # If still no match, try simpler pattern
        rates = re.findall(r'([0-9]+\.[0-9]+)', rate_str)
        # Filter out NaN values
        rates = [rate for rate in rates if str(rate).upper() != 'NAN']
        return rates


def parse_rate_counts(rate_str: str) -> Dict[str, int]:
    """
    Parse hourly rates and their counts from JSON-like string.
    
    Args:
        rate_str: String like '{"25.72": 1}' or '{"17.25": 1, "23.01": 1}'
                 The CSV has escaped quotes like '{""25.72"": 1}'
        
    Returns:
        Dictionary mapping rate strings to counts (excluding NaN entries)
    """
    if pd.isna(rate_str) or rate_str == '' or rate_str == '{}':
        return {}
    
    try:
        # The CSV has double quotes escaped as '""', so we need to handle that
        # First, replace '""' with '"' to unescape
        cleaned = rate_str.replace('""', '"')
        # Parse as JSON
        rate_dict = json.loads(cleaned)
        # Filter out NaN keys (case-insensitive)
        filtered_dict = {rate: count for rate, count in rate_dict.items() 
                        if str(rate).upper() != 'NAN'}
        return filtered_dict
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Try regex fallback for malformed JSON
        # Extract numbers and their counts
        result = {}
        # Pattern to match: "number": count or ""number"": count
        pattern = r'"{1,2}([0-9]+\.?[0-9]*)"{1,2}:\s*([0-9]+)'
        matches = re.findall(pattern, rate_str)
        for rate, count in matches:
            # Filter out NaN values
            if str(rate).upper() != 'NAN':
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
    
    # Ensure cache_dir is absolute path
    cache_dir = os.path.abspath(cache_dir)
    
    # Create cache directory if it doesn't exist
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{state_code}.csv")
        cache_file = os.path.abspath(cache_file) 
    except Exception as e:
        print(f"[ERROR] Failed to create cache directory {cache_dir}: {e}")
        import traceback
        traceback.print_exc()
        cache_file = os.path.join(cache_dir, f"{state_code}.csv")
    
    # Check in-memory cache first
    if state_code in state_cache:
        df = state_cache[state_code]
    # Check disk cache
    elif os.path.exists(cache_file):
        # print(f"[INFO] Loading cached data for state: {state_code} from disk...")
        try:
            df = pd.read_csv(cache_file)
            state_cache[state_code] = df  # Also store in memory cache
            # print(f"[INFO] Successfully loaded {len(df)} rows for {state_code} from cache")
        except Exception as e:
            print(f"[WARN] Failed to load cache file for {state_code}: {e}, will download...")
            df = None
    else:
        df = None
    
    # Download if not in cache
    if df is None:
        # print(f"[INFO] Downloading occupation data for state: {state_code}...")
        # print(f"[DEBUG] About to call extractor.get_state_data({state_code})...")
        try:
            df = extractor.get_state_data(state_code, clean_data=True)
            # print(f"[DEBUG] get_state_data returned: type={type(df)}, is None={df is None}")
            # print(f"[DEBUG] DataFrame is None - extraction may have failed")
            
            if df is not None and not df.empty:
                # Save FULL dataframe to disk cache BEFORE truncation
                try:
                    df.to_csv(cache_file, index=False)
                    # Verify file was created
                    if os.path.exists(cache_file):
                        file_size = os.path.getsize(cache_file)
                        # print(f"[INFO] Successfully downloaded and cached {len(df)} rows for {state_code}")
                        # print(f"[INFO] Cache file: {cache_file} ({file_size} bytes)")
                        # List cache directory contents for verification
                        try:
                            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
                            # print(f"[INFO] Cache directory now contains {len(cache_files)} files: {cache_files}")
                        except Exception:
                            pass  # Ignore listing errors
                    else:
                        print(f"[ERROR] Cache file was not created: {cache_file}")
                        print(f"[ERROR] Current working directory: {os.getcwd()}")
                        print(f"[ERROR] Cache directory exists: {os.path.exists(cache_dir)}")
                except Exception as save_error:
                    print(f"[ERROR] Failed to save cache file {cache_file}: {save_error}")
                    print(f"[ERROR] Current working directory: {os.getcwd()}")
                    print(f"[ERROR] Cache directory: {cache_dir}")
                    import traceback
                    traceback.print_exc()
                
                # Also store in memory cache
                state_cache[state_code] = df
            else:
                print(f"[WARN] No data returned for {state_code} (df is None or empty)")
                # print(f"[INFO] Selenium may not work in Databricks. To pre-populate cache:")
                # print(f"[INFO]   1. Run 'python download_all_bls_data.py' locally")
                # print(f"[INFO]   2. Upload the 'bls_cache' directory to Databricks")
                # Save a marker file to indicate we tried but got no data
                try:
                    marker_file = os.path.join(cache_dir, f"{state_code}.no_data")
                    with open(marker_file, 'w') as f:
                        f.write(f"No data extracted for {state_code}\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"\nTo fix: Run 'python download_all_bls_data.py' locally and upload bls_cache to Databricks\n")
                    # print(f"[DEBUG] Created marker file: {marker_file}")
                except Exception as marker_error:
                    print(f"[WARN] Could not create marker file: {marker_error}")
                state_cache[state_code] = None
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {state_code}: {e}")
            import traceback
            traceback.print_exc()
            # Save error marker file
            try:
                error_file = os.path.join(cache_dir, f"{state_code}.error")
                with open(error_file, 'w') as f:
                    f.write(f"Error extracting data for {state_code}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                # print(f"[DEBUG] Created error marker file: {error_file}")
            except Exception:
                pass
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


def process_seo_data(input_source: str = None, output_dest: str = None, 
                    input_table: str = None, output_table: str = None,
                    num_entries: str = "full", model: str = "gpt-4o", 
                    temperature: float = 0.0, verbose: bool = True):
    """
    Main function to process SEO data.
    
    Args:
        input_source: Path to input CSV file (for local execution)
        output_dest: Path to output CSV file (for local execution)
        input_table: Databricks table name (e.g., 'datascience_scratchpad.paystub_employer_city_metadata_pseo_251208')
        output_table: Databricks output table name (e.g., 'datascience_scratchpad.pseo_output_251209')
        num_entries: Number of entries to process, or "full" for all entries
        model: OpenAI model name
        temperature: Temperature setting
        verbose: Whether to print progress
    """
    # Try to load from input table first, fall back to CSV if that fails
    use_tables = False
    spark = None
    
    if input_table:
        try:
            from pyspark.sql import SparkSession  # type: ignore
            # spark = SparkSession.builder.getOrCreate()
            print(f"[STEP 1/5] Loading input from Databricks table: {input_table}...")
            df = spark.table(input_table).toPandas()
            # print(f"[STEP 1/5] ✓ Loaded {len(df)} rows from table")
            use_tables = True
        except Exception as e:
            print(f"[WARN] Failed to load from table {input_table}: {e}")
            print(f"[WARN] Falling back to CSV input...")
            if input_source:
                print(f"[STEP 1/5] Loading input CSV: {input_source}...")
                df = pd.read_csv(input_source)
                print(f"[STEP 1/5] ✓ Loaded {len(df)} rows from CSV")
            else:
                raise ValueError(f"Failed to load from table and no input_source provided: {e}")
    elif input_source:
        print(f"[STEP 1/5] Loading input CSV: {input_source}...")
        df = pd.read_csv(input_source)
        print(f"[STEP 1/5] ✓ Loaded {len(df)} rows from CSV")
    else:
        raise ValueError("Either input_table or input_source must be provided")
    
    # Apply num_entries limit
    original_count = len(df)
    if num_entries != "full":
        try:
            num_entries_int = int(num_entries)
            if num_entries_int > 0 and num_entries_int < original_count:
                df = df.iloc[:num_entries_int].copy()
                print(f"[STEP 1/5] Limited to {len(df)} entries (requested: {num_entries_int}, available: {original_count})")
            elif num_entries_int >= original_count:
                print(f"[STEP 1/5] Requested {num_entries_int} entries but only {original_count} available, processing all")
        except ValueError:
            print(f"[WARN] Invalid num_entries value: {num_entries}, processing all entries")
    
    print(f"[STEP 1/5] Processing {len(df)} rows")
    
    # Initialize state extractor
    print(f"[STEP 2/5] Initializing state data extractor...")
    extractor = BLSOESDataExtractor(headless=True, verbose=False)  # Disable verbose from extractor
    state_cache: Dict[str, Optional[pd.DataFrame]] = {}
    
    # Determine cache directory - use absolute path
    # This ensures cache persists in Databricks
    try:
        if use_tables:
            # In Databricks, use dbfs or local path
            cache_dir = "/dbfs/bls_cache" if os.path.exists("/dbfs") else os.path.abspath("bls_cache")
        else:
            # For local execution, use directory based on input source
            if input_source:
                if os.path.isabs(input_source):
                    input_dir = os.path.dirname(input_source)
                else:
                    input_dir = os.path.dirname(os.path.abspath(input_source))
                if not input_dir:
                    input_dir = os.getcwd()
                cache_dir = os.path.join(input_dir, "bls_cache")
            else:
                cache_dir = os.path.abspath("bls_cache")
            cache_dir = os.path.abspath(cache_dir)
        print(f"[STEP 2/5] Cache directory: {cache_dir}")
        
        # Test cache directory is writable
        try:
            os.makedirs(cache_dir, exist_ok=True)
            test_file = os.path.join(cache_dir, ".test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"[STEP 2/5] ✓ Cache directory is writable: {cache_dir}")
        except Exception as test_error:
            print(f"[WARN] Cache directory may not be writable: {test_error}")
            print(f"[WARN] Attempting to continue anyway...")
    except Exception as e:
        # Fallback to current directory
        cache_dir = os.path.abspath("bls_cache")
        print(f"[STEP 2/5] Using fallback cache directory: {cache_dir} (error: {e})")
    
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
    
    # Create progress bar with total, percentage, and ETA
    pbar = tqdm(df.iterrows(), total=total_rows, desc="Processing entries", unit="entry")
    for idx, row in pbar:
        # Update progress bar description with current entry info
        employer_name = row['employer_name']
        employee_city = row['employee_city']
        employee_state = row['employee_state']
        rate_str = row['regular_hourly_rate_parsed_counts']
        
        # Skip if missing state (handle NaN, None, empty string, or whitespace-only)
        if pd.isna(employee_state) or employee_state is None or str(employee_state).strip() == '':
            skipped_count += 1
            continue
        
        # Parse hourly rates
        hourly_rates = parse_hourly_rates(rate_str)
        
        # Parse rate counts to get paystub counts for each rate
        rate_counts = parse_rate_counts(rate_str)
        
        # Skip if no hourly rates
        if not hourly_rates:
            skipped_count += 1
            continue
        
        # Get state table CSV (cached, limited to 100 rows to avoid token limit)
        state_table_csv = get_state_table_csv(extractor, employee_state, state_cache, max_rows=100, cache_dir=cache_dir)
        
        # Process each hourly rate
        for rate_idx, hourly_rate in enumerate(hourly_rates):
            # print(f"  → Processing rate {rate_idx + 1}/{len(hourly_rates)}: ${hourly_rate}/hr")
            
            # Format prompt
            prompt_text = format_prompt(
                employer_name, employee_city, employee_state, 
                hourly_rate, state_table_csv
            )
            
            # Call OpenAI API
            # print(f"    → Calling OpenAI API...", end=" ", flush=True)
            response, token_usage = call_openai_api(llm, prompt_text)
            
            # Extract token counts
            input_tokens = token_usage.get('input_tokens', 0)
            output_tokens = token_usage.get('output_tokens', 0)
            
            # Track token usage
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            if response:
                # print("✓ Success")
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
    
    # Save output - use same mode as input (table or CSV)
    if use_tables and output_table:
        print(f"[STEP 5/5] Saving results to Databricks table: {output_table}...")
        # Convert pandas DataFrame to Spark DataFrame
        if spark is None:
            from pyspark.sql import SparkSession  # type: ignore
            # spark = SparkSession.builder.getOrCreate()
        spark_df = spark.createDataFrame(output_df)
        # Write to table with overwrite mode
        spark_df.write.mode("overwrite") \
            .option('overwriteSchema', 'true') \
            .saveAsTable(output_table)
        print(f"[STEP 5/5] ✓ Saved {len(output_df)} rows to table {output_table}")
    elif output_dest:
        print(f"[STEP 5/5] Saving results to: {output_dest}...")
        output_df.to_csv(output_dest, index=False)
        print(f"[STEP 5/5] ✓ Saved {len(output_df)} rows to {output_dest}")
    else:
        if use_tables:
            raise ValueError("output_table must be provided when using table input")
        else:
            raise ValueError("output_dest must be provided when using CSV input")
    
    # Print cache directory summary
    try:
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            csv_files = [f for f in cache_files if f.endswith('.csv')]
            marker_files = [f for f in cache_files if f.endswith('.no_data') or f.endswith('.error')]
            print(f"\n[CACHE SUMMARY]")
            print(f"  Cache directory: {cache_dir}")
            print(f"  CSV files cached: {len(csv_files)}")
            if csv_files:
                print(f"  Cached states: {[f.replace('.csv', '') for f in csv_files]}")
            if marker_files:
                print(f"  Marker files (failed/empty): {len(marker_files)}")
                print(f"  Marker files: {marker_files}")
        else:
            print(f"\n[WARN] Cache directory does not exist: {cache_dir}")
    except Exception as e:
        print(f"\n[WARN] Could not list cache directory: {e}")
    
    return output_df


if __name__ == "__main__":
    import sys
    
    # Default settings - try Databricks tables first
    input_table = "datascience_scratchpad.paystub_employer_city_metadata_pseo_251208"
    output_table = "datascience_scratchpad.pseo_output_251209"
    input_source = None
    output_dest = None
    
    # Get num_entries from environment or use default
    num_entries = os.getenv("NUM_ENTRIES", "full")
    
    # Allow command line arguments
    # Usage: python process_seo_data.py [input_table/input_source] [output_table/output_dest] [num_entries]
    if len(sys.argv) > 1:
        # Try to determine if it's a table (contains dot) or file path
        arg1 = sys.argv[1]
        if '.' in arg1 and '/' not in arg1 and '\\' not in arg1:
            # Looks like a table name
            input_table = arg1
            input_source = None
        else:
            # Looks like a file path
            input_source = arg1
            input_table = None
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        if '.' in arg2 and '/' not in arg2 and '\\' not in arg2:
            # Looks like a table name
            output_table = arg2
            output_dest = None
        else:
            # Looks like a file path
            output_dest = arg2
            output_table = None
    if len(sys.argv) > 3:
        num_entries = sys.argv[3]
    
    # Model name
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Process entries
    print("=" * 60)
    if num_entries == "full":
        print("PROCESSING ALL ENTRIES")
    else:
        print(f"PROCESSING {num_entries} ENTRIES")
    print("=" * 60)
    
    try:
        process_seo_data(
            input_source=input_source,
            output_dest=output_dest,
            input_table=input_table,
            output_table=output_table,
            num_entries=num_entries,
            model=model,
            temperature=0.0,
            verbose=True
        )
        print("\n" + "=" * 60)
        print("✓ ALL PROCESSING COMPLETE!")
        if output_table:
            print(f"✓ Results saved to table: {output_table}")
        elif output_dest:
            print(f"✓ Results saved to: {output_dest}")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

