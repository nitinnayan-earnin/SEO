import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from bs4 import BeautifulSoup
import re
from typing import Optional


class BLSOESDataExtractor:
    """
    A class to extract Occupational Employment and Wage Statistics (OEWS) data
    from the BLS website for different states.
    """
    
    STATE_LINKS = {
        "AK": "https://data.bls.gov/oes/#/area/0200000",
        "AL": "https://data.bls.gov/oes/#/area/0100000",
        "AR": "https://data.bls.gov/oes/#/area/0500000",
        "AS": "https://data.bls.gov/oes/#/area/6000000",
        "AZ": "https://data.bls.gov/oes/#/area/0400000",
        "CA": "https://data.bls.gov/oes/#/area/0600000",
        "CO": "https://data.bls.gov/oes/#/area/0800000",
        "CT": "https://data.bls.gov/oes/#/area/0900000",
        "DC": "https://data.bls.gov/oes/#/area/1100000",
        "DE": "https://data.bls.gov/oes/#/area/1000000",
        "FL": "https://data.bls.gov/oes/#/area/1200000",
        "GA": "https://data.bls.gov/oes/#/area/1300000",
        "GU": "https://data.bls.gov/oes/#/area/6600000",
        "HI": "https://data.bls.gov/oes/#/area/1500000",
        "IA": "https://data.bls.gov/oes/#/area/1900000",
        "ID": "https://data.bls.gov/oes/#/area/1600000",
        "IL": "https://data.bls.gov/oes/#/area/1700000",
        "IN": "https://data.bls.gov/oes/#/area/1800000",
        "KS": "https://data.bls.gov/oes/#/area/2000000",
        "KY": "https://data.bls.gov/oes/#/area/2100000",
        "LA": "https://data.bls.gov/oes/#/area/2200000",
        "MA": "https://data.bls.gov/oes/#/area/2500000",
        "MD": "https://data.bls.gov/oes/#/area/2400000",
        "ME": "https://data.bls.gov/oes/#/area/2300000",
        "MI": "https://data.bls.gov/oes/#/area/2600000",
        "MN": "https://data.bls.gov/oes/#/area/2700000",
        "MO": "https://data.bls.gov/oes/#/area/2900000",
        "MP": "https://data.bls.gov/oes/#/area/6900000",
        "MS": "https://data.bls.gov/oes/#/area/2800000",
        "MT": "https://data.bls.gov/oes/#/area/3000000",
        "NC": "https://data.bls.gov/oes/#/area/3700000",
        "ND": "https://data.bls.gov/oes/#/area/3800000",
        "NE": "https://data.bls.gov/oes/#/area/3100000",
        "NH": "https://data.bls.gov/oes/#/area/3300000",
        "NJ": "https://data.bls.gov/oes/#/area/3400000",
        "NM": "https://data.bls.gov/oes/#/area/3500000",
        "NV": "https://data.bls.gov/oes/#/area/3200000",
        "NY": "https://data.bls.gov/oes/#/area/3600000",
        "OH": "https://data.bls.gov/oes/#/area/3900000",
        "OK": "https://data.bls.gov/oes/#/area/4000000",
        "OR": "https://data.bls.gov/oes/#/area/4100000",
        "PA": "https://data.bls.gov/oes/#/area/4200000",
        "PR": "https://data.bls.gov/oes/#/area/7200000",
        "RI": "https://data.bls.gov/oes/#/area/4400000",
        "SC": "https://data.bls.gov/oes/#/area/4500000",
        "SD": "https://data.bls.gov/oes/#/area/4600000",
        "TN": "https://data.bls.gov/oes/#/area/4700000",
        "TX": "https://data.bls.gov/oes/#/area/4800000",
        "UT": "https://data.bls.gov/oes/#/area/4900000",
        "VA": "https://data.bls.gov/oes/#/area/5100000",
        "VI": "https://data.bls.gov/oes/#/area/7800000",
        "VT": "https://data.bls.gov/oes/#/area/5000000",
        "WA": "https://data.bls.gov/oes/#/area/5300000",
        "WI": "https://data.bls.gov/oes/#/area/5500000",
        "WV": "https://data.bls.gov/oes/#/area/5400000",
        "WY": "https://data.bls.gov/oes/#/area/5600000",
    }
    
    def __init__(self, headless: bool = True, verbose: bool = True):
        """
        Initialize the BLS OES Data Extractor.
        
        Args:
            headless: Whether to run Chrome in headless mode (default: True)
            verbose: Whether to print progress messages (default: True)
        """
        self.headless = headless
        self.verbose = verbose
    
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def clean_data_value(self, value) -> str:
        """
        Clean a single data value by removing parentheses and their contents.
        
        Args:
            value: The value to clean
            
        Returns:
            Cleaned value string
            
        Examples:
            - "()4,250,430" -> "4,250,430"
            - "(5)-" -> "-"
            - "(4)-" -> "-"
            - "()$36.69" -> "$36.69"
            - "()0.0" -> "0.0"
        """
        if pd.isna(value) or value == '':
            return value
        
        # Convert to string if not already
        value_str = str(value)
        
        # Remove empty parentheses at the start: "()" -> ""
        value_str = re.sub(r'^\(\)', '', value_str)
        
        # Remove numbered parentheses at the start: "(5)", "(4)", etc. -> ""
        value_str = re.sub(r'^\(\d+\)', '', value_str)
        
        # Remove any remaining empty parentheses: "()" -> ""
        value_str = re.sub(r'\(\)', '', value_str)
        
        # Remove any remaining numbered parentheses: "(5)", "(4)", etc. -> ""
        value_str = re.sub(r'\(\d+\)', '', value_str)
        
        return value_str.strip()
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by removing parentheses and their contents from all columns
        except the "Occupation (SOC code)" column.
        
        Args:
            df: The DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Find the occupation column (it might have variations in the name)
        occupation_col = None
        for col in df_cleaned.columns:
            if 'occupation' in col.lower() and 'soc' in col.lower():
                occupation_col = col
                break
        
        # Clean all columns except the occupation column
        for col in df_cleaned.columns:
            if col != occupation_col:
                # Apply cleaning function to each value in the column
                df_cleaned[col] = df_cleaned[col].apply(self.clean_data_value)
        
        return df_cleaned
    
    def get_table_data_with_selenium(self, url: str) -> Optional[pd.DataFrame]:
        """
        Use Selenium to render the JavaScript and extract table data from the page.
        
        Args:
            url: The URL to extract data from
            
        Returns:
            pandas DataFrame containing the extracted table data, or None if extraction fails
        """
        self._print(f"Loading page with Selenium: {url}")
        
        # Set up Chrome options for headless browsing
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        driver = None
        try:
            # Initialize the driver
            driver = webdriver.Chrome(options=chrome_options)
            
            # Navigate to the page
            driver.get(url)
            
            # Wait for the page to load and for Angular to render
            self._print("Waiting for page to load...")
            time.sleep(10)  # Give Angular more time to load
            
            # Try to find and click the "Retrieve Data" button if it exists
            try:
                # Look for buttons or links that might trigger data loading
                retrieve_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Retrieve') or contains(text(), 'Get Data')]")
                if retrieve_buttons:
                    self._print("Found retrieve button, clicking...")
                    retrieve_buttons[0].click()
                    time.sleep(5)  # Wait for data to load after clicking
            except:
                pass
            
            # Wait for data table to appear (look for specific table classes or IDs)
            try:
                # Wait for table with actual data (not just UI elements)
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
                )
                self._print("Data table found!")
            except:
                self._print("Data table not found, but continuing...")
            
            # Additional wait for data to fully load
            time.sleep(5)
            
            # Get the page source after JavaScript has rendered
            page_source = driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Find all tables
            tables = soup.find_all('table')
            
            if not tables:
                self._print("No tables found on the page.")
                driver.quit()
                return None
            
            # Extract data from tables, filtering out UI tables
            all_data = []
            for idx, table in enumerate(tables):
                self._print(f"Processing table {idx + 1}...")
                
                # Skip tables that are likely UI elements (small tables with buttons)
                table_text = table.get_text(strip=True).lower()
                if 'retrieve data' in table_text and len(table.find_all('tr')) < 3:
                    self._print(f"  Skipping UI table {idx + 1} (contains 'Retrieve Data')")
                    continue
                
                # Extract headers
                headers = []
                header_row = table.find('thead')
                if header_row:
                    header_cells = header_row.find_all(['th', 'td'])
                    headers = [cell.get_text(strip=True) for cell in header_cells]
                else:
                    # Try to find headers in first row
                    first_row = table.find('tr')
                    if first_row:
                        header_cells = first_row.find_all(['th', 'td'])
                        headers = [cell.get_text(strip=True) for cell in header_cells]
                
                # Extract rows
                rows = []
                tbody = table.find('tbody')
                if tbody:
                    table_rows = tbody.find_all('tr')
                else:
                    table_rows = table.find_all('tr')
                    # Skip header row if headers were found in first row
                    if headers and table_rows:
                        table_rows = table_rows[1:]
                
                for row in table_rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    # Filter out rows that are just buttons or UI elements
                    if row_data and len(row_data) > 1:  # Only add rows with multiple cells
                        # Skip rows that are just buttons
                        row_text = ' '.join(row_data).lower()
                        if 'retrieve data' not in row_text and 'reset form' not in row_text:
                            rows.append(row_data)
                
                # Only add tables with substantial data (more than 2 rows)
                if headers and len(rows) > 2:
                    # Create DataFrame
                    df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if rows else headers)
                    all_data.append(df)
                    self._print(f"  Extracted {len(rows)} rows from table {idx + 1}")
                elif headers and rows:
                    self._print(f"  Table {idx + 1} has only {len(rows)} rows, skipping (likely UI element)")
            
            if driver:
                driver.quit()
            
            if all_data:
                # Combine all tables or return the first one
                result_df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
                return result_df
            else:
                self._print("No data extracted from tables.")
                return None
                
        except Exception as e:
            self._print(f"Error with Selenium: {e}")
            if driver:
                driver.quit()
            return None
    
    def get_state_data(self, state_code: str, clean_data: bool = True) -> Optional[pd.DataFrame]:
        """
        Get OEWS data for a specific state.
        
        Args:
            state_code: Two-letter state code (e.g., "CA", "NY", "NJ")
            clean_data: Whether to clean the data by removing parentheses (default: True)
            
        Returns:
            pandas DataFrame containing the state's OEWS data, or None if extraction fails
            
        Raises:
            ValueError: If the state code is not found in STATE_LINKS
        """
        state_code = state_code.upper()
        
        if state_code not in self.STATE_LINKS:
            available_states = ', '.join(sorted(self.STATE_LINKS.keys()))
            raise ValueError(f"State code '{state_code}' not found. Available states: {available_states}")
        
        url = self.STATE_LINKS[state_code]
        self._print(f"Extracting data for state: {state_code}")
        
        # Get table data
        df = self.get_table_data_with_selenium(url)
        
        if df is not None and clean_data:
            self._print("Cleaning data (removing parentheses and their contents)...")
            df = self.clean_dataframe(df)
        
        return df
    
    def get_state_url(self, state_code: str) -> str:
        """
        Get the URL for a specific state.
        
        Args:
            state_code: Two-letter state code (e.g., "CA", "NY", "NJ")
            
        Returns:
            The URL for the state's OEWS data page
            
        Raises:
            ValueError: If the state code is not found in STATE_LINKS
        """
        state_code = state_code.upper()
        
        if state_code not in self.STATE_LINKS:
            available_states = ', '.join(sorted(self.STATE_LINKS.keys()))
            raise ValueError(f"State code '{state_code}' not found. Available states: {available_states}")
        
        return self.STATE_LINKS[state_code]
    
    def get_available_states(self) -> list:
        """
        Get a list of all available state codes.
        
        Returns:
            List of state codes (sorted alphabetically)
        """
        return sorted(self.STATE_LINKS.keys())


# Main execution (for testing)
if __name__ == "__main__":
    # Create an instance of the class
    extractor = BLSOESDataExtractor(headless=True, verbose=True)
    
    # Example: Get data for New Jersey
    state_code = "NJ"
    
    # Get the data
    df = extractor.get_state_data(state_code, clean_data=True)
    
    if df is not None:
        # Save to CSV
        output_file = f"{state_code.lower()}_table_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nTable data saved to {output_file}")
        print(f"\nData shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nColumn names:")
        print(df.columns.tolist())
    else:
        print("\nCould not extract table data. The page may require manual interaction or the table structure may be different.")
