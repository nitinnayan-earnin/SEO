'''
Prompt for determining the role of an employee based on employer name, city, state, and hourly rate.
'''


ROLE_DETERMINATION_PROMPT = """
Always extract and output exactly three plausible job titles ("predicted_titles") for a given employee, even if confidence is low and strong evidence is lacking. Do NOT use null placeholders—fill all three slots with the most reasonable, plausible, or speculative titles, ranked from most to least likely, and accompanied by confidence scores (which may be low if evidence is weak). Never provide null values in "predicted_titles" or "confidence"; always provide three of each. Proceed with stepwise reasoning before listing conclusions. Use internet sources as your primary research and the occupational CSV only as supplementary context (for generalized category mapping). 

Your only output must be a single, flat JSON object with these four keys:
- "predicted_titles": list of three plausible job titles (strings), always filled—never null.
- "confidence": list of three integer scores (1-100), each score matching the plausibility and lined up with the predicted_titles, even if speculative (use the lowest tier of confidence rubric for highly tentative predictions).
- "source": brief string listing the research sources (e.g., “LinkedIn, company jobs site, Glassdoor,” with domains/urls if possible).
- "reasoning": concise (30-40 words) justification, summarizing evidence and rationale for all selections, including the use of generalized CSV categories as necessary, and transparency if some selections are based on weaker or indirect evidence.

Employer Name: {{employer_name}}
Employer City: {{employer_city}}
Employer State: {{employer_state}}
Hourly Rate: {{hourly_rate}}

# Steps

1. Analyze all inputs: employer_name, employer_city, employer_state, hourly_rate.
2. Use internet-based research to determine the three most likely/plausible job titles for this employer at this location/pay. If evidence is weak, use related industry or role data to infer plausible titles, but always provide three.
3. Rank titles in descending order of plausibility, justifying each selection with evidence or logical inference.
4. Assign a confidence score (1-100) for each, using the rubric below. If evidence is tenuous, assign low confidence (but still provide titles).
5. Reference the CSV only for supplementary category mapping where appropriate—never use as sole source or filter.
6. Compose “reasoning” to fit all three titles and confidence scores, indicating when weaker evidence or generalization is used.
7. Output only the required JSON object—never include nulls, additional fields, markup, or commentary.

# Confidence Scoring Metrics

Assign each confidence integer (1-100) per the following rules:
- High certainty (90-100): Multiple strong sources confirm the title for this employer/location/rate; CSV corroborates directly or via generalized category.
- Medium-high (75-89): At least one strong source supports; minor discrepancies.
- Moderate (60-74): Some alignment, but with gaps or inconsistencies in evidence.
- Low-moderate (40-59): Very limited or indirect evidence; inference from related roles or employers.
- Low (1-39): Titles are speculative, based only on weak, overlapping evidence or general context. Use these for “filler” when required to guess to reach three.
- Use 100 only for overwhelmingly certain matches.
- Justify all confidences in “reasoning” (especially low or speculative cases).

# Output Format

Output a single JSON object (no markdown, no code block, no extra text) with these fields only:
- "predicted_titles": [string, string, string] (always three, no nulls)
- "confidence": [int (1-100), int (1-100), int (1-100)] (always three, no nulls)
- "source": brief string, listing research sources (with URLs/domains if possible)
- "reasoning": concise 30-40 word justification summarizing evidence/rationale for all titles, noting low-confidence or speculated titles as needed, and CSV category mapping if relevant.

# Examples

**Example 1:**
{
  "predicted_titles": ["Mechanical Assembler", "Production Technician", "Assembly Operator"],
  "confidence": [95, 87, 82],
  "source": "Indeed, company site, industry salary data",
  "reasoning": "Multiple listings support all three titles in order; pay and duties align. CSV occupation 'Assemblers' supports the group as a general category."
}

**Example 2:**
{
  "predicted_titles": ["Retail Sales Associate", "Cashier", "Customer Service Assistant"],
  "confidence": [89, 27, 15],
  "source": "Company jobs page, Glassdoor",
  "reasoning": "Only 'Retail Sales Associate' is confirmed. 'Cashier' and 'Customer Service Assistant' are plausible but speculative. CSV offers generic 'Sales' occupation as a general match."
}

**Example 3:**
{
  "predicted_titles": ["Barista", "Cafe Server", "Food Counter Attendant"],
  "confidence": [86, 62, 16],
  "source": "Starbucks careers, Glassdoor",
  "reasoning": "'Barista' and 'Cafe Server' are supported; 'Food Counter Attendant' is based on industry norms. All fit under the CSV's 'Food Preparation and Serving Related Occupations' category."
}

**Example 4 (when evidence is extremely weak):**
{
  "predicted_titles": ["Production Worker", "Warehouse Associate", "Material Handler"],
  "confidence": [28, 15, 7],
  "source": "Industry reports, similar employers' job postings",
  "reasoning": "Titles are based on general industry practices for this pay and region; very limited direct evidence available. All map to the CSV's broad 'Production Occupations'."
}

(Real examples must substitute variables for details, and every output must provide three titles and three scores, even if some are highly speculative with low scores.)

# Notes

- NEVER use nulls in any field. Always produce three "predicted_titles" and three "confidence" scores, no matter how speculative.
- If you cannot find strong evidence for three titles, fill remaining slots with plausible or related titles from similar roles, sectors, or employers, and assign appropriately low confidence scores.
- The "reasoning" should explicitly note when you are making speculative or filler guesses due to weak evidence.
- The CSV is only for broad contextual reference, never as a required source.
- Output a single, flat JSON object with exactly four fields.

# Reminders

- Research and reason FIRST, then list conclusions (titles/confidence) LAST.
- You must always provide exactly three titles and three confidence scores, even if some are speculative or very low-confidence.
- Output must be a flat JSON object; do not include markdown, code blocks, explanatory text, or nulls in any field. 

Remember: Your core task is to always output exactly three plausible job titles and three matching confidence scores, even if evidence for some is weak. Use internet research and logic first, then CSV for broad category mapping. Always justify confidences, especially speculative fillers. Output only the specified JSON object, following this schema."""