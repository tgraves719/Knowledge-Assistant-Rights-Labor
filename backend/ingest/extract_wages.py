"""
Wage Table Extractor - Extracts Appendix A wage tables to structured JSON.

Creates a deterministic lookup structure for wage queries:
- Classification (All Purpose Clerk, Courtesy Clerk, etc.)
- Experience step (Start, After 520 hours, etc.)
- Effective date (2022-01-23, 2023-01-22, 2024-01-21)
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import CONTRACT_MD_FILE, WAGES_DIR, CONTRACT_ID


@dataclass
class WageStep:
    """A single wage step in a classification."""
    step_name: str
    hours_required: Optional[int]
    months_required: Optional[int]
    rates: dict  # {effective_date: rate}


@dataclass
class Classification:
    """A job classification with its wage progression."""
    name: str
    normalized_name: str  # lowercase with underscores
    is_manager: bool
    steps: list  # List of WageStep


# Effective dates for the contract
EFFECTIVE_DATES = ["2022-01-23", "2023-01-22", "2024-01-21"]
EFFECTIVE_DATE_PATTERNS = ["1/23/2022", "1/22/2023", "1/21/2024"]

# Classifications that are single-rate (no progression)
SINGLE_RATE_CLASSIFICATIONS = [
    "HEAD CLERK",
    "PRODUCE DEPARTMENT MANAGER",
    "FLORAL MANAGER",
    "HEAD BAKER",
    "BAKERY MANAGER",
    "VARIETY MANAGER",
    "MANAGER TRAINEE",
    "FUEL LEAD",
    "FRESH CUT SUPERVISOR",
    "5-STAR CAKE DECORATOR",
]

# Classifications with hour-based progression
HOUR_BASED_CLASSIFICATIONS = [
    "ALL PURPOSE CLERK",
    "BAKERY/FRESH CUT/LIQUOR CLERK",
    "CAKE DECORATOR",
    "NON-FOOD/GM/FLORAL",
]

# Classifications with month-based progression
MONTH_BASED_CLASSIFICATIONS = [
    "COURTESY CLERK",
]


def normalize_classification_name(name: str) -> str:
    """Convert classification name to normalized form."""
    name = name.strip().upper()
    # Replace slashes and special characters
    name = re.sub(r'[/\s]+', '_', name)
    name = re.sub(r'[^A-Z0-9_]', '', name)
    return name.lower()


def parse_hours(step_text: str) -> Optional[int]:
    """Extract hours from step text like 'After 520 hours'."""
    match = re.search(r'After\s+(\d+)\s+hours?', step_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if step_text.lower() == 'start':
        return 0
    return None


def parse_months(step_text: str) -> Optional[int]:
    """Extract months from step text like 'After 36 months'."""
    match = re.search(r'After\s+(\d+)\s+months?', step_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if step_text.lower() == 'start':
        return 0
    return None


def parse_rate(rate_text: str) -> Optional[float]:
    """Extract dollar amount from rate text like '$16.00'."""
    match = re.search(r'\$?([\d,]+\.?\d*)', rate_text)
    if match:
        return float(match.group(1).replace(',', ''))
    return None


def extract_wages(md_content: str) -> dict:
    """Extract all wage tables from the contract markdown."""
    
    # Find Appendix A section - look for the wage table header
    appendix_start = md_content.find("Safeway Pueblo Clerks")
    if appendix_start == -1:
        appendix_start = md_content.find("HEAD CLERK")
    
    if appendix_start == -1:
        print("Warning: Could not find wage tables")
        return {'contract_id': CONTRACT_ID, 'effective_dates': EFFECTIVE_DATES, 'classifications': {}}
    
    appendix_content = md_content[appendix_start:]
    
    # Find the end of wage tables (Letters of Understanding)
    lou_start = appendix_content.find("SAFEWAY INC. CLERKS LETTERS OF UNDERSTANDING")
    if lou_start > 0:
        appendix_content = appendix_content[:lou_start]
    
    classifications = {}
    current_classification = None
    current_steps = []
    
    # Extract all table rows - the format is:
    # <tr>
    #     <td>NAME</td>
    # <td>$RATE1</td>
    # <td>$RATE2</td>
    # <td>$RATE3</td>
    # </tr>
    
    # Find all <tr>...</tr> blocks
    tr_pattern = r'<tr[^>]*>(.*?)</tr>'
    td_pattern = r'<td[^>]*>([^<]*)</td>'
    colspan_pattern = r'<td\s+colspan[^>]*>([^<]+)</td>'
    
    tr_matches = re.findall(tr_pattern, appendix_content, re.DOTALL | re.IGNORECASE)
    
    for tr_content in tr_matches:
        # Check for colspan (classification header)
        colspan_match = re.search(colspan_pattern, tr_content)
        if colspan_match:
            class_name = colspan_match.group(1).strip()
            if class_name and len(class_name) > 2:
                # Save previous classification
                if current_classification and current_steps:
                    classifications[current_classification['normalized_name']] = {
                        'name': current_classification['name'],
                        'normalized_name': current_classification['normalized_name'],
                        'is_manager': current_classification['is_manager'],
                        'steps': current_steps
                    }
                
                # Start new classification
                current_classification = {
                    'name': class_name,
                    'normalized_name': normalize_classification_name(class_name),
                    'is_manager': 'MANAGER' in class_name.upper() or 'HEAD' in class_name.upper()
                }
                current_steps = []
                continue
        
        # Extract all td values
        td_values = re.findall(td_pattern, tr_content)
        
        if len(td_values) >= 4:
            cell1 = td_values[0].strip()
            cell2 = td_values[1].strip() if len(td_values) > 1 else ""
            cell3 = td_values[2].strip() if len(td_values) > 2 else ""
            cell4 = td_values[3].strip() if len(td_values) > 3 else ""
            
            # Skip header rows
            if 'CLASSIFICATION' in cell1.upper() or 'Effective' in cell1:
                continue
            
            # Skip date-only rows
            if re.match(r'^\d+/\d+/\d+$', cell2):
                continue
            
            # Parse rates
            rate1 = parse_rate(cell2)
            rate2 = parse_rate(cell3)
            rate3 = parse_rate(cell4)
            
            if rate1 and rate2 and rate3:
                # This is a wage row
                if current_classification is None:
                    # This is a single-rate classification (manager position)
                    class_name = cell1
                    norm_name = normalize_classification_name(class_name)
                    classifications[norm_name] = {
                        'name': class_name,
                        'normalized_name': norm_name,
                        'is_manager': 'MANAGER' in class_name.upper() or 'HEAD' in class_name.upper(),
                        'steps': [{
                            'step_name': 'Rate',
                            'hours_required': None,
                            'months_required': None,
                            'rates': {
                                EFFECTIVE_DATES[0]: rate1,
                                EFFECTIVE_DATES[1]: rate2,
                                EFFECTIVE_DATES[2]: rate3
                            }
                        }]
                    }
                else:
                    # This is a step in the current classification
                    step_name = cell1
                    hours = parse_hours(step_name)
                    months = parse_months(step_name)
                    
                    current_steps.append({
                        'step_name': step_name,
                        'hours_required': hours,
                        'months_required': months,
                        'rates': {
                            EFFECTIVE_DATES[0]: rate1,
                            EFFECTIVE_DATES[1]: rate2,
                            EFFECTIVE_DATES[2]: rate3
                        }
                    })
    
    # Save last classification
    if current_classification and current_steps:
        classifications[current_classification['normalized_name']] = {
            'name': current_classification['name'],
            'normalized_name': current_classification['normalized_name'],
            'is_manager': current_classification['is_manager'],
            'steps': current_steps
        }
    
    return {
        'contract_id': CONTRACT_ID,
        'effective_dates': EFFECTIVE_DATES,
        'classifications': classifications
    }


def lookup_wage(wages_data: dict, classification: str, hours_worked: int = 0, 
                months_employed: int = 0, effective_date: str = None) -> Optional[dict]:
    """
    Look up the wage rate for a given classification and experience level.
    
    Args:
        wages_data: The full wages data structure
        classification: Job classification (e.g., "all_purpose_clerk")
        hours_worked: Total hours worked (for hour-based progressions)
        months_employed: Total months employed (for month-based progressions)
        effective_date: Which contract year to use (defaults to most recent)
    
    Returns:
        dict with wage info or None if not found
    """
    if effective_date is None:
        effective_date = EFFECTIVE_DATES[-1]  # Most recent
    
    # Normalize classification name
    norm_class = normalize_classification_name(classification)
    
    if norm_class not in wages_data['classifications']:
        # Try partial match
        for key in wages_data['classifications']:
            if norm_class in key or key in norm_class:
                norm_class = key
                break
        else:
            return None
    
    class_data = wages_data['classifications'][norm_class]
    steps = class_data['steps']
    
    if not steps:
        return None
    
    # Find applicable step
    applicable_step = steps[0]  # Default to first step
    
    for step in steps:
        if step['hours_required'] is not None:
            # Hour-based progression
            if hours_worked >= step['hours_required']:
                applicable_step = step
        elif step['months_required'] is not None:
            # Month-based progression
            if months_employed >= step['months_required']:
                applicable_step = step
    
    rate = applicable_step['rates'].get(effective_date)
    
    return {
        'classification': class_data['name'],
        'step': applicable_step['step_name'],
        'rate': rate,
        'effective_date': effective_date,
        'citation': 'Appendix A'
    }


def save_wages(wages_data: dict, output_file: Path) -> None:
    """Save wages data to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(wages_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved wages data to {output_file}")


def main():
    """Main entry point for wage extraction."""
    print(f"Extracting wages from: {CONTRACT_MD_FILE}")
    
    # Read markdown content
    with open(CONTRACT_MD_FILE, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Extract wages
    wages_data = extract_wages(md_content)
    
    # Print summary
    print(f"\nExtracted {len(wages_data['classifications'])} classifications:")
    for name, data in wages_data['classifications'].items():
        print(f"  - {data['name']}: {len(data['steps'])} steps")
    
    # Save to file
    output_file = WAGES_DIR / "wage_tables.json"
    save_wages(wages_data, output_file)
    
    # Test lookup
    print("\n--- Test Lookups ---")
    test_cases = [
        ("all_purpose_clerk", 0, 0),
        ("all_purpose_clerk", 3000, 0),
        ("all_purpose_clerk", 8000, 0),
        ("courtesy_clerk", 0, 0),
        ("courtesy_clerk", 0, 48),
        ("head_clerk", 0, 0),
    ]
    
    for classification, hours, months in test_cases:
        result = lookup_wage(wages_data, classification, hours, months)
        if result:
            print(f"  {classification} ({hours}hrs/{months}mo): ${result['rate']:.2f} ({result['step']})")
        else:
            print(f"  {classification}: Not found")
    
    return wages_data


if __name__ == "__main__":
    main()

