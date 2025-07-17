# ingredient_parser.py
import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def parse_quantity(qty_str: str) -> float:
    """Convert quantity string to grams"""
    qty_str = qty_str.lower().replace(',', '')
    
    # Handle common units
    conversions = {
        'kg': 1000,
        'g': 1,
        'mg': 0.001,
        'l': 1000,
        'ml': 1,
        'cup': 240,
        'cups': 240,
        'tbsp': 15,
        'tablespoon': 15,
        'tsp': 5,
        'teaspoon': 5,
        'piece': 50,  # Default weight for pieces
        'pieces': 50,
        'pc': 50,
        'pcs': 50,
        'slice': 30,   # Default weight for slices
        'slices': 30,
        'whole': 100,  # Default weight for whole items
    }
    
    # Extract numeric value and unit
    match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z]*)', qty_str)
    if not match:
        logger.warning(f"Couldn't parse quantity: {qty_str}")
        return 0.0
        
    value = float(match.group(1))
    unit = match.group(2)
    
    # Handle fractions (e.g., 1/2)
    if '/' in qty_str:
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', qty_str)
        if fraction_match:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            value = numerator / denominator
    
    return value * conversions.get(unit, 1)

def parse_ingredients(ingredient_str: str) -> List[Tuple[str, float]]:
    """Parse ingredients into (item, quantity in grams)"""
    parsed = []
    # More robust splitting that handles various formats
    parts = re.split(r',\s*(?![^()]*\))|\n|\band\b', ingredient_str)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Handle different formats
        if ':' in part:
            # Format: "Ingredient: Quantity"
            item, quantity = part.split(':', 1)
        elif re.search(r'\d', part):
            # Format: "Ingredient Quantity"
            match = re.search(r'(.+?)\s+(\d.*)$', part)
            if match:
                item = match.group(1)
                quantity = match.group(2)
            else:
                # Last resort: try to find quantity at end
                match = re.search(r'(\d+\s*[a-zA-Z]*$)', part)
                if match:
                    quantity = match.group(1)
                    item = part.replace(quantity, '').strip()
                else:
                    logger.warning(f"Skipping unparseable ingredient: {part}")
                    continue
        else:
            # No quantity specified
            item = part
            quantity = "1 piece"  # Default quantity
            
        parsed.append((
            item.strip().lower(), 
            parse_quantity(quantity.strip())
        ))
        
    return parsed