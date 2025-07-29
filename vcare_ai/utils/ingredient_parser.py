# ingredient_parser.py
import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def parse_quantity(qty_str: str) -> float:
    """Convert quantity string to grams"""
    qty_str = qty_str.lower().replace(',', '').strip()
    
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
        'item': 50,    # Generic countable item
        'items': 50,
        '': 50,        # No unit means countable item
    }
    
    # Handle fractions (e.g., 1/2)
    fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', qty_str)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        value = numerator / denominator
        # Remove fraction part to find unit
        unit_str = qty_str.replace(fraction_match.group(0), '').strip()
    else:
        # Extract numeric value and unit
        match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z]*)', qty_str)
        if not match:
            logger.warning(f"Couldn't parse quantity: {qty_str}")
            return 0.0
            
        value = float(match.group(1))
        unit_str = match.group(2).strip()
    
    # Handle plural units
    if unit_str.endswith('s'):
        singular_unit = unit_str[:-1]
        if singular_unit in conversions:
            unit_str = singular_unit
    
    # Handle countable items
    if not unit_str and value.is_integer():
        unit_str = 'piece'
    
    return value * conversions.get(unit_str, 1)

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
        else:
            # Try to find quantity at beginning
            begin_match = re.match(r'^\s*([\d/\.]+\s*[a-zA-Z]*)\s+(.+)', part)
            if begin_match:
                quantity = begin_match.group(1)
                item = begin_match.group(2)
            else:
                # Try to find quantity at end
                end_match = re.search(r'(.+?)\s+([\d/\.]+\s*[a-zA-Z]*)$', part)
                if end_match:
                    item = end_match.group(1)
                    quantity = end_match.group(2)
                else:
                    # Try to extract any quantity
                    qty_match = re.search(r'([\d/\.]+\s*[a-zA-Z]*)', part)
                    if qty_match:
                        quantity = qty_match.group(1)
                        item = part.replace(quantity, '').strip()
                    else:
                        # Check if it's a countable item without explicit quantity
                        if re.match(r'^\d+\s+\w+', part):
                            # Format: "2 eggs" without quantity marker
                            qty_match = re.match(r'^(\d+)\s+(.+)', part)
                            if qty_match:
                                quantity = qty_match.group(1)
                                item = qty_match.group(2)
                            else:
                                logger.warning(f"Skipping unparseable ingredient: {part}")
                                continue
                        else:
                            logger.warning(f"Skipping unparseable ingredient: {part}")
                            continue
        
        # Handle countable items without explicit unit
        if not re.search(r'[a-z]', quantity) and re.match(r'^\d+$', quantity):
            quantity += " piece"
            
        parsed.append((
            item.strip().lower(), 
            parse_quantity(quantity.strip())
        ))
        
    return parsed