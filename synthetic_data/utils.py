import re


def remove_numbered_lists(text):
    # Use regular expression to match numbered lists
    pattern = r'^\d+\.\s+'
    regex = re.compile(pattern, re.MULTILINE)
    # Remove numbered lists from the text
    text = regex.sub('', text)
    return text


def load_attributes(attr_name: str = '') -> dict | list:
    """Returns a dictionary for class-dependent features, and a list for generic features"""
    lst = []
    with open(f"synthetic_data/raw_data/attributes/{attr_name}/{attr_name}.txt", 'r') as f:
        for lines in f:
            lst.append(lines.strip("\n"))
    return lst
