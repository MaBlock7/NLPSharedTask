import re


def remove_numbered_lists(text):
    # Use regular expression to match numbered lists
    pattern = r'^\d+\.\s+'
    regex = re.compile(pattern, re.MULTILINE)
    # Remove numbered lists from the text
    text = regex.sub('', text)
    return text


def load_attributes(attr_name: str = '', classes: list = None) -> dict | list:
    """Returns a dictionary for class-dependent features, and a list for generic features"""

    attr_names = [
        'subtopics_filter', 'subtopics', 'similar', 'technique'
    ]

    if attr_name in attr_names:
        assert classes is not None
        return_dict = {} 
        for c in classes:
            return_dict[c] = []
            with open(f"synthetic_data/raw_data/attributes/{attr_name}/{c}.jsonl", 'r') as f:
                for lines in f:
                    clean_text = remove_numbered_lists(lines.lstrip('0123456789.').lstrip('-()').strip("\"\'\n").strip())
                    if clean_text != "":
                        return_dict[c].append(clean_text)
            if attr_name not in "similar":
                return_dict[c].append(f"Others {attr_name} for {c}.")
        return return_dict

    else:
        lst = []
        with open(f"synthetic_data/raw_data/attributes/{attr_name}.txt", 'r') as f:
            for lines in f:
                lst.append(lines.strip("\n"))
        return lst
