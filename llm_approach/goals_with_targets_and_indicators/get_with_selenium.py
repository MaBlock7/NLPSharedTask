from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
import json

driver = webdriver.Edge()
url = 'https://en.wikipedia.org/wiki/List_of_Sustainable_Development_Goal_targets_and_indicators'
# Dictionary to store the goals
goals = {}

# Start the browser and open the URL
driver.get(url)



# for a single goal, given its table element, populate the goals dictionary, ingoring the UNSD codes.
def populate_goal(goal: int, table: WebElement):
    rows = table.find_elements(By.TAG_NAME, 'tr')
    
    
    # Initialize an empty list for the current goal
    goals[goal] = {}
    
    # Variable to keep track of the current target text
    current_target_text = ""
    
    # Iterate over the rows to extract targets and indicators
    for row in rows[1:]:  # Skip the header row
        cells = row.find_elements(By.TAG_NAME, 'td')
        
        # Check if this row starts with a target cell
        if len(cells) == 3:
            current_target_text = cells[0].text.strip()  # This cell contains the target text
            indicator_text = cells[1].text.strip()  # This cell contains the indicator text
            # Initialize the list for this target with the first indicator
            goals[goal][current_target_text] = [indicator_text]
        elif len(cells) == 2:
            # This row only has indicators for the current target
            indicator_text = cells[0].text.strip()  # This cell contains the indicator text
            # Add the indicator to the current target's list
            goals[goal][current_target_text].append(indicator_text)


# On the Wiki page there's a table element for each goal. Above we process a single table.
# Since the wikitable class is only sued by those tables, we can use that to get the tables and simply iterate over them.

tables = driver.find_elements(By.CLASS_NAME, 'wikitable')
for goal_number, table in enumerate(tables, start=1):
    print(f"processing goal {goal_number}..")
    populate_goal(goal_number, table)
    print('done.')

fname = 'goals.json'
print(f"saving as {fname}")

with open(fname, 'w') as json_file:
    json.dump(goals, json_file, indent=4)
