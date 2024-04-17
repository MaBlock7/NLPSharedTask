import argparse


def load_env_variable(env_file_path='essentials/auth/.env', variable_name='all'):
    """
    Load a specific variable or all variables from a .env file.

    :param env_file_path: Path to the .env file
    :param variable_name: The name of the specific variable to load or 'all' to load everything
    :return: The value of the specified variable, or a dictionary of all variables if 'all' is specified
    """
    env_variables = {}
    try:
        with open(env_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                key, value = key.strip(), value.strip()
                env_variables[key] = value
                # If the specific variable is found, return its value immediately
                if variable_name != 'all' and key == variable_name:
                    return value
    except FileNotFoundError:
        print(f'Error: The file {env_file_path} was not found.')
        return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

    # If 'all' was specified, return the dictionary of all variables
    if variable_name == 'all':
        return env_variables
    else:
        # If the specific variable was not found, return None
        return env_variables.get(variable_name, None)
