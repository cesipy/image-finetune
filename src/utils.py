

def get_prompts(path):
    with open(path, "r") as f:
        content = f.read()

    prompts = content.replace('",', '').replace('"', '').splitlines()
    return prompts


print(get_prompts("prompts.txt"))