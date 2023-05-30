import json


def save_json(path, var):
    json_object = json.dumps(var, indent=4)
    with open(path, "w") as outfile:
        outfile.write(json_object)


def load_json(path):
    with open(path, 'r') as openfile:
        var = json.load(openfile)
    return var
