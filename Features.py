import json
import functools

countriesCities = json.load(open('countries.json'))
all_countries = set([c.lower() for c in countriesCities.keys()])
all_countries.add("u.s")
for country in set(all_countries):
    all_countries.add(str.replace(country, ' ', '-'))

all_cities = set(functools.reduce(lambda a, b: a + [c.lower() for c in b], countriesCities.values(), []))

for city in set(all_cities):
    all_cities.add(str.replace(city, ' ', '-'))

def forward_tag(chunk, sentence):
    id_first_word_in_chunk = chunk["lastWordIndex"]
    if id_first_word_in_chunk < len(sentence):
        return sentence[id_first_word_in_chunk]['pos']
    return "END"
def next_word(chunk, sentence):
    id_last_word_in_chunk = chunk["lastWordIndex"]
    if id_last_word_in_chunk < len(sentence):
        return sentence[id_last_word_in_chunk]['word']
    return "START"
def previous_word(chunk, sentence):
    id_first_word_in_chunk = chunk["firstWordIndex"]
    if id_first_word_in_chunk - 2 >= 0:
        return sentence[id_first_word_in_chunk - 2]['word']
    return "START"

def BOW(first_chunk, second_chunk, sentence):
    first = first_chunk if first_chunk["firstWordIndex"] < second_chunk["firstWordIndex"] else second_chunk
    second = second_chunk if first_chunk["firstWordIndex"] < second_chunk["firstWordIndex"] else second_chunk
    between_words = sentence[first["lastWordIndex"]:second["firstWordIndex"] - 1]
    return ["BagOfWords%s" % word["lemma"] for word in between_words]

def find_dependency_route(chunk, sentence):    
    firstWord = sentence[chunk["firstWordIndex"]]
    parent = firstWord['parent']
    current_id = firstWord['id']
    while chunk["firstWordIndex"] <= parent - 1 <= chunk["lastWordIndex"]:
        current_id = parent
        parent = sentence[parent - 1]['parent']
    path = [current_id-1]#############################################################
    while True:
        path.append(parent - 1)
        if parent == 0:
            break
        parent = sentence[parent - 1]['parent']
    return path
def find_dependency_routes(first_chunk, second_chunk, sentence):
    first_route = find_dependency_route(first_chunk, sentence)
    second_route = find_dependency_route(second_chunk, sentence)
    
    overlapping = -1
    while overlapping > -len(first_route) and overlapping > -len(second_route) and first_route[overlapping] == \
            second_route[overlapping]:
        overlapping -= 1
    if overlapping == -1:
        return first_route, second_route
    
    return first_route[0:overlapping + 1], second_route[0:overlapping + 1]
def dependency_tags(first_chunk, second_chunk, sentence):
    first, second = find_dependency_routes(first_chunk, second_chunk, sentence)    
    #print(first_chunk)
    #print(second_chunk)
    #print("first")
    #print(list(first))
    #print("second")
    #print(list(reversed(second)))
    graph = first + list(reversed(second))
    #if first_chunk["id"] == "sent2203":
    #    print(graph)
    all_dependency_tags = []
    i = 0
    while graph[i] != graph[i + 1]:
        all_dependency_tags.append("dependTag%s(%s)" % (1, sentence[graph[i]]["tag"]))
        i += 1
        if i + 1 >= len(graph):
            return all_dependency_tags
    i += 1
    all_dependency_tags.append("dependTag%s(%s)" % (1, sentence[graph[i]]["tag"]))
    i += 1
    while i < len(graph):
        #print("###################################################################")
        #print("i: " + str(i))      
        #print("graph[i]: " + str(graph[i]))
        #print(graph)
        #print(sentence)
        all_dependency_tags.append("dependTag%s(%s)" % (1, sentence[graph[i]]["tag"]))
        i += 1
    return all_dependency_tags

def dependency_words(first_chunk, second_chunk, sentence):
    first, second = find_dependency_routes(first_chunk, second_chunk, sentence)
    graph = first + list(reversed(second))
    all_dependency_tags = []
    i = 0
    while graph[i] != graph[i + 1]:
        all_dependency_tags.append("dependWord%s(%s)" % (1, sentence[graph[i]]["lemma"]))
        i += 1
        if i + 1 >= len(graph):
            return all_dependency_tags
    i += 1
    all_dependency_tags.append("dependWord%s(%s)" % (1, sentence[graph[i]]["lemma"]))
    i += 1
    while i < len(graph):
        all_dependency_tags.append("dependWord%s(%s)" % (1, sentence[graph[i]]["lemma"]))
        i += 1
    return all_dependency_tags

def dependency_types(first_chunk, second_chunk, sentence):
    first, second = find_dependency_routes(first_chunk, second_chunk, sentence)
    graph = first + list(reversed(second))
    all_dependency_tags = []
    i = 0
    while graph[i] != graph[i + 1]:
        all_dependency_tags.append("dependType%s(%s)" % (1, sentence[graph[i]]["dependency"]))
        i += 1
        if i + 1 >= len(graph):
            return all_dependency_tags
    i += 1
    all_dependency_tags.append("dependType%s(%s)" % (1, sentence[graph[i]]["dependency"]))
    i += 1
    while i < len(graph):
        all_dependency_tags.append("dependType%s(%s)" % (1, sentence[graph[i]]["dependency"]))
        i += 1
    return all_dependency_tags

class FeaturesBuilder:
    def __init__(self):
        pass

    def build_features(self, first_chunk, second_chunk, sentence):
        
        return ["type%s(%s)" % (1, first_chunk["entType"]),
                "type%s(%s)" % (2, second_chunk["entType"]),
                "chunkWords(%s)" % first_chunk["text"],
                "chunkWords(%s)" % second_chunk["text"],
                "typetype(%s)" % (first_chunk["entType"] + second_chunk["entType"]),
                "location_ind%s(%s)" % (1, "T" if first_chunk["text"].lower() in all_countries or first_chunk["text"].lower() in all_cities else "F"),
                "location_ind%s(%s)" % (2, "T" if second_chunk["text"].lower() in all_countries or second_chunk["text"].lower() in all_cities else "F"),
                "beforeChunk%s(%s)" % (1, previous_word(first_chunk, sentence)),
                "afterChunk%s(%s)" % (2, next_word(second_chunk, sentence)),
                "forwardTag%s(%s)" % (1, forward_tag(first_chunk, sentence)),
                "headTag%s(%s)" % (1, first_chunk["headWordTag"]),
                "headTag%s(%s)" % (2, second_chunk["headWordTag"])] + BOW(first_chunk, second_chunk, sentence) + dependency_tags(first_chunk, second_chunk, sentence) + dependency_words(first_chunk, second_chunk, sentence) + dependency_types(first_chunk, second_chunk, sentence)
        

class FB:
    ALL = [FeaturesBuilder()]
    


