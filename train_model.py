import sys
import codecs
import spacy
from sklearn import svm
import numpy
from scipy import sparse
from Features import FB
import pickle

#import en_core_web_sm
ANNOT_TAGS = {'OrgBased_In', 'Live_In', 'Kill', 'Located_In', 'Work_For'}
#ANNOT_TAGS = {'Live_In'}
nlp = spacy.load('en_core_web_sm')
#nlp = en_core_web_sm.load()

class Model(object):
    def __init__(self,corpus_data):
        self.features = {}
        data_features, data_tags = self.getFeatures(corpus_data.processed_data, True)
        self.lSVC = svm.LinearSVC()
        self.lSVC.fit(data_features, data_tags)
    
    #Get all features from data based on definitions in Features.py file
    def getFeatures(self, corpus_data, trainBool):
        print("START: get features")
        tags = []
        features = []
        for ((ar1, ar2, s), t) in corpus_data:
            curr_features = []
            for f_b in FB.ALL:
                for f in f_b.build_features(ar1, ar2, s):
                    if f not in self.features and trainBool:
                        self.features[f] = len(self.features)
                    if f in self.features:
                        curr_features.append(self.features[f])

            features.append(curr_features)
            tags.append(t)
        
        sparse_lst = []
        features_len = len(self.features)
        for dense_feature in features:
            sprs = numpy.zeros(features_len)
            for i in dense_feature:
                sprs[i] = 1
            sparse_lst.append(sprs)
            
        return sparse.csr_matrix(sparse_lst), numpy.array(tags)
    
    #run prediction on data based on model
    def predict(self, data):
        features, _ = self.getFeatures(data, False)
        return self.lSVC.predict(features)

def add_dot(s):
    """
    if len(s) >= 2 and s[-1] == "." and s[-2] != " ":
        return s + " ."        
    else:
        return s
    """
    return s
class ModelData(object):
    def __init__(self, corpus_file, annotations_file = None):
        self.filters = []
        self.entities = []
        
        #read the corpus file
        print("START: read corpus")
        self.corpus = {}
        for line in codecs.open(corpus_file, encoding="utf8"):
            sent_num, sent = line.strip().split("\t")
            sent = add_dot(sent)
            sent = sent.replace("-LRB-", "(").replace("-RRB-", ")")
            self.corpus[sent_num] = sent
        
        #read the annotations file
        print("START: read annotations")
        self.annotations = {}
        if annotations_file:
            for annotation in set(filter(lambda l: l != "", open(annotations_file).read().split("\n"))):
                ID, ARG1, LINK, ARG2, OTHER = annotation.split("\t")
                ARG1 = ARG1.rstrip(".")#add_dot(ARG1)#.rstrip(".")
                ARG2 = ARG2.rstrip(".")#add_dot(ARG2)#.rstrip(".")
                if ID not in self.annotations:
                    self.annotations[ID] = []
                if LINK in ANNOT_TAGS:
                    self.annotations[ID].append((LINK, ARG1, ARG2))
        
        self.processed_data = self.processData()
        
    #create data based on spacy
    def processData(self):
        print("START: process data")
        processed_data = []
        #parse each sentence from the corpus
        for id, s in self.corpus.items():
            nlp_s = nlp(s)
            s_data = []
            full_s = ""
            counter = 0
            s_dict = {}
            
            for i, w in enumerate(nlp_s):
                h_id = w.head.i + 1  
                if w == w.head:  
                    assert (w.dep_ == "ROOT"), w.dep_
                    h_id = 0  

                s_data.append({
                    "id": w.i + 1,
                    "word": w.text,
                    "lemma": w.lemma_,
                    "pos": w.pos_,
                    "tag": w.tag_,
                    "parent": h_id,
                    "dependency": w.dep_,
                    "bio": w.ent_iob_,
                    "ner": w.ent_type_
                })
                full_s += " " + w.text
                s_dict[counter + 1] = i
                counter += 1 + len(w.text)

            entities = {}
            for e in nlp_s.ents:
                stripped = e.text.rstrip(".")
                #stripped = e.text
                entities[stripped] = {
                    "text": stripped,
                    "originalText": e.text,
                    "entType": e.root.ent_type_,
                    "rootText": e.root.text,
                    "rootDep": e.root.dep_,
                    "rootHead": e.root.head.text,
                    "id": id
                }

            for c in nlp_s.noun_chunks:
                stripped = c.text.rstrip(".")
                #stripped = c.text
                if stripped not in entities:
                    entities[stripped] = {
                        "text": stripped,
                        "originalText": c.text,
                        "entType": u'UNKNOWN',
                        "rootText": c.root.text,
                        "rootDep": c.root.dep_,
                        "rootHead": c.root.head.text,
                        "id": id
                    }

            for e in entities.values():
                #find first sentence index
                first_s_i = full_s.find(e["originalText"])
                e["firstWordIndex"] = s_dict[first_s_i] if first_s_i in s_dict else 0

                last_s_i = first_s_i + len(e["originalText"]) + 1
                e["lastWordIndex"] = s_dict[last_s_i] - 1 if last_s_i in s_dict else 0

                dep_i = full_s.find(e["rootDep"])
                e["depWordIndex"] = s_dict[dep_i] if dep_i in s_dict else 0

                head_i = full_s.find(e["rootHead"])
                headWord_i = s_dict[head_i] if head_i in s_dict else 0
                e["headWordTag"] = s_data[headWord_i]["tag"]
                

            self.entities = entities.values()

            for ne1 in self.entities:
                for ne2 in self.entities:
                    if ne1["text"] != ne2["text"]:
                        arg1 = ne1
                        arg2 = ne2

                        relevant_data = s_data  

                        boolFlag = False
                        if id in self.annotations:
                            for LINK, ARG1, ARG2 in self.annotations[id]:
                                if arg1["text"] == ARG1 and arg2["text"] == ARG2:
                                    if LINK in ANNOT_TAGS:
                                        processed_data.append(((arg1, arg2, relevant_data), LINK))
                                        boolFlag = True
                        if not boolFlag:
                            processed_data.append(((arg1, arg2, relevant_data), "NoConn"))
        return processed_data
        
#check if input file in correct format
def check_file(file_name):
    file = open(file_name,"r")
    line = file.readline()
    if line[0] == "#":
        print("input corpus in wrong format. please provide .txt file")
        return True
    else:
        return False

if __name__ == "__main__":
    corpus_file = sys.argv[1]
    annotation_file = sys.argv[2]
    #corpus_file = "./data/Corpus.TRAIN.txt" #"./data/Corpus.TRAIN.PROCESSED"
    #annotation_file = "./data/TRAIN.ANNOTATIONS"
    if not check_file(corpus_file):
        corpus_data = ModelData(corpus_file,annotation_file)
        m = Model(corpus_data)
        print("Number of Features:", len(m.features))
        pickle.dump(m, open("model", "wb"))
