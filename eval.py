import sys

def read_annotations_file(annotation_file):
    with open(annotation_file) as file:
        conns = {}
        relations = set()
        for line in file.readlines():
            splt = line.split('\t')
            id = splt[0]
            chunk1 = splt[1]
            conn = splt[2]
            chunk2 = splt[3]
            relations.add(conn)
            if conn not in conns:
                conns[conn] = set()
            conns[conn].add((id, chunk1.rstrip('.'), chunk2.rstrip('.')))
        return conns, relations

if __name__ == "__main__":
    gold_file = sys.argv[1]
    pred_file = sys.argv[2]
    #gold_file = "./data/TRAIN.ANNOTATIONS"#sys.argv[1]
    #pred_file = "pred_train.txt"#sys.argv[2]
    
    
    gold_annotations, gold_relations = read_annotations_file(gold_file)
    pred_annotations, pred_relations = read_annotations_file(pred_file)
    
    for relation in gold_relations.intersection(pred_relations):
        correct = gold_annotations[relation].intersection(pred_annotations[relation])
        precision = len(correct) / float(len(pred_annotations[relation]))
        recall = len(correct) / float(len(gold_annotations[relation]))
        f1 = 2 * precision * recall / float(precision + recall)
        print(relation + "\tPrecision: " + str(precision) + "\tRecall: " + str(recall) + "\tF1: " + str(f1))
        
        
    pred = pred_annotations["Live_In"]
    gold = gold_annotations["Live_In"]
    
    """
    #print PRECISION errors
    print("PRECISION MISTAKE")
    for p in pred:
        if p not in gold:
            print(p)
    #print RECALL errors
    print("RECALL MISTAKE")
    for g in gold:
        if g not in pred:
            print(g)    
    """
