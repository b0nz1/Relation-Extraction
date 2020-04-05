import sys
import pickle
from sklearn import metrics
from train_model import check_file,ModelData,Model

def output_to_file(output_file, predicted,processed_data):
    lines = []
    for i, prediction in enumerate(predicted):
            if prediction == "NoConn":
                continue
            arg1, arg2, _ = processed_data[i][0]
            lines.append('%s\t%s\t%s\t%s\t' % (arg1['id'], arg1['text'], prediction, arg2['text']))
            
    with open(output_file, 'w') as out:
        out.write('\n'.join(lines))

if __name__ == "__main__":
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    #data_file = "./data/Corpus.TRAIN.txt"
    #output_file = "pred_train.txt"
    check_file(data_file)
    if not check_file(data_file):
        dataObj = ModelData(data_file)
        model = pickle.load(open("model", "rb"))
        pred = model.predict(dataObj.processed_data)
        print("Positive predictions: ", sum([0 if p == "NoConn" else 1 for p in pred]))
        output_to_file(output_file, pred,dataObj.processed_data)
