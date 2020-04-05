#All programs and files need to be located in the same directory(a model is created and the extract.py is based on it)
#The model is created based on a text file 

Step1 - creating the model for the prediction:
	python train_model.py train_file_path.txt annotations_file_path 
	
	example: python train_model.py ./data/Corpus.TRAIN.txt ./data/TRAIN.ANNOTATIONS 

Step2 - extracting relations
	python extract.py input_file_path.txt prediction_output
	
	example: python extract.py ./data/Corpus.DEV.txt prediction_output

Step3 - evaluating the output file
	python eval.py gold_file_path output_file
	
	example: python eval.py ./data/DEV.ANNOTATIONS prediction_output