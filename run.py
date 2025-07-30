from main import *
from config import *

if __name__ == '__main__':
    data_path = ""
    model_path = ""
    if args.task == 'classify':
        KFold_CV(data_path, model_path)
        
    elif args.task == 'predict':
        predict_data_path = ""
        model_path1 = ""
        model_path2 = ""
        
        predA = predict(model_path1, predict_data_path, '')
        predB = predict(model_path2, predict_data_path, '')
        print(f"{predA},{predB}")

        
