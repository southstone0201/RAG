import pandas as pd


max = 100


def data_preprocessing(file_path):
    
    #choose right column
    data = pd.read_csv(file_path, usecols=[5])
    
    
    
    #to dictionary
    data_dict = data.squeeze().to_dict()
    
    #processing for RAG tokenizer
    ids = [str(key) for key in data_dict if isinstance(data_dict[key],str)]
    values = [data_dict[key] for key in data_dict if isinstance(data_dict[key],str)]
    
    
    processed_data = {
        'id': ids,
        'text': values
    }
    
    return processed_data







    








