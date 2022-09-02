import json

#Function reads a json file
def read_json_file(path):
    
    f = open(path)
    data = json.load(f)
    f.close()
    
    return data

    
#Function returns a dataframe with the text of the pairs
def get_json_document_pair(data_path, pair_id):
    
    list_ids = pair_id.split('_')
    doc1_id = list_ids[0]
    doc2_id = list_ids[1]

    doc1_path = data_path + doc1_id[-2:] + '/' + doc1_id + '.json' 
    doc2_path = data_path + doc2_id[-2:] + '/' + doc2_id + '.json' 

    doc1_json = read_json_file(doc1_path)
    doc2_json = read_json_file(doc2_path)

    return (doc1_json, doc2_json)


