from gravityai import gravityai as grav
import pickle  #used for serialization and deserialiaztion of files python objects (converted into bytestrean and stored in a binary file)
import pandas as pd #open source library for data analysis and manipulation 


model = pickle.load(open('./pickle_files/financial_text_classifier.pkl','rb'))
tfidf_vectorizer = pickle.load(open('./pickle_files/financial_text_vectorizer.pkl','rb')) # Term Frequency Inverse Document Frequency
label_encoder = pickle.load(open('./pickle_files/financial_text_encoder.pkl','rb'))


def process(inPath, outPath):
  # read input file
  input_def = pd.read_csv(inPath)
  # vectorize the data into features
  features = tfidf_vectorizer.transform(input_def['body'])
  # predict the classes from the features
  predictions = model.predict(features)
  # convert output LABELS into categories
  input_def['category'] = label_encoder.inverse_transform(predictions)
  
  # save results into CSV (comma seperated values)
  output_def = input_def[['id','category']]
  output_def.to_csv(outPath, index=False)

  grav.wait_for_requests(process)

