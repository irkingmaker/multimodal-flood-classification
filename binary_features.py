import json
import pandas as pd

# change this path with your data path
path ='/content/drive/MyDrive/ALDA/Project/Dataset/'

#Load the training dataset using metadata and label data.
train_df = pd.read_csv(path + 'devset_images_gt.csv')
train_df.columns=(['image_id', 'label'])
train_df["image_id"] = train_df["image_id"].astype(str)
train_df = train_df[train_df.columns[[0, 1]]]

json_filename = path + "devset_images_metadata.json"
with open(json_filename) as json_file:
    data = json.load(json_file)

required_columns = ['description', 'user_tags', 'image_id', 'title']
json_df = pd.DataFrame(data['images'], columns = required_columns)
print(json_df.head())
#merge label and json dataframes
train_df = pd.merge(train_df, json_df, how='inner')
train_df.head()


#Load the testing dataset using metadata and label data.
test_df = pd.read_csv(path + 'test_set/testset_images_gt.csv')
test_df.columns=(['image_id', 'label'])
test_df["image_id"] = test_df["image_id"].astype(str)
test_df = test_df[test_df.columns[[0, 1]]]

json_filename = path + "test_set/testset_images_metadata.json"
with open(json_filename) as json_file:
    data = json.load(json_file)

required_columns = ['description', 'user_tags', 'image_id', 'title']
json_df = pd.DataFrame(data['images'], columns = required_columns)
#merge label and json dataframes
test_df = pd.merge(test_df, json_df, how='inner')
test_df.head()



# add three binary features to metadata based on keywords.
def addBinaryFeatures(df):
  descr_flooded = []
  user_tags_flooded = []
  title_flooded = []
  for row in df.itertuples():
    if(df.at[row.Index,'description'] is not None and "flood" in df.at[row.Index,'description']):
      descr_flooded.append(1)
    else:
      descr_flooded.append(0)
    if(df.at[row.Index,'title'] is not None and "flood" in df.at[row.Index,'title']):
      title_flooded.append(1)
    else:
      title_flooded.append(0)
    if(df.at[row.Index,'user_tags'] is not None and "flood" in df.at[row.Index,'user_tags']):
      user_tags_flooded.append(1)
    else:
      user_tags_flooded.append(0)

  df['descr_flooded'] = descr_flooded
  df['title_flooded'] = title_flooded
  df['user_tags_flooded'] = user_tags_flooded
  return df



from sklearn.svm import SVC
train_df = addBinaryFeatures(train_df)
train_x = train_df[['descr_flooded', 'title_flooded', 'user_tags_flooded']]
train_y = train_df[['label']]

#Train the svm model using binary features

# define support vector classifier
svm = SVC(kernel='linear')

svm.fit(train_x, train_y)



from sklearn.metrics import classification_report
#Score the model
#Predictions are calculated on the test data, followed by the report.
test_df = addBinaryFeatures(test_df)
test_x = test_df[['descr_flooded', 'title_flooded', 'user_tags_flooded']]
test_y = test_df[['label']]

#predict response using SVM
svm_y_pred = svm.predict(test_x)

# calculate report for svm model
svm_report = classification_report(test_y, svm_y_pred, target_names=['not flooded', 'flooded'])

print('SVM Model classification report is: \n', svm_report)
