import pandas as pd

# -------------
# Enter File name of csv and json
# current code will replace the csv
csvFileName = 'chat.csv'
jsonFileName = 'new_json.json'

# -------------
# Enter Heading Variables from csv dataset
col1 = 'Unnamed: 0.1'
col2 = 'Unnamed: 0'
col3 = 'idx'
col4 = 'content'
col5 = 'true_label'
col6 = 'pred_id'

# Enter column location of is_adversarial
location_is_adversarial = 1

# Enter column variables to be removed
remove = [col2, col3, 'No']
# -------------

# Creates dictionary with original column names
# Only change if number of columns increases
column = {col1: col1, col2: col2, col3:col3, col4:col4, col5: col5, col6: col6}

# -------------



# fixes heading names to correct names in csv for advglue datasets
def fix_csv_heading(csvFilePath):
    # reading the csv file
    df = pd.read_csv(csvFilePath)

    # rename columns to be readable for our use case
   # df.rename(columns={'Unnamed: 0': 'Unnamed: 0','idx':'idx','content': 'content','true_label': 'true_label','pred_id':'pred_id'}, inplace=True)
    df.rename(columns=column, inplace=True)
    df.rename(columns={col1: 'No',col2: col2 ,col3: col3, col4: 'question', col5: col5, col6: col6}, inplace=True)

    # remove unnecessary columns
    df.drop(remove, axis=1, inplace=True)

    # add is_adversarial as a column and set each value to 1
    is_adversarial = []
    for n in range(df.shape[0]):
        is_adversarial.append(1)

    df.insert(location_is_adversarial, 'is_adversarial',is_adversarial)

    # writing into the file
    df.to_csv(csvFilePath, index=False, header=True)


fix_csv_heading(csvFileName)


# converts modified csv to json
df = pd.read_csv(csvFileName)
df.to_json(jsonFileName, indent=1, orient='records')
