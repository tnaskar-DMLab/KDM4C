import pandas as pd
from sklearn.model_selection import LeaveOneOut
import os

# Function to perform leave one out splitting and save data
def leave_one_out_split_and_save(data, output_dir):
    loo = LeaveOneOut()

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx_train, idx_test in loo.split(data):
        train_data = data.iloc[idx_train]
        test_data = data.iloc[idx_test]

        # Save train and test data into separate Excel files
        train_file = os.path.join(output_dir, f"train_split_{idx_test[0]}.xlsx")
        test_file = os.path.join(output_dir, f"test_split_{idx_test[0]}.xlsx")
        
        train_data.to_excel(train_file, index=False)
        test_data.to_excel(test_file, index=False)

# Read data from Excel file
excel_file = 'train_set.xlsx'  # Replace 'data.xlsx' with your file name
data = pd.read_excel(excel_file)

# Perform leave one out splitting and save data
output_directory = 'ACA_Dis_data'
leave_one_out_split_and_save(data, output_directory)

