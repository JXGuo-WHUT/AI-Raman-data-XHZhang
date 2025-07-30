import os
import pandas as pd


def merge_second_columns(input_folder, output_file):

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        print("")
        return


    combined_data = pd.DataFrame()

    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)


        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 2:
                print(f"File {csv_file} has fewer than two columns, skipping...")
                continue


            combined_data[csv_file] = df.iloc[:, 1]
        except Exception as e:
            print(f"Error reading file {csv_file}:  {e}")


    combined_data.to_csv(output_file, index=False)
    print(f"All second columns have been merged and saved to {output_file}")



input_folder =""
output_file = ""
merge_second_columns(input_folder, output_file)
