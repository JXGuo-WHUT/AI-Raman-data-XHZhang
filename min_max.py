import pandas as pd
import numpy as np


input_file =""
output_file =""



data = pd.read_csv(input_file, header=None)



def minmax_normalize(row):
    min_val = np.min(row)
    max_val = np.max(row)
    if max_val - min_val == 0:
        return row
    return (row - min_val) / (max_val - min_val)



normalized_data = data.apply(minmax_normalize, axis=1)


normalized_data.to_csv(output_file, index=False, header=False)

print(f"Normalization complete; results saved to {output_file}")
