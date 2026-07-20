import pandas as pd


df = pd.read_csv('gesture_data.csv', header=None)


label_to_remove = "NEED"  


df_cleaned = df[df[0] != label_to_remove]


df_cleaned.to_csv('gesture_data.csv', index=False, header=False)

print(f"Removed all rows for gesture: '{label_to_remove}'")