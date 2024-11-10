#https://cookbook.openai.com/examples/rag_with_graph_db
import os
import pandas as pd

def create_Assistant(file_path):
    # Loading a json dataset from a file
    file_path = 'Actionable_Insight.xlsx'
    
    # Read CSV file
    df = pd.read_excel(file_path, sheet_name="Sheet6")
    
    # Set display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)   # Show all columns
    
    # Print the entire DataFrame
    print(df.head(90))