import pandas as pd

def insert_col_after(df, d):
    '''
    Pass in dataframe and dictionary in the following format: 
    {
        colname_to_move1 : colname_after1,
        colname_to_move2 : colname_after2,
    }
    Returns dataframe with keys' columns placed after the values' columns.
    '''
    
    df_new = df.copy()
    
    for col_insert, col_after in d.items():
        col = df_new.pop(col_insert)
        i = list(df_new.columns).index(col_after) + 1
        df_new.insert(i, col.name, col)
        
    return df_new