
# Get all Cards from sqlite database 

import sqlite3
import pandas as pd

conn = sqlite3.connect('data/AllPrintings.sqlite')
all_cards = pd.read_sql_query("SELECT * FROM cards", conn)

print(f"Total cards retrieved: {len(all_cards)}")
