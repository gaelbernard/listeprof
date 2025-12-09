from code.profdb.ProfDB import ProfDB
import pandas as pd

csv_path = "input/List of professors (Gaël_labList incl. SPC).csv"
profdb = ProfDB(csv_path, 2025, 2025) # 2018
profdb.build()