from code.profdb.ProfDB import ProfDB
import pandas as pd

csv_path = "input/List_of_professors_(Gael_labList_incl._SPC).csv"
profdb = ProfDB(csv_path,  2019, 2026) # 2018
profdb.build()