import duckdb
con = duckdb.connect(f'/data/gael/2025-10-08-listProf/output/db_20260106_021111/db.duckdb', read_only=True)
df = con.execute("""
 SELECT sciper,
        title,
        abstract,
 FROM pub
     
          INNER JOIN sciper_pub USING (id_pub)
     WHERE id_infoscience IS NOT NULL 
 """).df()
con.close()
print (df.shape)

df.to_pickle('epfl_pub.pickle')