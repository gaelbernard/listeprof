import duckdb
import pandas as pd
import numpy as np
researchers = '''
Nicolai Cramer
Xile Hu
Volkan Cevher
Negar Kiyavash
Dimitri Van de Ville
Fabrizio Carbone
Georg Fantner
Anders Meibom
 Dimitrios Lignos
 Aleksandra Radenovic
Stéphanie Lacour
Erwan MORELLEC
Felix NAEF
Michel BIERLAIRE
Nicolas	Thomä
Christian HEINIS
Françoise Gisou van der Goot Grunberg 
Alexandre PERSAT
Pavan Ramdya
Grégoire Courtine
'''
researchers = [x.strip() for x in researchers.split('\n') if len(x)>1]

duck_db_path = "/data/gael/2025-10-08-listProf/output/db_20251203_170029/db.duckdb"
con = duckdb.connect(duck_db_path, read_only=True)

print (con.execute("""
    SELECT timestamp
    FROM metadata
""").df())
exit()

prof = con.execute("""
    SELECT sciper, email, lastname, firstname, class_acc
    FROM prof
""").df()


researchers_sciper = {}
for r in researchers:
    parts = r.split()
    firstname = parts[0].lower()
    lastname = ' '.join(parts[1:]).lower()

    match = prof[
        (prof['firstname'].str.lower().str.contains(firstname)) &
        (prof['lastname'].str.lower().str.contains(lastname))
        ]

    if len(match) == 1:
        researchers_sciper[r] = int(match.iloc[0]['sciper'])
    elif len(match) > 1:
        print(f"Multiple matches for {r}: {match[['sciper', 'firstname', 'lastname']].values.tolist()}")
        researchers_sciper[r] = int(match.iloc[0]['sciper'])  # Take first match
    else:
        print(f"No match found for {r}")
        researchers_sciper[r] = None

researchers_sciper['Françoise Gisou van der Goot Grunberg'] = 171549

embeddings = con.execute("""
    SELECT sciper, embedding
    FROM pub_embedding
        INNER JOIN pub USING (id_pub)
        INNER JOIN sciper_pub USING (id_pub)
""").df()

# Build a numpy matrix per sciper in a dictionary
prof_embeddings = {}
for sciper, group in embeddings.groupby('sciper'):
    prof_embeddings[sciper] = np.vstack(group['embedding'].values)

# Build a vector per sciper with the avg
prof_avg_embeddings = {
    sciper: emb.mean(axis=0) for sciper, emb in prof_embeddings.items()
}

# iterate the prof we are looking for
output = []
for name, sciper in researchers_sciper.items():

    # Calculate the euclidean distance with each prof_avg_embeddings and store in dict with key sciper
    avg_embeddings = prof_avg_embeddings[sciper]
    avg_distances = {}
    for other_sciper, other_embedding in prof_avg_embeddings.items():
        avg_distances[other_sciper] = float(np.linalg.norm(avg_embeddings - other_embedding))

    # Calculate all the distance between 2 researchers, keep only the avg of the 5 closest distances
    prof_embedding = prof_embeddings[sciper]
    closest_distances = {}
    for other_sciper, other_embeddings in prof_embeddings.items():
        all_dists = []
        for emb in prof_embedding:
            for other_emb in other_embeddings:
                all_dists.append(np.linalg.norm(emb - other_emb))
        all_dists.sort()
        closest_distances[other_sciper] = float(np.mean(all_dists[:3]))

    # Make a dataframe with 2 distances and then transform to rank
    df_distances = pd.DataFrame({
        'prof_name': name,
        'prof_sciper': sciper,
        'sciper': list(avg_distances.keys()),
        'avg_dist': list(avg_distances.values()),
        'closest_dist': [closest_distances[s] for s in avg_distances.keys()]
    })

    df_distances['avg_rank'] = df_distances['avg_dist'].rank()
    df_distances['closest_rank'] = df_distances['closest_dist'].rank()
    df_distances['final_mean_rank'] =  (df_distances['avg_rank'] + df_distances['closest_rank'])/2
    # Sort by closest_rank or combined rank
    df_distances = df_distances.sort_values('final_mean_rank')

    # Add the details of the prof (merge with prof df)
    df_distances = df_distances.merge(prof, on='sciper').head(21)
    df_distances = df_distances[df_distances['prof_sciper']!=df_distances['sciper']]
    output.append(df_distances)

output = pd.concat(output)
output.to_excel("dist_prof.xlsx")


