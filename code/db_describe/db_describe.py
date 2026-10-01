"""
DuckDB Schema Explorer
======================
Prints a full description of every table in a DuckDB database:
  - Table name, row count
  - Columns with type, nullability, and sample values

Usage:
    python describe_schema.py                          # uses default path
    python describe_schema.py path/to/db.duckdb        # custom path
"""

import duckdb

# The diagram is here: https://dbdiagram.io/d/699ec3c8bd82f5fce2c6f8d3

def describe_schema(db_path: str = "../../output/latest/db.duckdb"):
    con = duckdb.connect(db_path, read_only=True)

    tables = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()

    print(f"Database: {db_path}")
    print(f"Tables:   {len(tables)}\n")
    print("=" * 80)

    for (table_name,) in tables:
        row_count = con.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
        columns = con.execute(f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}' AND table_schema = 'main'
            ORDER BY ordinal_position
        """).fetchall()

        print(f"\n📋 {table_name}  ({row_count:,} rows, {len(columns)} columns)")
        print("-" * 80)
        print(f"  {'Column':<35} {'Type':<25} {'Nullable'}")
        print(f"  {'------':<35} {'----':<25} {'--------'}")

        for col_name, col_type, nullable in columns:
            null_str = "YES" if nullable == "YES" else "NO"
            print(f"  {col_name:<35} {col_type:<25} {null_str}")

        # Show 3 sample rows
        try:
            sample = con.execute(f"SELECT * FROM \"{table_name}\" LIMIT 3").fetchall()
            col_names = [c[0] for c in columns]
            if sample:
                print(f"\n  Sample rows:")
                for row in sample:
                    pairs = ", ".join(
                        f"{c}={repr(v)[:60]}" for c, v in zip(col_names, row)
                    )
                    print(f"    {{ {pairs} }}")
        except Exception:
            pass

        print()

    con.close()


if __name__ == "__main__":
    describe_schema()