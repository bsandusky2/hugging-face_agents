from pybaseball import statcast
from pathlib import Path

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_date = "2025-04-01"
end_date = "2025-04-30"

print(f"Downloading Statcast: {start_date} → {end_date}")

df = statcast(start_date, end_date)

print(f"Downloaded {len(df):,} pitches")
print(f"Columns: {len(df.columns)}")

output_file = OUTPUT_DIR / "statcast_2025_04.parquet"

df.to_parquet(output_file, index=False)

print(f"Saved to {output_file}")