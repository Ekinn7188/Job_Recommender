import polars as pl
from gensim.models import Word2Vec

DATASET_DIR = "dataset"
TRAIN_CSV = f"{DATASET_DIR}/train.csv"
TEST_CSV = f"{DATASET_DIR}/test.csv"

print("Loading CSVs...")
train_df = pl.read_csv(TRAIN_CSV)
test_df = pl.read_csv(TEST_CSV)

resume_col = "resume_text"
job_col = "job_description_text"

texts = train_df[resume_col].cast(str).to_list() + train_df[job_col].cast(str).to_list()
texts += test_df[resume_col].cast(str).to_list() + test_df[job_col].cast(str).to_list()

sentences = [t.lower().split() for t in texts]

print(f"Training Word2Vec on {len(sentences)} sentences...")
model = Word2Vec(
    sentences=sentences,
    vector_size=100, # 100-dim, good for baseline
    window=5,
    min_count=2, # ignore super rare words
    workers=4
)

out_path = f"{DATASET_DIR}/w2v.model"
model.save(out_path)
print(f"Saved Word2Vec model to {out_path}")
