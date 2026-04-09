import pandas as pd

train_df = pd.read_parquet("train-00000-of-00001.parquet")
val_df = pd.read_parquet("validation-00000-of-00001.parquet")
test_df = pd.read_parquet("test-00000-of-00001.parquet")

def get_label(category_path):
    leaf = category_path.split('->')[-1].strip()
    if leaf.startswith('cs.'):
        return 'Computer Science'
    elif leaf.startswith(('astro-ph', 'cond-mat', 'gr-qc', 'hep-', 
                          'math-ph', 'nlin.', 'nucl-', 'physics.', 'quant-ph')):
        return 'Physics'
    elif leaf.startswith('math.'):
        return 'Mathematics'
    elif leaf.startswith('q-bio.'):
        return 'Biology'
    elif leaf.startswith('stat.'):
        return 'Statistics'
    elif leaf.startswith('eess.'):
        return 'Electrical Engineering'
    elif leaf.startswith(('econ.', 'q-fin.')):
        return 'Economics'
    else:
        return None

def filter_articles(df, name=""):
    filtered = []
    for _, row in df.iterrows():
        for cat_path in row['categories']:
            label = get_label(cat_path)
            if label:
                filtered.append({
                    'title': row['title'],
                    'abstract': row['abstract'],
                    'label': label
                })
                break
    return pd.DataFrame(filtered)

train_clean = filter_articles(train_df, "Train")
val_clean = filter_articles(val_df, "Val")
test_clean = filter_articles(test_df, "Test")

train_clean.to_csv("train.csv", index=False)
val_clean.to_csv("val.csv", index=False)
test_clean.to_csv("test.csv", index=False)


