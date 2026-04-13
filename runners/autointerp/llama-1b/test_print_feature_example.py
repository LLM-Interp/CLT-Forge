from pathlib import Path
from circuitlab.autointerp.pipeline_new import load_features_parquet
import circuitlab
STORAGE_ROOT = Path(circuitlab.__file__).resolve().parents[2] / "storage"

LAYER = 0
FEATURE_ID = 42
MODEL = "gpt2"
features = load_features_parquet(
    root=STORAGE_ROOT / "autointerp" / MODEL,
    layer=LAYER,
    feature_ids=[FEATURE_ID],
)

if not features:
    print("Feature not found.")
else:
    f = features[0]

    print(f"layer={f['layer']}  feature_id={f['feature_id']}")
    print(f"avg_act={f['average_activation']}")
    print()

    print("description:")
    print(f["description"])
    print()

    print("top_tokens:")
    for t in f["top_activating_tokens"]:
        print(f"{t['token']}  freq={t['frequency']}")

    print()
    print("examples:")
    for ex in f["top_examples"][:5]:
        print(ex)
