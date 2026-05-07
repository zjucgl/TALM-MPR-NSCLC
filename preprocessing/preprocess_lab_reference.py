import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_VALID_SPECIMEN_TYPES = [
    "血液",
    "静脉血",
    "血清",
    "血浆",
    "组织",
    "组织穿刺物",
    "肺泡灌洗液",
    "支气管灌洗液",
    "胸水",
    "胸腹水",
    "穿刺液",
    "痰液",
    "痰（导管）",
]

REQUIRED_COLUMNS = {
    "reg_no",
    "report_no",
    "item_name",
    "report_time",
    "result_data",
    "spec_type",
}


def clean_value(value):
    if pd.isna(value) or value == "":
        return np.nan

    text = str(value).strip()
    if "阴" in text or text == "-":
        return 0.0

    if "阳" in text or "+" in text:
        match = re.search(r"(\d+)\+", text)
        return float(match.group(1)) if match else 1.0

    text = text.replace("<", "").replace(">", "").replace("=", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def validate_columns(df):
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def preprocess_lab_table(df, missing_threshold=0.7, valid_specimen_types=None, standardize=True):
    validate_columns(df)
    if valid_specimen_types is None:
        valid_specimen_types = DEFAULT_VALID_SPECIMEN_TYPES

    df = df.copy()
    df["item_name"] = df["item_name"].str.replace("*", "", regex=False).str.strip()
    df["parsed_value"] = df["result_data"].apply(clean_value)
    df = df.dropna(subset=["parsed_value"]).copy()

    df = df.sort_values("report_time")
    df = df.drop_duplicates(
        subset=["reg_no", "report_no", "item_name", "report_time"],
        keep="last",
    )

    df["date"] = pd.to_datetime(df["report_time"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(
        subset=["reg_no", "item_name", "date"],
        keep="last",
    )

    df = df[df["spec_type"].isin(valid_specimen_types)].copy()
    wide_df = df.pivot_table(
        index=["reg_no", "date"],
        columns="item_name",
        values="parsed_value",
        aggfunc="last",
    )

    missing_rates = wide_df.isnull().mean()
    kept_features = missing_rates[missing_rates <= missing_threshold].index
    wide_df = wide_df[kept_features]

    variances = wide_df.var()
    kept_by_variance = variances[variances > 0].index
    wide_df = wide_df[kept_by_variance]

    wide_df = wide_df.sort_index(level=["reg_no", "date"])
    for column in wide_df.columns:
        wide_df[column] = wide_df.groupby(level="reg_no")[column].ffill()

    final_df = wide_df.fillna(0.0)
    if not standardize:
        return final_df, final_df.columns.tolist()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(final_df.values)
    scaled_df = pd.DataFrame(
        scaled_values,
        index=final_df.index,
        columns=final_df.columns,
    )
    return scaled_df, scaled_df.columns.tolist()


def generate_model_input(final_df):
    patient_lab_dict = {}

    for (reg_no, date), row in final_df.iterrows():
        reg_no = str(reg_no)
        patient_lab_dict.setdefault(
            reg_no,
            {
                "reg_no": reg_no,
                "time_steps": [],
            },
        )
        patient_lab_dict[reg_no]["time_steps"].append(
            {
                "date": date,
                "has_lab": True,
                "lab_features": row.values.tolist(),
            }
        )

    for record in patient_lab_dict.values():
        record["time_steps"].sort(key=lambda item: item["date"])

    return list(patient_lab_dict.values())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    schema_path = Path(args.schema_output)

    raw_df = pd.read_json(input_path)
    final_df, feature_schema = preprocess_lab_table(
        raw_df,
        missing_threshold=args.missing_threshold,
        standardize=not args.no_standardize,
    )
    model_input = generate_model_input(final_df)

    write_json(output_path, model_input)
    write_json(schema_path, feature_schema)

    print(f"Input rows: {len(raw_df)}")
    print(f"Output patients/registrations: {len(model_input)}")
    print(f"Feature dimension: {len(feature_schema)}")
    print(f"Saved lab JSON: {output_path}")
    print(f"Saved feature schema: {schema_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reference pipeline for converting raw inspection records into lab time-series features."
    )
    parser.add_argument("--input", default="inspection_parse.json", help="Raw inspection JSON exported from SQL.")
    parser.add_argument("--output", default="dataset/lab.json", help="Output lab JSON used by the training dataset.")
    parser.add_argument(
        "--schema-output",
        default="outputs/lab_feature_schema.json",
        help="Output JSON containing the final lab feature order.",
    )
    parser.add_argument("--missing-threshold", type=float, default=0.7, help="Drop features above this missing rate.")
    parser.add_argument("--no-standardize", action="store_true", help="Disable global Z-score standardization.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
