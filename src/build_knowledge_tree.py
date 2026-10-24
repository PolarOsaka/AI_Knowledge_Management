import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configuration import EXCEL_PATH, OUTPUT_DIR, OUTPUT_PATH, REQUIRED_COLUMNS, OPTIONAL_COLUMNS
import pandas as pd
from collections import defaultdict
import json

# ============================================================
# 1. LOAD PROJECT REGISTRY
# ============================================================

if not EXCEL_PATH.exists():
    raise FileNotFoundError(
        f"Registry Excel not found: {EXCEL_PATH}"
    )

df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.str.strip()

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Add optional columns if any are absent.
for col in OPTIONAL_COLUMNS:
    if col not in df.columns:
        df[col] = None

# ============================================================
# 2. DATA CLEANING HELPERS
# ============================================================

def clean_value(value):
    """Convert Excel values into JSON-friendly values."""
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return value


def required_text(value, column, row_number):
    value = clean_value(value)

    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Missing '{column}' in Excel row {row_number}"
        )

    return str(value).strip()


def normalize_year(value, row_number):
    value = clean_value(value)

    if value is None:
        raise ValueError(
            f"Missing 'Start Year' in Excel row {row_number}"
        )

    try:
        year_number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid Start Year {value!r} "
            f"in Excel row {row_number}"
        )

    if not year_number.is_integer():
        raise ValueError(
            f"Start Year must be a whole year: {value!r} "
            f"in Excel row {row_number}"
        )

    return str(int(year_number))


# ============================================================
# 4. VALIDATE PROJECT RECORDS
# ============================================================

projects = []
seen_codes = set()

for index, row in df.iterrows():
    # Excel header is row 1, so data begins at row 2.
    excel_row = index + 2

    project_code = required_text(
        row["Project Code"], "Project Code", excel_row
    )

    if project_code in seen_codes:
        raise ValueError(
            f"Duplicate Project Code {project_code!r} "
            f"found at Excel row {excel_row}"
        )

    seen_codes.add(project_code)

    project = {
        "project_code": project_code,
        "project_name": required_text(
            row["Project Name"], "Project Name", excel_row
        ),
        "project_type": required_text(
            row["Project Type"], "Project Type", excel_row
        ),
        "practice_area": required_text(
            row["Practice Area"], "Practice Area", excel_row
        ),
        "start_year": normalize_year(
            row["Start Year"], excel_row
        ),
        "client_non_client_billable": clean_value(
            row["Client/Non-Client Billable"]
        ),
        "client_name": clean_value(row["Client Name"]),
        "sharepoint_folder": clean_value(
            row["Sharepoint Folder"]
        ),
        "status": clean_value(row["Status"]),
    }

    projects.append(project)


# ============================================================
# 5. BUILD HIERARCHICAL KNOWLEDGE TREE
# ============================================================

root = {
    "name": "Dummy Knowledge Base",
    "node_type": "root",
    "children": {},
}


def get_or_create_node(parent, key, name, node_type):
    """Create a child node once and reuse it."""
    if key not in parent["children"]:
        parent["children"][key] = {
            "name": name,
            "node_type": node_type,
            "children": {},
        }

    return parent["children"][key]


for project in projects:
    # Level 1: Project Type
    type_node = get_or_create_node(
        root,
        project["project_type"],
        project["project_type"],
        "project_type",
    )

    # Level 2: Practice Area
    practice_node = get_or_create_node(
        type_node,
        project["practice_area"],
        project["practice_area"],
        "practice_area",
    )

    # Level 3: Start Year
    year_node = get_or_create_node(
        practice_node,
        project["start_year"],
        project["start_year"],
        "year",
    )

    # Level 4: Project
    project_key = project["project_code"]

    year_node["children"][project_key] = {
        "name": project["project_name"],
        "node_type": "project",
        "project_code": project["project_code"],
        "metadata": {
            "project_name": project["project_name"],
            "project_type": project["project_type"],
            "practice_area": project["practice_area"],
            "start_year": project["start_year"],
            "client_non_client_billable":
                project["client_non_client_billable"],
            "client_name": project["client_name"],
            "sharepoint_folder": project["sharepoint_folder"],
            "status": project["status"],
        },
        "children": {},
    }


# ============================================================
# 6. MAKE OUTPUT DETERMINISTIC
# ============================================================

def sort_children(node):
    children = node.get("children", {})

    # Sort consistently for repeatable output.
    node["children"] = dict(
        sorted(children.items(), key=lambda item: item[0].lower())
    )

    for child in node["children"].values():
        sort_children(child)


sort_children(root)


# ============================================================
# 7. EXPORT KNOWLEDGE TREE
# ============================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        root,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 8. REPORT
# ============================================================

project_types = root["children"]
practice_areas = sum(
    len(node["children"].values())
    for node in project_types.values()
)
total_projects = len(projects)

print("Knowledge Tree built successfully!")
print(f"Input file       : {EXCEL_PATH}")
print(f"Output file      : {OUTPUT_PATH}")
print(f"Total projects   : {total_projects}")
print(f"Project types    : {len(project_types)}")
print(f"Practice areas   : {practice_areas}")

print("\nTree overview:")

for type_name, type_node in project_types.items():
    print(f"\n[Project Type] {type_name}")

    for practice_name, practice_node in (
        type_node["children"].items()
    ):
        print(f"  [Practice Area] {practice_name}")

        for year_name, year_node in (
            practice_node["children"].items()
        ):
            count = len(year_node["children"])
            print(
                f"    [Year] {year_name} "
                f"({count} projects)"
            )
