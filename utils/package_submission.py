#!/usr/bin/env python3
"""
package_submission.py
=====================
Builds the final submission zip package matching the exact specification
required in student_resource/README.md:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse

def main():
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge submission zip")
    parser.add_argument("--team-name", default="EntityLink_AI", help="Team name for zip filename")
    parser.add_argument("--output-zip", default=None, help="Explicit path for output zip file")
    args = parser.parse_args()

    project_root = os.path.abspath(".")
    zip_name = args.output_zip or f"{args.team_name}_submission.zip"
    zip_path = os.path.join(project_root, zip_name)

    print(f"Creating submission package: {zip_path}")

    files_to_pack = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]

    # Add all files in code/business_entity_resolution/src
    src_dir = os.path.join(project_root, "code", "business_entity_resolution", "src")
    if os.path.isdir(src_dir):
        for root, dirs, files in os.walk(src_dir):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith((".py", ".json", ".txt")):
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, project_root).replace("\\", "/")
                    files_to_pack.append((rel_p, rel_p))

    # Verify presence of required files
    missing = []
    for disk_path, arc_path in files_to_pack:
        if not os.path.exists(disk_path):
            missing.append(disk_path)

    if missing:
        print("ERROR: Missing required files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"Archiving {len(files_to_pack)} items into {zip_name} (using ZIP_DEFLATED)...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for disk_path, arc_path in files_to_pack:
            print(f"  + {arc_path}")
            zf.write(disk_path, arcname=arc_path)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSUCCESS! Package created: {zip_path} ({zip_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
