#!/usr/bin/env python3
"""
Complete Project Inventory - Windows Version
Searches YOUR computer for all project files
"""

import os
import sys
from pathlib import Path


def get_size_str(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes > 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes > 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def scan_directory(path, max_depth=3, current_depth=0):
    """Recursively scan directory"""
    items = {'dirs': {}, 'files': []}

    if current_depth >= max_depth:
        return items

    try:
        for item in sorted(os.listdir(path)):
            if item.startswith('.') or item == '__pycache__' or item == '.venv':
                continue

            item_path = os.path.join(path, item)

            try:
                if os.path.isdir(item_path):
                    items['dirs'][item] = scan_directory(item_path, max_depth, current_depth + 1)
                else:
                    size = os.path.getsize(item_path)
                    items['files'].append((item, size, item_path))
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        pass

    return items


def print_tree(items, prefix="", max_files=50):
    """Print directory tree"""
    files = sorted(items['files'], key=lambda x: x[0])

    # Limit file display
    display_files = files[:max_files]

    for i, (name, size, path) in enumerate(display_files):
        is_last_file = (i == len(display_files) - 1) and len(items['dirs']) == 0
        connector = "└── " if is_last_file else "├── "
        print(f"{prefix}{connector}{name} ({get_size_str(size)})")

    if len(files) > max_files:
        print(f"{prefix}    ... and {len(files) - max_files} more files")

    # Print directories
    dirs = sorted(items['dirs'].items())
    for i, (name, subitems) in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1)
        connector = "└── " if is_last_dir else "├── "
        print(f"{prefix}{connector}📁 {name}/")

        extension = "    " if is_last_dir else "│   "
        print_tree(subitems, prefix + extension, max_files)


print("=" * 80)
print("📦 COMPLETE PROJECT INVENTORY - YOUR COMPUTER")
print("=" * 80)
print()

# Get current directory
current_dir = os.getcwd()
print(f"📂 Scanning: {current_dir}")
print()

# Scan current directory
items = scan_directory(current_dir, max_depth=4)


# Count totals
def count_all_files(items):
    total = len(items['files'])
    for subdir in items['dirs'].values():
        total += count_all_files(subdir)
    return total


def sum_all_sizes(items):
    total = sum(f[1] for f in items['files'])
    for subdir in items['dirs'].values():
        total += sum_all_sizes(subdir)
    return total


total_files = count_all_files(items)
total_size = sum_all_sizes(items)

print(f"📊 Summary: {total_files} files, Total size: {get_size_str(total_size)}")
print()

print_tree(items, max_files=100)

# Now find specific files we need
print("\n")
print("=" * 80)
print("🎯 SPECIFIC FILES WE NEED FOR DEPLOYMENT")
print("=" * 80)

needed_files = {
    'Source Code': [
        'streamlit_app.py',
        'config.py',
        'credit_decision_engine.py',
        'interest_rate_calculator.py',
        'data_processing.py',
        'model_evaluation.py',
        'model_validation.py',
        'fairness_testing.py',
        'threshold_analysis.py',
        'visualization.py',
        'business_impact_real.py',
    ],
    'Data Files': [
        'best_pd_model.h5',
        'scaler.pkl',
        'X_test.csv',
        'y_test.csv',
        'pd_test_predictions.csv',
        'all_models_comparison.csv',
        'feature_medians.csv',
    ],
    'Notebooks': [
        '03_pd_model_development.ipynb',
        '04_lgd_ead_models.ipynb',
    ]
}

# Search for all files in current directory tree
all_paths = []
for root, dirs, files in os.walk(current_dir):
    # Skip certain directories
    dirs[:] = [d for d in dirs if d not in ['.venv', '__pycache__', '.git', 'node_modules']]

    for f in files:
        all_paths.append((f, os.path.join(root, f)))

for category, files in needed_files.items():
    print(f"\n{category}:")
    print("-" * 80)
    for filename in files:
        found = [path for name, path in all_paths if name == filename]
        if found:
            try:
                size = os.path.getsize(found[0])
                print(f"  ✅ {filename:40s} {get_size_str(size):>10s}")
                # Show relative path
                rel_path = os.path.relpath(found[0], current_dir)
                print(f"     ./{rel_path}")
            except:
                print(f"  ✅ {filename:40s}")
                rel_path = os.path.relpath(found[0], current_dir)
                print(f"     ./{rel_path}")
        else:
            print(f"  ❌ {filename:40s} NOT FOUND")

print("\n")
print("=" * 80)
print("🔍 CONCLUSION")
print("=" * 80)

source_found = sum(1 for f in needed_files['Source Code']
                   if any(name == f for name, _ in all_paths))
data_found = sum(1 for f in needed_files['Data Files']
                 if any(name == f for name, _ in all_paths))

print(f"\n✅ Source Code: {source_found}/11 files found")
print(f"{'✅' if data_found == 7 else '❌'} Data Files: {data_found}/7 files found")

if source_found == 11 and data_found == 7:
    print("\n🎉 ALL FILES FOUND! Ready to deploy!")
    print("\n📋 Next step: Run the cleanup script to organize for GitHub")
elif source_found == 11 and data_found == 0:
    print("\n⚠️  Source code complete, but data files missing.")
    print("   Data files need to be in data/processed/ directory")
    print("   Check if you have a 'data' folder with 'processed' subfolder")
elif source_found == 0:
    print("\n⚠️  No source files found in current directory!")
    print(f"   Current directory: {current_dir}")
    print("   Make sure you're running this from your credit_risk_final folder")
else:
    print("\n⚠️  Some files are missing. Check the list above.")

print()