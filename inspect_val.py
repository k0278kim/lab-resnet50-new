import os

val_dir = "../tiny-imagenet-200/val"
print(f"Inspecting '{val_dir}'...")

if not os.path.exists(val_dir):
    print("Directory does not exist!")
    exit()

subdirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
files = [f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f))]

print(f"Subdirectories: {len(subdirs)}")
print(f"Files: {len(files)}")

if len(subdirs) > 0:
    print(f"First 5 subdirs: {subdirs[:5]}")
    # Check inside a subdir
    first_subdir = os.path.join(val_dir, subdirs[0])
    inner_files = os.listdir(first_subdir)
    print(f"Files inside '{subdirs[0]}': {len(inner_files)} (First 5: {inner_files[:5]})")

if len(files) > 0:
    print(f"First 5 files: {files[:5]}")
