from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# CROP GUARD - DATASET PREPARATION
# ==========================================

# Location of the PlantVillage color images
DATASET_DIR = Path("data/raw/PlantVillage/raw/color")

# Where our prepared dataset information will be saved
OUTPUT_DIR = Path("data/processed")

# Crops we are using for CropGuard V1
SELECTED_CROPS = [
    "Apple",
    "Corn",
    "Grape",
    "Potato",
    "Tomato"
]


def get_crop_name(class_name):
    """
    Extract the crop name from a PlantVillage class name.

    Example:
    Tomato___Late_blight
    becomes:
    Tomato
    """

    return class_name.split("___")[0]


def main():

    print("\n===================================")
    print("   CropGuard Dataset Preparation")
    print("===================================\n")

    # Check that the dataset exists
    if not DATASET_DIR.exists():
        print("ERROR: PlantVillage dataset was not found.")
        print(f"Expected location: {DATASET_DIR}")
        return

    print(f"Dataset found: {DATASET_DIR}\n")

    records = []

    # Find every disease/healthy class
    class_directories = [
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]

    print(f"Total classes found: {len(class_directories)}\n")

    # Read images from selected crops
    for class_dir in class_directories:

        class_name = class_dir.name
        crop_name = get_crop_name(class_name)

        # Ignore crops we are not using in V1
        if crop_name not in SELECTED_CROPS:
            continue

        # Find image files
        image_files = [
            file for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{class_name}: {len(image_files)} images")

        for image_file in image_files:

            records.append({
                "image_path": str(image_file),
                "crop": crop_name,
                "class": class_name
            })

    # Convert everything into a DataFrame
    df = pd.DataFrame(records)

    print("\n-----------------------------------")
    print(f"Total selected images: {len(df)}")
    print(f"Total selected classes: {df['class'].nunique()}")
    print("-----------------------------------\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------
    # TRAIN / VALIDATION / TEST SPLIT
    # ------------------------------------------

    # First split:
    # 80% training
    # 20% temporary data
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["class"]
    )

    # Second split:
    # Temporary data is divided equally:
    # 10% validation
    # 10% test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["class"]
    )

    # Add split information
    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    validation_df["split"] = "validation"
    test_df["split"] = "test"

    # Combine everything
    final_df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True
    )

    # Save complete dataset manifest
    output_file = OUTPUT_DIR / "dataset.csv"

    final_df.to_csv(output_file, index=False)

    # Save separate files too
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # ------------------------------------------
    # SUMMARY
    # ------------------------------------------

    print("Dataset preparation complete! ✅\n")

    print(f"Training images:   {len(train_df)}")
    print(f"Validation images: {len(validation_df)}")
    print(f"Test images:       {len(test_df)}")
    print(f"Total images:      {len(final_df)}\n")

    print("Files created:")

    print(f"  {OUTPUT_DIR / 'dataset.csv'}")
    print(f"  {OUTPUT_DIR / 'train.csv'}")
    print(f"  {OUTPUT_DIR / 'validation.csv'}")
    print(f"  {OUTPUT_DIR / 'test.csv'}")

    print("\nCropGuard V1 crops:")

    for crop in SELECTED_CROPS:
        print(f"  🌱 {crop}")

    print("\n===================================")
    print("       Preparation Finished")
    print("===================================\n")


if __name__ == "__main__":
    main()