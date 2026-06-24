if __name__ == "__main__":
    # Define where your raw unzipped dataset sits
    # (Adjust the base path if your script is in a different folder)
    INPUT_BASE = "Downloads/Deepfake dataset/deepfake_database"
    OUTPUT_BASE = "dataset_clean"
    
    # 1. Clean the Training Set
    print("\n--- Processing Training Data ---")
    clean_and_crop_dataset(f"{INPUT_BASE}/train/real", f"{OUTPUT_BASE}/train/real")
    clean_and_crop_dataset(f"{INPUT_BASE}/train/df", f"{OUTPUT_BASE}/train/fake")
    
    # 2. Clean the Validation Set
    print("\n--- Processing Validation Data ---")
    clean_and_crop_dataset(f"{INPUT_BASE}/validation/real", f"{OUTPUT_BASE}/validation/real")
    clean_and_crop_dataset(f"{INPUT_BASE}/validation/df", f"{OUTPUT_BASE}/validation/fake")
    
    # 3. Clean the Test Set
    print("\n--- Processing Test Data ---")
    clean_and_crop_dataset(f"{INPUT_BASE}/test/real", f"{OUTPUT_BASE}/test/real")
    clean_and_crop_dataset(f"{INPUT_BASE}/test/df", f"{OUTPUT_BASE}/test/fake")