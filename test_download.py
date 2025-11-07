#!/usr/bin/env python3
"""
Test if the model can be downloaded from Google Drive
Run this locally to verify before deploying
"""

import gdown
import os
from pathlib import Path

FILE_ID = "1nrza381tsPprpML7mAafd6DNU93y5BT7"
OUTPUT_PATH = "test_model.pkl"

print(f"🔗 Testing download for file ID: {FILE_ID}")
print(f"📥 Downloading to: {OUTPUT_PATH}")

try:
    # Method 1: Using gdown with file ID
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, OUTPUT_PATH, quiet=False, fuzzy=True)
    
    # Check file size
    if os.path.exists(OUTPUT_PATH):
        size = os.path.getsize(OUTPUT_PATH)
        size_mb = size / (1024 * 1024)
        
        print(f"\n✅ Download successful!")
        print(f"📊 File size: {size_mb:.2f} MB")
        
        if size < 1000:
            print(f"⚠️  WARNING: File is very small ({size} bytes)")
            print(f"This might indicate a download error or HTML error page")
            print(f"\nFirst 500 bytes of file:")
            with open(OUTPUT_PATH, 'rb') as f:
                print(f.read(500))
        else:
            print(f"✅ File appears valid (large enough to be a model)")
            
        # Cleanup
        os.remove(OUTPUT_PATH)
        print(f"\n🗑️  Test file removed")
        
    else:
        print(f"❌ File not found after download")
        
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print(f"\n💡 Solutions:")
    print(f"1. Check if file is shared publicly: https://drive.google.com/file/d/{FILE_ID}/view")
    print(f"2. Make sure 'Anyone with the link' can view")
    print(f"3. Try re-uploading the file to Google Drive")