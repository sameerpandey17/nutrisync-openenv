from huggingface_hub import HfApi
import os
import sys

def deploy():
    repo_id = "strangesam17/nutrisync-v2"
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)
    folder_path = "."

    if not os.path.exists("app.py"):
        print("Error: app.py not found. Renaming app_ui.py if possible...")
        if os.path.exists("app_ui.py"):
            os.rename("app_ui.py", "app.py")
        else:
            print("Error: No application file found.")
            sys.exit(1)

    api = HfApi()

    print(f"Starting upload to Hugging Face Space: {repo_id}...")
    
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="space",
            token=token,
            path_in_repo="",
            ignore_patterns=[
                ".git*", 
                "**/__pycache__/*", 
                ".venv/*", 
                ".env*", 
                "*.log", 
                "push_*.txt", 
                "error.txt",
                "deploy_hf.py"
            ]
        )
        print("\nSUCCESS: All files uploaded successfully!")
        print(f"Your Space is now available at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"\nERROR during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy()
