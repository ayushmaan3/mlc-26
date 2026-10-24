# Paste into a NEW cell of FT_check.ipynb (same Colab runtime, it holds /content/e5_ft). Run once.
# 1) keeps a copy of the fine-tuned model in this account's Drive (MyDrive/e5_ft)
# 2) uploads it to Kaggle as the private dataset mlc26-ftmodel (a file picker opens: choose your kaggle.json)
import os, json, shutil, subprocess
from google.colab import files
shutil.copytree("/content/e5_ft", "/content/drive/MyDrive/e5_ft", dirs_exist_ok=True)
print("saved to Drive: MyDrive/e5_ft")
up = files.upload()
os.makedirs("/root/.kaggle", exist_ok=True)
open("/root/.kaggle/kaggle.json", "wb").write(list(up.values())[0])
os.chmod("/root/.kaggle/kaggle.json", 0o600)
user = json.load(open("/root/.kaggle/kaggle.json"))["username"]
d = "/content/ftmodel"
shutil.copytree("/content/e5_ft", d, dirs_exist_ok=True)
json.dump({"title": "mlc26-ftmodel", "id": f"{user}/mlc26-ftmodel", "licenses": [{"name": "CC0-1.0"}]},
          open(f"{d}/dataset-metadata.json", "w"))
print(os.listdir(d))
r = subprocess.run(f"kaggle datasets create -p {d} --dir-mode skip", shell=True, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
