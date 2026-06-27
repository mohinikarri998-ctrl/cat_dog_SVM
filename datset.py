import os
import requests

os.makedirs("dataset/train/cats", exist_ok=True)
os.makedirs("dataset/train/dogs", exist_ok=True)

print("Downloading Cats...")
for i in range(20):
    try:
        url = f"https://cataas.com/cat?unique={i}"
        r = requests.get(url, timeout=10)

        if r.status_code == 200:
            with open(f"dataset/train/cats/cat_{i}.jpg", "wb") as f:
                f.write(r.content)
            print(f"Saved cat {i}")
        else:
            print(f"Failed cat {i}")

    except Exception as e:
        print(f"Error downloading cat {i}: {e}")

print("Downloading Dogs...")
for i in range(20):
    try:
        res = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10).json()
        img_url = res["message"]

        img = requests.get(img_url, timeout=10)

        if img.status_code == 200:
            with open(f"dataset/train/dogs/dog_{i}.jpg", "wb") as f:
                f.write(img.content)
            print(f"Saved dog {i}")
        else:
            print(f"Failed dog {i}")

    except Exception as e:
        print(f"Error downloading dog {i}: {e}")

print("DONE")