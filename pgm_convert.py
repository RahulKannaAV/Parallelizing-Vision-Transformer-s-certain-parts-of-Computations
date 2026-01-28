# run once to convert dataset
from PIL import Image
import os
from pathlib import Path

input_dir = Path("images")
output_dir = "pgm_images"
os.makedirs(output_dir, exist_ok=True)

for f in input_dir.glob("*/*.jpg"):
    img = Image.open(str(f)).convert("L")
    img = img.resize((32,32))

    os.makedirs(f"{output_dir}/{f.parent.stem}", exist_ok=True)
    img.save(os.path.join(output_dir, str(f.parent.stem)+ f"/{str(f.parent.stem)}_{f.stem}.pgm"))
    
