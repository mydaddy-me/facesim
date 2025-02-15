from pathlib import Path
import cv2
from tqdm import tqdm
import typer

from facesim import crop_face, crop_parts

app = typer.Typer()

class fs:
    data = Path('data')
    faces = Path('faces')
    eyebrow = Path('eyebrow')
    eye = Path('eye')
    nose = Path('nose')
    lips = Path('lips')

@app.command()
def faces():
    fs.faces.mkdir(exist_ok=True)
    for f in tqdm(fs.data.rglob('*.jpg')):
        label = f.parent.name
        image = cv2.imread(str(f))
        face = crop_face(image)
        
        if face is None:
            continue
        
        h, w = face.shape[:2]
        
        if(min(h, w) < 256):
            continue

        out_dir = fs.faces / label
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f.name
        cv2.imwrite(str(out_file), face)

@app.command()
def parts():
    fs.eyebrow.mkdir(exist_ok=True)
    fs.eye.mkdir(exist_ok=True)
    fs.nose.mkdir(exist_ok=True)
    fs.lips.mkdir(exist_ok=True)


    for f in tqdm(fs.faces.rglob('*.jpg')):
        label = f.parent.name
        face = cv2.imread(str(f))

        parts = crop_parts(face)
        
        if parts.left_eyebrow is not None:
            left_eyebrow_dir = fs.faces / label
            left_eyebrow_dir.mkdir(exist_ok=True)
            left_eyebrow_file = left_eyebrow_dir / f.name

            cv2.imwrite(str(left_eyebrow_file), parts.left_eyebrow)

    
        if parts.right_eyebrow is not None:
            right_eyebrow_dir = fs.faces / label
            right_eyebrow_dir.mkdir(exist_ok=True)
            right_eyebrow_file = right_eyebrow_dir / f.name

            cv2.imwrite(str(right_eyebrow_file), parts.right_eyebrow)

        if parts.left_eye is not None:
            left_eye_dir = fs.faces / label
            left_eye_dir.mkdir(exist_ok=True)
            left_eye_file = left_eye_dir / f.name

            cv2.imwrite(str(left_eye_file), parts.left_eye)

        if parts.right_eye is not None:
            right_eye_dir = fs.faces / label
            right_eye_dir.mkdir(exist_ok=True)
            right_eye_file = right_eye_dir / f.name

            cv2.imwrite(str(right_eye_file), parts.right_eye)

        if parts.nose is not None:
            nose_dir = fs.nose / label
            nose_dir.mkdir(exist_ok=True)
            nose_file = nose_dir / f.name

            cv2.imwrite(str(nose_file), parts.nose)

        if parts.lips is not None:
            lips_dir = fs.lips / label
            lips_dir.mkdir(exist_ok=True)
            lips_file = lips_dir / f.name

            cv2.imwrite(str(lips_file), parts.lips)

        

@app.command()
def nop():
    pass

if __name__ == "__main__":
    app()