# Palmistry AI - YOLOv8
Được thực hiện bởi nhóm HKT - DA0001, gồm các thành viên:
1. Nguyễn Bảo Hân - 31251027458 
2. Trần Thế Đăng Khoa - 31251020280 
3. Hoàng Bảo Trân - 31251020280

Du an nay dung YOLOv8 de nhan dien 4 duong chi tay chinh tu anh hoac webcam:

- `life`: duong sinh menh
- `heart`: duong tinh cam
- `head`: duong tri dao
- `fate`: duong van menh

Ung dung chinh nam trong `gui.py`. File nay load best model da train tai:

```text
runs/detect/palm_finetune_fate/weights/best.pt
```

## Best model

Model chinh cua project la:

```text
runs/detect/palm_finetune_fate/weights/best.pt
```

File `best.pt` la weight tot nhat sau khi fine-tune YOLOv8 tren dataset palmistry. Khi upload GitHub, co the chi can dua file `best.pt` nay kem voi code de chay giao dien nhan dien.

## Dataset

Dataset duoc lay tu Roboflow Universe:

```text
https://universe.roboflow.com/palmistry-ccmq5/palmistry-dhpnb/dataset/2/download
```

Khi tai dataset, chon dinh dang **YOLOv8**. Sau khi giai nen, cau truc dataset gom:

```text
train/
valid/
test/
data.yaml
```

File `data.yaml` cau hinh duong dan dataset va 4 lop nhan dien:

```yaml
names: ['fate', 'head', 'heart', 'life']
```

## Cau truc thu muc

```text
PALM/
├── gui.py              # Ung dung giao dien Tkinter de xem chi tay
├── train.ipynb         # Notebook train/kiem tra model YOLOv8
├── data.yaml           # Cau hinh dataset YOLOv8
├── yolov8n.pt          # Model YOLOv8 nano goc de fine-tune
├── train/              # Du lieu train tu Roboflow
├── valid/              # Du lieu validation tu Roboflow
├── test/               # Du lieu test tu Roboflow
└── runs/
    └── detect/
        └── palm_finetune_fate/
            └── weights/
                └── best.pt  # Best model dung cho gui.py
```

## Cach chay ung dung

Mo terminal tai thu muc `PALM`, sau do chay:

```bash
python gui.py
```

Trong giao dien:

- Chon `Mo webcam realtime` de quet bang webcam.
- Nhan `C` de chup/giu lai ket qua phan tich.
- Nhan `R` de quet lai.
- Nhan `Q` hoac `Esc` de thoat.
- Chon `Import anh` de phan tich anh co san.

## Cach hoat dong

`gui.py` dung `best.pt` de detect cac bounding box cua 4 duong chi tay. Sau khi detect, chuong trinh:

1. Lay nhan duong chi tay tu ket qua YOLO.
2. Loc cac nhan thuoc `life`, `heart`, `head`, `fate`.
3. Neu mot duong bi detect nhieu lan, giu box co confidence cao nhat.
4. Tinh do dai, do doc va vi tri cua tung duong.
5. Sinh phan dien giai ngan dua tren ket qua detect.

## Train lai model

Mo `train.ipynb` va chay cell train:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=416,
    batch=4,
    device="cpu",
    workers=0,
    name="palmistry_yolov8_cpu"
)
```

Ket qua train se duoc luu vao:

```text
runs/detect/<ten_lan_train>/
```

File model tot nhat thuong nam o:

```text
runs/detect/<ten_lan_train>/weights/best.pt
```

Neu muon `gui.py` dung model moi, sua bien `MODEL_PATH` trong `gui.py` ve duong dan file `best.pt` moi.
