This directory stores downloaded model weights.

Run the following command to download them:
    python scripts/download_weights.py

Expected structure after download:
    models/
    ├── codeformer.pth
    ├── bisenet_face_parsing.pt
    ├── midas_small.pt
    ├── 3DDFA_V2/           ← Manual setup (see README)
    └── dreamid_weights/    ← Manual setup (see README)
        ├── vae/
        ├── unet/
        ├── scheduler/
        └── id_projection.pt
