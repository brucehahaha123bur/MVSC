from torch.utils.data import DataLoader
from data.harmony4d_mvsc import Harmony4DMVSCDataset

def main():
    root = "/Volumes/ResearchSSD/data/01_hugging/001_hugging/exo"

    ds = Harmony4DMVSCDataset(
        root=root,
        num_views=4,
        num_frames=4,
        crop_size=256,
        repeat=20,
    )

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for i, batch in enumerate(dl):
        print(f"[{i}] x.shape = {batch['x'].shape}")
        print("cam_names =", batch["cam_names"])
        print("frame_names =", batch["frame_names"])
        if i == 2:
            break

if __name__ == "__main__":
    main()