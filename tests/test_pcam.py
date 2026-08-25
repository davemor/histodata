from __future__ import annotations

import csv
from pathlib import Path

import h5py

import histodata.adapters.pcam as pcam_module
from histodata.adapters.pcam import PCamAdapter


def test_builds_and_deep_validates_native_hdf5_index(tmp_path: Path, monkeypatch) -> None:
    counts = {"train": 2, "validation": 1, "test": 1}
    monkeypatch.setattr(pcam_module, "SPLIT_COUNTS", counts)
    package = tmp_path / "pcam"
    data = package / "data" / "pcam"
    data.mkdir(parents=True)
    for split, count in counts.items():
        publisher_split = "valid" if split == "validation" else split
        prefix = data / f"camelyonpatch_level_2_split_{publisher_split}"
        with h5py.File(f"{prefix}_x.h5", "w") as file:
            file.create_dataset("x", shape=(count, 96, 96, 3), dtype="uint8")
        with h5py.File(f"{prefix}_y.h5", "w") as file:
            labels = [index % 2 for index in range(count)]
            file.create_dataset("y", data=[[[[value]]] for value in labels], dtype="uint8")
        with Path(f"{prefix}_meta.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["coord_y", "coord_x", "tumor_patch", "center_tumor_patch", "wsi"],
            )
            writer.writeheader()
            for index in range(count):
                writer.writerow(
                    {
                        "coord_y": index + 10,
                        "coord_x": index + 20,
                        "tumor_patch": bool(index % 2),
                        "center_tumor_patch": bool(index % 2),
                        "wsi": f"slide-{split}",
                    }
                )

    adapter = PCamAdapter()
    table = adapter.build_index(package)
    assert table.num_rows == 4
    assert table["id"].to_pylist() == [
        "test/000000",
        "train/000000",
        "train/000001",
        "validation/000000",
    ]
    assert table["path"].to_pylist()[0].endswith("split_test_x.h5")
    assert adapter.validate_table(table) == []
    assert adapter.deep_validate(package, table) == []
