#!/usr/bin/env python3
"""Copy one Bonsai MLP's packed tensors into a small benchmark fixture.

Usage: python3 crates/higgs-models/examples/extract_prefill_weights.py CHECKPOINT OUTPUT [LAYER]
Uses the standard library; never modifies the checkpoint.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys

if len(sys.argv) not in (3, 4):
    raise SystemExit(__doc__)
source, destination = map(Path, sys.argv[1:3])
layer = int(sys.argv[3]) if len(sys.argv) == 4 else 0
if source.resolve() == destination.resolve() or destination.exists():
    raise SystemExit("Output must be a new file distinct from the checkpoint")
prefix = f"language_model.model.layers.{layer}.mlp."
selected = {}
payload = bytearray()
with source.open("rb") as checkpoint:
    header_len = struct.unpack("<Q", checkpoint.read(8))[0]
    header = json.loads(checkpoint.read(header_len))
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            name = f"{prefix}{projection}.{field}"
            tensor = header[name]
            start, end = tensor["data_offsets"]
            checkpoint.seek(8 + header_len + start)
            data = checkpoint.read(end - start)
            if len(data) != end - start:
                raise SystemExit(f"Truncated tensor: {name}")
            selected[f"{projection}.{field}"] = {
                "dtype": tensor["dtype"], "shape": tensor["shape"],
                "data_offsets": [len(payload), len(payload) + len(data)],
            }
            payload.extend(data)
selected["__metadata__"] = {
    "source": str(source.resolve()), "layer": str(layer),
    "payload_sha256": hashlib.sha256(payload).hexdigest(),
}
encoded = json.dumps(selected, separators=(",", ":")).encode()
encoded += b" " * (-len(encoded) % 8)
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("xb") as output:
    output.write(struct.pack("<Q", len(encoded)))
    output.write(encoded)
    output.write(payload)
print(json.dumps({"fixture": str(destination), "bytes": destination.stat().st_size,
                  "metadata": selected["__metadata__"]}, indent=2))
