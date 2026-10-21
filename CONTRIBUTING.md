# Contributing

Most useful contributions come from people with real microstructure data.

- **Results on your data.** `stk measure your_images/ -o stk.csv`, fit any simple classifier or regression, and share how the coordinates separate your classes or track your property, including where they don't. Summary numbers are enough; you don't need to share images.
- **Group by specimen.** If several images come from one specimen, split by specimen, not by image. On UHCSDB, splitting by image inflated every descriptor by 0.06–0.09.
- **Expert-disagreement data.** A public microstructure dataset with labels from several annotators would let the ambiguity measure be tested properly.
- **Terminology.** If a coordinate's description doesn't match how your field uses a term, open an issue.

## Development

```bash
git clone https://github.com/parallaxmetrology/structural-topology-kernel
cd structural-topology-kernel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## The firm rule

`src/stkernel/_stk50.py` is frozen, including its known issues. Results across studies are only comparable if the kernel doesn't change. Fixes and new coordinates go into a new version with its own name, golden files and changelog; `tests/test_golden.py` must keep passing.

Be kind and specific. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
