# Environments of the Tetris runs

These files are exported from the image the Tetris baseline runs used,
`bombe-revision-leap:scored-env` (`sha256:7a33b77817b5`).
That image is LEAP's own image built from `Dockerfile` on 2026-03-13
(`sha256:2e4b25e098c8`) with `/opt/conda/envs/leap` replaced by the env of the container the scored runs ran in.
detectron2, yacs, fvcore, and faiss were installed into that running container by hand,
so the image built from the original Dockerfile alone could not run LEAP.

- OS: Ubuntu 22.04.3, CUDA 11.8.
- `leap`: Python 3.10.20, torch 2.1.2+cu118, ultralytics 8.4.22,
  detectron2 at `8a9d885b3d4dcf1bef015f0593b872ed8d32b4ab`, yacs 0.1.8, fvcore 0.1.5.post20221221,
  faiss-gpu 1.7.4 (conda, `pytorch` channel).
- `retinanet`: Python 3.13.12, torch 2.6.0+cu124.

Files per env:

- `<env>.explicit.txt`: `conda list --explicit`, the exact conda packages (`conda create -n <env> --file <env>.explicit.txt`).
- `<env>.conda.yml`: `conda env export`, conda and pip packages with versions.
- `<env>.pip.txt`: `pip freeze`.
  Lines with `@ file:///...` point at conda build artifacts and are installed by the conda lists instead.

`Dockerfile` pins detectron2 to the commit above and installs the packages that were added by hand.
