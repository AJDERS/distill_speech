# Distillation of acoustic models

Distillation of acoustic models. In collaboration with the [Danish Foundational Models](https://pure.au.dk/portal/da/projects/danish-foundation-models(073ab12f-0429-4ce0-877e-3f16eb38242d).html) project. This repository implements the student-teacher distillation paradigm, and utilizes it to distill Wav2Vec2 models, to smaller Wav2Vec2 models. The fundamental idea is that of [_DistilBERT_](https://arxiv.org/pdf/1910.01108.pdf).

All parameters for training, models and datasets are set in `src/config.py`.

Developers:

- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)

