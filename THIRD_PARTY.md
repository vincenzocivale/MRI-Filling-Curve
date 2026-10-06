# External resources

This repository contains no third-party model or dataset files.

- **Gated DeltaNet-2** — official NVIDIA implementation: `NVlabs/GatedDeltaNet-2`; NVIDIA Source Code License-NC. The benchmark imports `lit_gpt.gdn2.GatedDeltaNet2` from an external installation.
- **MR-RATE-atlas** — `Forithmus/MR-RATE-atlas`; atlas-registered MRI; gated access; CC BY-NC-SA terms described by the dataset provider.
- **FOMO300K** — `FOMO-MRI/FOMO300K`; gated dataset collection with dataset-specific DUAs/licences. FOMO300K is a superset of OpenMind.
- **OpenMind** — `MIC-DKFZ/OpenMind`; OpenNeuro-derived 3D MRI collection; CC BY 4.0 dataset card.

Users are responsible for accepting and complying with the current upstream terms before use.

## Foundation-model comparison (`sfc fm`, optional `fm` extra)

No weights or model code are redistributed. Adapters in `src/sfc_gdn2/fm/` build each encoder from the
original library (MONAI, `dynamic-network-architectures`, Apache-2.0) or re-implement it in torch with the
original state_dict keys (MedicalNet, BrainMVP, BrainFM, MoME, AMAES), and load checkpoints in each repo's own format.
Upstream repos: BrainIAC, BrainSegFounder, nnssl (CC-BY-SA-4.0; OpenMind / nnFoundation), AMAES and FOMO26
baseline (no LICENSE file upstream: licence undetermined), BrainMVP, BrainFM (Apache-2.0), MoME / MoME+,
MedicalNet, nnU-Net (Apache-2.0). Check each repo's and checkpoint's current terms before use.
