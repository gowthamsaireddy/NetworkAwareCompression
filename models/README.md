# Model Checkpoints Directory

This directory is designated for the pre-trained, frozen U-Net model checkpoints used by the Receiver application.

## Policy & Notice
- **Do not commit binary checkpoints** (`.pt`, `.pth`, `.onnx`) to Git.
- Model architecture and trained weights will be provided in Phase 5.
- Do not train models inside the receiver application; all inference runs strictly in frozen evaluation mode.
- Checkpoint files should be placed here locally by the researcher before executing evaluation pipelines.

