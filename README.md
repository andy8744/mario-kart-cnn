# mario-kart-cnn

Joy-Con input capture and dataset tooling for behavioural cloning in
Mario Kart 8 Deluxe.

## Scope
- Read Joy-Con input on macOS (pygame)
- Map buttons/sticks consistently (two-hand right Joy-Con)
- Stream input over UDP (for NXBT or logging)
- Serve as the input side for dataset collection (video + actions)

NXBT, Bluetooth, and Switch pairing live in a separate repo / VM.

## Files
- `joycon_sender.py` – reads Joy-Con and emits UDP JSON packets
- `read_pad.py` – utility to inspect Joy-Con button/axis indices

## Notes
- Button mapping assumes rotated XYBA (two-hand orientation)
- Drift and pause are logged but not required for v0 models
