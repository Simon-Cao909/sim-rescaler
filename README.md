# sim-rescaler

A tool created from my research with SuperCDMS. When a particle hits a germanium detector, phonons propagate through the crystal and cause a response in 12 channels: P\*S1 and P\*S2 (\* from A to F)

From that, we use machine learning to predict where the particle had landed and the energy deposition. However, as machine learning requires labelled training data, we use simulated events that tell us the event position and energy. Expectedly, these simulations do not perfectly match real events, so we must perform a rescaling.

This tool was created to provide an interactive interface to do that rescaling and streamline the process.
