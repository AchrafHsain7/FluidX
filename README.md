# FluidX


## To DO:
- Implement LSTM sampling
- Implement QCBM sampling
- Implement PixelCNN++ sampling
- ADD Physic informed loss
- Add perceptual loss
- Add frequency domain loss
- Larger Network + Residual Connections ?
- Data Analysis Dashbord ....

## Problem:
- Oops the LLM hallucinated as always (who could've tahought): QCBM in its current form only learn the codebook distribution which is useless since we already know ir anyways (duh!)
- The solution is to make it sample continious values instead of only discrete ones. Maybe by using some rotation gates at the end with scaling?
- Also instead of the LSTM approach which is autoregressive assume that the entanglement wil take care of that, and just sample all 24 vectors in one pass
- What about the loss function? I dont think Maximum average means will do the job?
- This will require some strong GPU!

