# TODOs and ideas for composing generative models:

- [ ] **Train per-class energy models**
  - [ ] Train an energy model for each class independently.
  - [ ] Use the trained energies to generate or score samples.

- [ ] **Train joint energy-and-denoising models**
  - [ ] Combine energy-based modeling with denoising for each class.
  - [ ] Explore hybrid training objectives (e.g., joint score + energy consistency).

- [ ] **Combine generative and energy models**
  - [ ] Use the *generative process* of one class and the *energy* of another.
  - [ ] Apply importance weighting or rejection sampling during generation.

- [ ] **Orthodox approach**
  - [ ] Once energies for each class are trained, use MCMC (message passing, Gibbs, etc.) to sample from the product of distributions.

- [ ] **Metropolis-Hastings / MALA**
  - [ ] Implement Metropolis-Hastings to reject Langevin steps of the generative model using the likelihood ratio (`exp(-energy)`).
  - [ ] Explore Metropolis-Adjusted Langevin (MALA) dynamics for combined sampling.

- [ ] **Naive product of samples**
  - [ ] Generate samples separately for each class.
  - [ ] Compute their “cut” or intersection (cf. precision–recall density metric paper).
  - [ ] Experiment with sampling “balls around” the overlapping regions.

- [ ] **Sequential combination**
  - [ ] Sample from class 1.
  - [ ] Use that sample to seed the generative model of class 2.
  - [ ] Reject early if the trajectory goes in the “wrong direction”.

- [ ] **Iterative refinement**
  - [ ] Alternate sampling back and forth between class 1 and class 2 (especially with Flow Matching).
  - [ ] Check whether samples converge to a stationary point under both flow models.

- [ ] **Correctness and efficiency**
  - [ ] Analyze whether the iterative method samples the *true* product distribution.
  - [ ] Investigate how to fix potential bias using Metropolis-Hastings or importance weighting.

