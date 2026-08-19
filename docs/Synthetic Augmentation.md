# Synthetic Augmentation

CA 19-9, C-peptide, and insulin are generated measurements, never observed measurements. The modules operate on canonical native event rows and preserve those native rows. Each generated row has `provenance="augmented"`, a non-empty `augmentation_module`, and the canonical unit from the Step 2 feature registry.

The modules receive a seeded NumPy generator. Their assumptions JSON records module/version, causal assumptions, input features, noise distribution, clipping policy, seed, and limitations. Version and assumptions hashes are linked from the cohort manifest. The Step 2 schema has no extra derivation-version column, so derivation version is represented in assumptions and manifest metadata while event shape remains unchanged.

No module reads cancer labels, future events, tumour attributes, treatment, pathology, survival, or any other post-index information. CA 19-9 uses weak overlapping stochastic distributions and cannot encode a site label. Adversarial tests remove labels, shuffle labels, remove later events, repeat seeds, and change seeds to ensure generated values are not a hidden label channel.

Generated biomarkers are simulation artifacts for pipeline development. They are not observed laboratory data and do not establish a biological or population claim.
