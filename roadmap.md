## Note: This sample will be filled in later. This is just the sample text for doucumentation


Part 1: Iris dataset
The Iris dataset is ideal. It has 150 rows, 4 floating-point features per row (sepal/petal length and width in cm), and a categorical species label. Each row so to be treated as a record Alice wants to keep private. The numeric features map cleanly to the real-number arithmetic that CKKS supports natively.

Step 2 — Query functions
The natural queries for Iris are statistical: average sepal length across all 150 samples, min/max petal width, and sum  in which any partially homomorphic or leveled FHE scheme handles. for additional work, also count how many samples have sepal length above a threshold. this requires a comparison, which needs FHE rather than PHE and introduces significantly more noise.
But this can also be done with the average, sum, min, and max for a simple implementation. Add the threshold comparison as a stretch goal to demonstrate the contrast between PHE and FHE.

Step 3 — Scheme selection
Current scheme is CKKS (Cheon-Kim-Kim-Song). CKKS is designed for approximate arithmetic over real numbers. Batch multiple values into a single ciphertext using SIMD-style packing, so Alice can encrypt an entire column of 150 values in one ciphertext. Carol then computes a sum by adding all slots together using a rotation-and-add reduction.
For the comparison in Step 4b, compare CKKS against BFV, which operates over integers. Scale the float features to integers (multiply by 10), which introduces approximation error. This will be used as the point for analysis.

Part 4: Performance analysis
Current step:  Step 4a to compare encrypted vs. plaintext computation

Plaintext: NumPy computes average of 150 values in microseconds.
Encrypted (CKKS): key generation takes n seconds; encryption of all rows takes seconds; the homomorphic sum takes additional time; decryption adds more. Total latency is 10–100× slower depending on parameters.

I'll vary the different data set sizes with 10, 30, 75, 150 rows and polynomial degree (8192, 16384) to show how both latency and ciphertext size scale. Plotting these curves with matplotib. This will tell us the precise the cost Alice pays for confidentiality.
