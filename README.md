
### NMT_V2

This is my second implementation of an end-to-end neural machine translation system.

My first implementation is here (https://github.com/rpryzant/nmt) and it's not as good.

Improvements:
  - Simpler structure: fewer classes, no seperate logic for attention, encoding/decoding is rolled into a single model base class
  - Better training pipeline with periodic sampling
  - Proper evaluation
  - Proper inference
  - Beam search decoding
  

Me fooling around with the nmt implementation of

https://github.com/tensorflow/nmt


TODO FIX GRAPHS
