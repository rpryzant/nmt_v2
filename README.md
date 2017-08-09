
# NMT_V2

This is my second implementation of an end-to-end neural machine translation system. It's based primarily on [[Sundermeyer et al.]](https://pdfs.semanticscholar.org/d29c/f0f457ec2089fd4d776ef9a246de810be689.pdf) and [[Luong et al.]](https://arxiv.org/abs/1508.04025).

My first implementation is here (https://github.com/rpryzant/nmt) and it's not as good.


## Usage

The primary way to configure training and inference is by creating a _configuration file_. An example of a complete configuration file is `config.yaml`. 

Note that the first three tokens of any vocabulary files you are using must begin with unknown (`config.unk`), sentance start (`config.sos`), and sentence end (`config.eos`). By default this is `<unk>`, `<s>`, and `</s>`.

**Train a model**: this will execute the training run specified by `config.yaml`, saving all outputs, models, and graphs in `config.out_dir`. 
```
python main.py --config config.yaml
```

**Perform inference**: This will load a trained model in `config.out_dir` and print a complete line-by-line translation of  the file specified by `config.data_dir + config.test_prefix + config.tgt`. 
```
python inference.py --config config.yaml
```

## Performance

**Training details**: single-layer bidirectional LSTM with shared BPE vocabulary of 32000, embedding size of 512, hidden size of 512, trained with adam at 0.0001, batch size of 128, and 200k iterations.

| Dataset | BLEU |
|------------------------------------|------|
| Kyoto Free Translation Task (KFTT) | 21.1 |
| IWSLT English-Vietnamese           | 25.3 |


## Improvements over V1
  - Simpler structure: fewer classes, no seperate logic for attention, encoding/decoding is rolled into a single model base class
  - Better training pipeline with periodic sampling
  - Proper evaluation
  - Proper inference
  - Beam search decoding
  - Better input pipelines (using `tf.Iterator` + `tf.Dataset`
  - Only the useful stuff is written to tensorboard  



