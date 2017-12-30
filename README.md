# stack-lstm-ner
PyTorch implementation of Transition-based NER system [1].

## Usage
### Requirements
  * Python 3.x
  * PyTorch 0.3.0


### Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```
Corresponding sequence of actions

```
SHIFT
REDUCE(PER)
OUT
OUT
SHIFT
SHIFT
REDUCE(LOC)
```

### Train

To train the model, simple run:
```
python train.py
```

## Author
* Huimeng Zhang: zhang_huimeng@foxmail.com


## References

[1] [Neural Architectures for Named Entity Recognition, Lample et al., 2016](http://www.aclweb.org/anthology/N16-1030.pdf)