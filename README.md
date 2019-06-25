# Finger Knuckle Print (FKP)
## 02238-Biometric-Systems


### Data

The dataset, THU-FVFDT2[^data] contains Region-of-Interest (ROI) images of finger dorsal texture from 610 different subjects. A subject's sample consists of two normalized images of the same ROI - taken separately and normalized to 200x100 pixels. The first 390 subjects are sampled within 3-7 days. The remaining 220 subjects are sampled within approximately half a minute.[^data]

### Training
Training is based loosely on Keras MNIST example[^mnist].

### Acronyms:
* FAR - False Acceptance Rate
* GAR - Genuine Acceptance Rate
* ROC - Receiver Operator Characteristic
* DET - Detection Error Tradeoff


### Research papers

| Paper | Description     | Performance     |
| :------------- | :------------- | :------------- |
| Kannala[^1] |...|...|
| Kumar[^2]   |...|...|
| Zhang[^3]   |...|     97%GAR / 0.02%FAR |




[^1]: Kannala-BSIF-(2012)
[^2]: Kumar-FK-(2009)
[^3]: Zhang-FKP-(2009)
[^data]: [Tsinghua University Finger Vein and Finger Dorsal Texture Database](http://www.sigs.tsinghua.edu.cn/labs/vipl/thu-fvfdt.html)
[^mnist]: [Keras MNIST Example GitHub](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
