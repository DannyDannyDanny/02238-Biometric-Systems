# Finger Knuckle Print (FKP)
## 02238-Biometric-Systems
[Assignment Link](https://www.overleaf.com/project/5d1c6d1d5f279e0d54109a2d)

### Paper formalities
* [Paper Submission Instructions](https://fg-biosig.gi.de/biosig-2019/paper-submission.html)


### Data

The dataset, THU-FVFDT2[^data] contains Region-of-Interest (ROI) images of finger dorsal texture from 610 different subjects. A subject's sample consists of two normalized images of the same ROI - taken separately and normalized to 200x100 pixels. The first 390 subjects are sampled within 3-7 days. The remaining 220 subjects are sampled within approximately half a minute.[^data]

### Pre-processing
* [histogram equalization](https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html)


### Training
Training is based loosely on Keras MNIST example[^mnist].

### Acronyms:
* FAR - False Acceptance Rate
* GAR - Genuine Acceptance Rate
* ROC - Receiver Operator Characteristic
* DET - Detection Error Tradeoff


### Research papers
* Kannala[^1]
* Kumar[^2]
* Zhang[^3]
* Siamese CNN[^4]
* Siamese Neural Networks for One-shot Image Recognition[^5]
* Old Paper on siamese networks[^6]




[^1]: Kannala-BSIF-(2012)
[^2]: Kumar-FK-(2009)
[^3]: Zhang-FKP-(2009)
[^4]: https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/learning-tracking-siamese.pdf
[^5]: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
[^6]: http://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
[^data]: [Tsinghua University Finger Vein and Finger Dorsal Texture Database](http://www.sigs.tsinghua.edu.cn/labs/vipl/thu-fvfdt.html)
[^mnist]: [Keras MNIST Example GitHub](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
