# Application of Deep Learning (CNN's) to the Detection and Classification of Cracks in an Aircraft Panel

###
## Contents
1. [General Information](#gen_info)
2. [Abstract](#abstract)
3. [Usage Guide](#abstract)

## General Information <a name="gen_info"></a>


* ResearchGate Link: https://www.researchgate.net/publication/328334741_Incorporating_Inductive_Bias_into_Deep_Learning_A_Perspective_from_Automated_Visual_Inspection_in_Aircraft_Maintenance

* Conference paper title: *Incorporating Inductive Bias into Deep Learning: A Perspective from Automated Visual Inspection in Aircraft Maintenance, Dresden, Germany*

* Conference name: *10th International Symposium on NDT in Aerospace*

* Authors: *Vincentius EWALD, Xavier GOBY, Hidde JANSEN, Roger M. GROVES, Rinze BENEDICTUS*

* University: Delft University of Technology, Delft, The Netherlands

* Department: Aerospace Non-Destructive Testing Laboratory | Structural Integrity and Composites Group

## Abstract <a name="abstract"></a>

Narrow artificial intelligence, commonly referred as ‘weak AI’ in the last
couple years, has developed due to advances in machine learning (ML), particularly
deep learning, which has currently the best in-class performance among other
machine learning algorithms. In the deep learning framework, many natural tasks
such as object, image, and speech recognition that were impossible in the previous
decades using classical ML algorithms can now be done by a typical home personal
computer.
Deep learning requires a rapid collection of a large amount of data (also known
as ‘big data’) to create robust model parameters that are able to predict future
occurrences of certain event. In some domains, large datasets such as the CIFAR-10
image database and the MNIST handwriting database already exist. However, in
many other domains such as aircraft visual inspection, such a large dataset of
damage events is not available, and this is a challenge in training deep learning
algorithms to perform well to recognize material damage in aircraft structures.
As many computer science researchers believe, we also think that in order to
achieve a performance similar to human-level intelligence, AI should not start from
scratch. Introducing an inductive bias into deep learning is one way to achieve this
human-level intelligence in the aircraft inspection for damage. In this paper, we give
an example of how to incorporate domain knowledge from aerospace engineering
into the development of deep learning algorithms. We demonstrate the suitability of
our approach using data from fatigue testing of an aerospace grade aluminum
specimen to build a deep convolutional neural network that classifies crack length
according to the crack propagation curve obtained from fatigue test. The results of
this network were then compared to the same network that was not trained with
domain knowledge and the biased learning achieved a validation accuracy of
97.55% on determining crack length, while unbiased network selected the unwanted
parameter of sunlight intensity, however with 99.45% accuracy.


## Usage Guide <a name="usage_guide"></a>

The very first thing to do in order to test out and witness the prediction-making performance,
presented and discussed in the conference research paper, is as simple as just running the 
OOP_Predicter.py script! 