# Notes about initial documentation

### 1.1 Articles about explainability in CNNs

#### Survey about XAI:

Relevant fragments:

- Abstract
- 1.Introduction
- 2.Explainability
  - 2.1, 2.2(definitions " "), 2.4(concept names), 2.5(all subsections (especially 2.5.1(three levels) and 2.5.2) and Fig.3 & Fig.4)
- 3.Transparent ML models. (Table 2 is enough: read the subsection if there are doubts)
- 4.Post-hoc explainability
  - 4.1 (summaries in section 1.Introduction)
  - 4.2 (fast reading)
  - 4.3 MLP (fast reading), CNN's (IMPORTANT: Grad-cam, LIME, last lines: explainability in adversarial detection. Understand why model fails in detecting adversarial examples) RNN, Hybrid transparent (fast reading).
- 5.XAI: Opportunities, challenges and future research needs.
  - Nice section for understanding the way taken by XAI: the current studies of XAI (all section is interesting, can be used for the introduction)
- 6.Toward responsible AI.
  - 6.1  (fast reading)
  - 6.2 (fast reading)
    - Discrimination: problem, how to solve it (fast reading)
    - Accountability: read the 4 topic understanding the problem
  - 6.3 "This section speculates about the potential of data fusion techniques to enrich the explainability of ML models and to compromise the privay of the data from which ML models are learned"
    - Fig13: different data fusion kinds or methods
    - 6.3.3 glance the section except the paragraphs where there are descriptions of XAI's implications .



### 1.2 Class Activation Mappings

- Section 1. Weakly-supervised object localization (IMPORTANT problem)
- Section 2. The bible. Understanding the meaning of f_k(x,y), F^k and M_c(x,y).
- Section 3.
  - Subsection 3.1. how to add the GAP to the network. Sometimes it's necessary to add another Convolutional Layers, read about it.
  - Subsection 3.2. Results.
- Section 4. Doubts: What is `fc6,fc7`? (maybe fc = full connected?)
  - Subsection 4.1. fast reading
  - Subsection 4.2. Overlapping between scene and object and detecting patterns.
- Section 5. Expected, fast reading.



### 1.3 Grad-CAM

- Subsection 5.3: i don't understand the method for evaluating faithfulness



### 1.4 Grad-CAM++

- Section 3.

  - 3.1: **Intuition**. At the beginning, author says that the derivative: 
    $$
    \frac{\partial y^c}{\partial A^k_{ij}}
    $$
    must to be higher for the feature map pixels that contribute to the presence of the object. The explanation is: if you detect a pattern, this pattern can contribute to the presence of the object or not. A^k_{ij}=1 on the pixels where the pattern is detected. If this pattern contribute, then the weights must to be positive (or higher than other patterns which don't contribute) because the presence of the object is labeled by 1 and no presence is labeled by 0. "So the higher number you add, the more contribute to the presence of the object"

  - 3.2: **Methodology**. Why is 
    $$
    \alpha_{ij}^{kc}
    $$
    defined like this? We try to weight the pattern with the same importance at 3.1. **Intuition**,  but here?



### General doubts

1. For example, at GradCAM they say that we apply RELU because "we are only interested in the features that have a positive influence". What is the meaning of a negative influence? Why are positive values  accepted and negatives not?

