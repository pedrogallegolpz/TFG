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

  - 3.2: **Methodology**. When we define
    $$
    \alpha_{ij}^{kc}
    $$
    "_combining Eqn1 and Eqn5_". At the definition's ending, the text says: "_the only constraint being that Y^c must be a smooth function_". BUT, I THINK THAT IT IS NOT TRUE. Eqn1 is a equation which was born in CAM with Neural Networks where their last layer before the classifier is a GAP. We only have 1 FC layer, so Eqn1 is TRUE for this case. But, is Eqn1 True for all NN? I think that no.

    On the other hand, for Grad-CAM, we define $$ \omega_k^c $$ from Eqn1, and it is not True for general NN either.

          if these wieghts are equal to 1. We have  that Y^c is the solution of the differential equation
       $$y'' + y'''(M+t)=0, \ \ \ \ \ \ \ M\in \mathbb{R}$$ 
      



### General doubts

1. For example, at GradCAM they say that we apply RELU because "we are only interested in the features that have a positive influence". What is the meaning of a negative influence? Why are positive values  accepted and negatives not? **Answer**: if we accepted negative values, when we normalize for visualization, we carry the min value (which is negative) to zero, so we don't know in the saliency map what pixels contributed to the class. Two examples:

        1.[-1,-2,1] --(normalization)--> [0.33,0,1]

        2.[1,-1,5] --(normalization)--> [0.33,0,1]

    Both examples have the same output, but in the first, attribute 1 doesn't contribute to class. However, in the second, attribute 1 does contribute to class. AND THEY HAVE THE SAME WEIGHT IN THE OUTPUT.
    If it was with ReLu we would have:
    
        1.relu([-1,-2,1])=[0,0,1] --(normalization)--> [0,0,1]

        2.relu([1,-1,5])=[1,0,5] --(normalization)--> [0.2,0,1]



# PROGRAMMING
We have to modulate the techniques. I've thought in two alternatives:

1. Create a **`class`** that inherits from `nn.Module`. We define here the NN and create methods that return the saliency map.
2. Create a separate **`class`** from the the **`class`** that inherits from `nn.Module`. Here, we define the methods for creating the saliency maps.

**Option 1** is like: CAM techniques are NN where you can consult a image saliency map, it's like the NN has an aditional service.

**Option 2** is like: you have your NN separated, and with this weights you want to know the saliency map of an image. Separating classes means separating concepts.