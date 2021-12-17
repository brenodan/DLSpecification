# Design by deep learning contracts for Keras APIs
 
| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | My Name (me@example.org), AN Other (you@example.org) |
| **Sponsor**   | A N Expert (expert@example.org)                      |
| **Updated**   | YYYY-MM-DD                                           |
| **Obsoletes** | RFC it replaces, else remove this header             |
 
## Objective
 
What are we doing and why? What problem will this solve? What are the goals and
non-goals? This is your executive summary; keep it short, elaborate below.
 
We intend to annotate a set of Keras APIs (e.g., compile, fit) with contracts. Each contract contains a set of client obligations in the form of preconditions and prevalent training problems in DNN as postconditions. In case of a contract breach during runtime, Keras will throw warning messages to the app developers displaying location and type of silent bug (i.e., overfitting, dying relu, vanishing gradient, exploding gradient problem, etc.). Contracts will inform the application developers about the reason and how to solve an issue causing an unexpected output, such as low accuracy, high training time, etc.
Goals: 
To document missing specifications that are not specified by the existing library yet and does not throw any message to the user.
To release a new version of Keras with a set of contracts to prevent silent bugs due to a subset of incorrect model architecture, data normalization problems, and some well-known training problems.
To facilitate end-users of Keras APIs so that they get the expected output, after performing runtime assertion with postconditions, and the client code obligations as preconditions. 
Non-goals: 
Follow the design-by-contract principle while writing contracts with abstracted variables across different stages of ML pipeline (https://www.eiffel.com/values/design-by-contract/). 
To introduce its benefits in the deep learning community so that it can be extended by other researchers and practitioners.
To enhance the Keras debugging techniques and to facilitate its usage for beginner application developers.
 
## Motivation
 
Why this is a valuable problem to solve? What background information is needed
to show how this design addresses the problem?
 
This is a valuable problem to solve because, currently, developers have to rely on purposely-built debugging facilities to detect silent bugs. These are often hard-to-use and require some training, which incurs additional development and maintenance costs. By using contracts, we can enable the compiler to detect the Keras API misuses. By doing so, developers will not need to use any additional debugging or instrumentation tools for their code. Instead, they will receive messages at runtime, which would inform about silent bug location and type.
 
The library developers who will annotate contracts need background information about new contracts that can prevent which kinds of bugs. We provide an initial set of contracts that can prevent sequential model architecture, some well known training probems (dying all, vanishing gradient, exploding gradient, oscillating loss, overfitting etc.). We also show how an additional set of variables can be applied to write and implement these contracts. We will provide the implementation of those contracts using a python contract package that has been designed, evaluated using real-world buggy and correct versions of programs. We will also provide guidelines how to extend more contracts targeting other classes of bugs. We will share some examples and benchmarks to determine the precision and recall of our proposed technique.
 
The end-users who will develop DL applications don’t need any background information about those contracts as they just use the new version of Keras with annotated contracts.
 
Which users are affected by the problem? Why is it a problem? What data supports
this? What related work exists?
 
Currently, Keras app developers sometimes do not receive the expected output from API and . This is a problem because the existing documentation also states the valid usage of those APIs. Several StackOverflow posts and GitHub commits show such kinds of problems. Several research publications investigate this issue while putting forward debugging techniques (i.e., DeepLocalize (ICSE 21), AUTOTRAINER (ICSE 21), NeuraLint (TOSEM), UMLAUT (CHI 21), and DeepDiagnosis (ICSE 22)). But we envision that leveraging the Design-by-Contract principles on Keras can help address the problem above while requiring less effort than previous methods.
 
## User Benefit
 
How will users (or other contributors) benefit from this work? What would be the
headline in the release notes or blog post?
 
Users will directly be benefited from this proposed technique by importing the new version of Keras with annotated contracts. It would be very helpful for the deep learning developers community who are suffering from some well-known model architecture, data, training problems that does not exhibit crash while execution. They will also get a precise suggestion to fix the issue as shown in the examples.
 
The headline in the release notes or blog post could be: 
Design by deep learning contracts for Keras APIs
 
## Design Proposal
 
This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.
 
Factors to consider include:
 
* UX and usability (Show demo)
* How will this change impact users, and how will that be managed? (Show demo)
* Performance implications 
* Dependencies (Pycontracts)
* Maintenance (Not sure)
* Backwards compatibility (<Python 3.9)

To demonstrate the usability of DL Contract, let us consider a simple Convolutional Neural Network(CNN) that achieves 99% test accuracy MNIST dataset from Keras documentation [https://keras.io/examples/vision/mnist_convnet/]. The following buggy example (accuracy~9.78%) shows the three different structure-related bugs which could appear in this DNN model. If the data is not normalized before it is fed into the Fit API [7, 8] (lines 5, 6). If ‘relu’ or ‘sigmoid’ activation functions have been used in the last layer (line 19) of Dense API [2, 6, 7]. If loss=‘binary_crossentropy’ (line 20) has been applied as the loss function of Compile API [2, 4, 11]. 

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="relu"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 15

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

```

This shows the three different performance-related bugs which could appear in this DNN model.3If the data is not normalized before it is fed into the Fit API [7,8] (lines 5, 6). If‘relu’or‘sigmoid’activation functions have been used in the last layer (line 19) of Dense API [2,6,7].If loss=‘binary_crossentropy’(line 20)  has been applied as the loss function of Compile API [2,4,11]. Any of the above statements could result in performance-related bugs, defined to include low-accuracy and high-training time bugs [45]. This type of performance bug [45] is not prevalent in the traditional software as this deep learning program will not crash due to the aforementioned usage of parameters of the APIs. TheKerasdocumentation [5] suggests that all of these values are valid for the respective APIs. Thus, the documentation is still insufficient to detect such bugs.


So, we propose this technique to annotate respective Keras API in the later part of ML pipeline (compile, fit) so that those bugs could be detected and shown to the end-users without need of any other debugging facilities. For example, users will get these error messages directly with the annotated version of Keras without any lines of code.



So, we propose this technique to annotate respective Keras API in the later part of ML pipeline (compile, fit) so that those bugs could be detected and shown to the end-users without need of any other debugging facilities. For example, users will get these error messages directly with the annotated version of Keras without any lines of code.


## Detailed Design

We can achieve this goal by writing our designed contracts on top of compile API and Fit API with the ML variables that have the abstraction. The current DbC does not provide such kind of mechanism to write contracts without the formal parameters of an API and also write contracts involving multiple parameters.
To detect the issue with the last layer activation function and appropriate loss for multiclass classification, we can write the below contract. Our runtime assertion checker can detect contract violations without anything to do on the client application developer side.

We can write below contracts on compile API to specify accurate activation function for multiclass and binary classification in the following way:

```python
@contract(context='last_layer', activation='softmax', loss_func='categorical_crossentropy')
@contract(context='last_layer', activation='activation_func_binary', loss_func='str,=$binary_crossentropy')
def compile(self,
            optimizer='rmsprop',
            loss=None,
            metrics=None,
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None,
            **kwargs):
```


## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
