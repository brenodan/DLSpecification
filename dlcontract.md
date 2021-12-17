# Design by deep learning contracts for Keras APIs
 
| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | My Name (me@example.org), AN Other (you@example.org) |
| **Sponsor**   | A N Expert (expert@example.org)                      |
| **Updated**   | YYYY-MM-DD                                           |
| **Obsoletes** | RFC it replaces, else remove this header             |
 
## Objective
We intend to annotate a set of Keras APIs (e.g., compile, fit) with contracts. Each contract contains a set of client obligations in the form of **preconditions** and prevalent training problems in DNN as **postconditions**. In case of a contract breach during runtime, Keras will throw warning messages to the app developers displaying location and type of silent bug (i.e., overfitting, dying relu, vanishing gradient, exploding gradient problem, etc.). Contracts will inform the application developers about the reason and how to solve an issue causing an unexpected output (e.g, low accuracy, high training time).

### Goals
- Document missing specifications, which are not covered by Keras yet, and do not throw any messages to users.
- Enable Keras to prevent silent bugs due to a subset of incorrect model architecture, data normalization, and some well-known training problems.
- Help users get their expected output by performing runtime assertion.

### Non-goals
- Follow the design-by-contract (DbC) principle while writing contracts with abstracted variables across different stages of ML pipeline (https://www.eiffel.com/values/design-by-contract/). 
- Introduce its benefits in the deep learning community so that it can be extended by other researchers and practitioners. 

## Motivation
Currently, DL app developers rely on purposely-built debugging facilities to detect silent bugs. These are often hard-to-use, require training, and incur additional development and maintenance costs. Also, even though existing documentation states the correct APIs utilization, developers often do not receive the expected output from the API. Several StackOverflow posts and GitHub commits show such kinds of problems. There are also research publications that investigate this issue while putting forward debugging techniques (i.e., DeepLocalize (ICSE 21) [1], AUTOTRAINER (ICSE 21) [2], NeuraLint (TOSEM 21) [3], UMLAUT (CHI 21) [4]).

But we envision that leveraging the DbC principles on Keras can help address the above problem with less effort than previous methods. Contracts can enable the compiler to detect API misuses early in the ML pipeline. Also, DL app developers will not need to use additional debugging or instrumentation tools for their code. Instead, the runtime will provide messages about the location and silent bugs types. 

## User Benefit
We will enable the compiler catch silent bugs early in the ML pipeline. Developers will receive debugging messages with location and fix suggestions as shown in the examples.

The headline in the release notes or blog post could be: **Catching Silent Bugs with DL Contracts**.

## Design Proposal

Our main idea is to annotate the compile and fit Keras APIs with contracts. Each contract contains a set of client expectations (**precondition**) and expected DNN output (**postcondition**). In particular, preconditions provide the location in the client code, and the postconditions refer to the type of silent bug. The compiler can use the information from contracts to directly detect performance bugs and provide messages with location and suggested fixes to the users at runtime. 

Consider a simple CNN with 99% test accuracy for the MNIST dataset from [Keras documentation](https://keras.io/examples/vision/mnist_convnet/). The following buggy example (accuracy~9.78%) shows the three different structure-related bugs which could appear in this DNN model. 

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
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
batch_size = 128
epochs = 15
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

If the data is not normalized before it is fed into the Fit API [7, 8] (lines 5, 6). If ‘relu’ or ‘sigmoid’ activation functions have been used in the last layer (line 19) of Dense API [2, 6, 7]. If loss=‘binary_crossentropy’ (line 20) has been applied as the loss function of Compile API [2, 4, 11]. These types of performance bugs will not cause the crash of the DL app but will reduce its accuracy. Currently, the Keras Documentation is insufficient to detect those [5] bugs, as it suggests those values are valid for the respective APIs.

## Detailed Design

The example below shows contracts on top of the **compile** API. They specify the activation function for multiclass and binary classification:

```python
@contract(context='last_layer', activation='softmax', loss_func='categorical_crossentropy')
@contract(context='last_layer', activation='activation_func_binary', loss_func='binary_crossentropy')
def compile(self,
            optimizer='rmsprop',
            loss=None,
            metrics=None,
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None,
            **kwargs):
```
The traditional DbC technique does not support writing contracts on an API with parameters of other ones. To address this issue, we put forward *ML Variables*, which work globally across all APIs in the ML stage pipeline.

Our proposed technique on Keras library will throw the following message for this incorrect model architecture-related bug that yields low performance:

```
Contract Breach for context given last_layer activation_function must not be relu, should be softmax.
```

The example below shows a contract on top of the **fit** API. It specifies the context, which determines the starting point of the runtime assertion checker. The runtime checks the ML variable "normalization_interval" with the data interval, which has been fed before training, after calling the Fit API.

```python
@contract(context='data_normalized', normalization_interval <=2')
def fit(self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False):
```

Keras will throw the following message for this data normalization bug:

```python
ContractViolated: Breach for Sequential:fit().context given: data_normalized, data should benormalized, should not be within 0.0 and 255.0. Condition 255.0 <= 2 not respected
```

In addition to this kind of data properties and model architecture-related bugs, our proposed technique will support specifying training-related properties and handle well-known training problems. For example, when training a DNN with ReLu as the activation function, the gradients of a large percentage of the neurons are zero, and the training accuracy is low [2]. We convert this training behavior as postcondition (expectation from training API) and correct activation function, which is precondition (obligations of client code). We use the ML variable ‘zero_gradients_percentage’ to specify the percentage of neurons whose gradients are 0 in recent iterations. 

The code below shows a contract on Keras Fit API that captures properties of hidden layers’ activation functions and corresponding training problems.

```python
@contract(context = ‘hidden_layers’,activation!=‘relu’,zero_gradients_percentage≤𝜆, accuracy≤𝜃)
def fit(self, x=None, y=None,...):
```

If Fit API call in the client code violates this contract during training, the application developer will be notified of the following contract violation message. 

```
Contract violation for Model:fit(). zero_gradients_percentage of 12.07 caused dying relu problem,
activation function should not be relu.
```

## References

[1] Mohammad Wardat, Wei Le, and Hridesh Rajan. 2021. DeepLocalize: Fault Localization for Deep Neural Networks. In ICSE’21: The 43rd International Conference on Software Engineering.

[2] Xiaoyu Zhang, Juan Zhai, Shiqing Ma, and Chao Shen. 2021. AUTOTRAINER: An Automatic DNN Training Problem Detection and Repair System. In ICSE’21: The 43rd International Conference on Software Engineering

[3] Amin Nikanjam, Houssem Ben Braiek, Mohammad Mehdi Morovati, and Foutse Khomh. 2021. Automatic Fault Detection for Deep Learning Programs Using Graph Transformations. ACM Trans. Softw. Eng. Methodol. 30, 5 (2021), 26 pages.

[4] Eldon Schoop, Forrest Huang, and Björn Hartmann. May 8–13, 2021. UMLAUT: Debugging Deep Learning Programs using Program Structure and Model Behavior. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems.

[5] Dalwinder Singh and Birmohan Singh. 2020. Investigating the impact of data normalization on classification performance. Applied Soft Computing 97 (2020), 105524.

[6] Simple MNIST convnet example from Keras documentation. https://keras.io/examples/vision/mnist_convnet/.

[7] Sergey Ioffe and Christian Szegedy. 2015. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning. PMLR, 448–456.

[8] B. Meyer. 1992. Applying ’design by contract’. Computer 25, 10 (1992), 40–51. https://doi.org/10.1109/2.161279
