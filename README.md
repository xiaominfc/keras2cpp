# Keras2cpp ![release](https://img.shields.io/github/release/gosha20777/keras2cpp.svg?colorB=red) ![lisense](https://img.shields.io/github/license/gosha20777/keras2cpp.svg) [![Build Status](https://travis-ci.org/gosha20777/keras2cpp.svg?branch=master)](https://travis-ci.org/gosha20777/keras2cpp)
![keras2cpp](docs/img/keras2cpp.png)

Keras2cpp is a small library for running trained Keras models from a C++ application without any dependences. 

Design goals:

- Compatibility with networks generated by Keras using TensorFlow backend.
- CPU only, no GPU.
- No external dependencies, standard library, C++17.
- Model stored on disk in binary format and can be quickly read.
- Model stored in memory in contiguous block for better cache performance.

*Not not layer and activation types are supported yet. Work in progress*

Supported Keras layers:
- [x] Dense
- [x] Convolution1D
- [x] Convolution2D
- [ ] Convolution3D
- [x] Flatten
- [x] ELU
- [x] Activation
- [x] MaxPooling2D
- [x] Embedding
- [x] LocallyConnected1D
- [x] LocallyConnected2D
- [x] LSTM
- [ ] GRU
- [ ] CNN
- [X] BatchNormalization
- [X] Bidirectional

Supported activation:
- [x] linear
- [x] relu
- [x] softplus
- [x] tanh
- [x] sigmoid
- [x] hard_sigmoid
- [x] elu
- [x] softsign
- [x] softmax

Other tasks:
- [x] Create unit tests
- [x] Create Makefile
- [x] Code refactoring *(in progress)*

The project is compatible with Keras 2.x (all versions) and Python 3.x

# Example

python_model.py:

```python
import numpy as np
from keras import Sequential
from keras.layers import Dense

#create random data
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(1, input_dim=10)
])
model.compile(loss='mse', optimizer='adam')

#train model by 1 iteration
model.fit(test_x, test_y, epochs=1, verbose=False)

#predict
data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
prediction = model.predict(data)
print(prediction)

#save model
from keras2cpp import export_model
export_model(model, 'example.model')
```

cpp_mpdel.cc:

```c++
#include "src/model.h"

using keras2cpp::Model;
using keras2cpp::Tensor;

int main() {
    // Initialize model.
    auto model = Model::load("example.model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in{10};
    in.data_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Run prediction.
    Tensor out = model(in);
    out.print();
    return 0;
}
```

# How to build and run

*Tested with Keras 2.2.1, Python 3.6*

```bash
$ git clone https://github.com/xiaominfc/keras2cpp.git
$ cd keras2cpp
$ mkdir build && cd build
$ python3 ../python_model.py
[[-1.85735667]]

$ cp ./example.model ./build/ 
$ cmake ..
$ cmake --build .

#mac os 
$ g++ -std=c++17 -Wl,-rpath -Wl,$(pwd)  -lkeras2cpp -L./ -I../src/ ../cpp_model.cc -o cpp_model
#linux
$ g++ -std=c++17 -L. -I ../src/  -o cpp_model ../cpp_model.cc ./libkeras2cpp.* 

$ ./cpp_model
[ -1.857357 ]
```

# License

MIT

# Similar projects

I found another similar projects on Github:
- <https://github.com/pplonski/keras2cpp/>;
- <https://github.com/moof2k/kerasify>
- <https://github.com/Dobiasd/frugally-deep>

But It works only with Kekas 1 and didn’t work for me. 
That's why I wrote my own implementation.
