#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

class Layer {
protected:
    Matrix mInput;
    Matrix mOutput;

public:
    Layer() {}
    virtual Matrix forwardPropagation(const Matrix& input) = 0;
    virtual Matrix backwardPropagation(const Matrix& outputError, double learningRate) = 0;
    virtual ~Layer() {}
};

#endif