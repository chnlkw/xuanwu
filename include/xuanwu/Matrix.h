//
// Created by chnlkw on 11/20/17.
//

#ifndef LDA_MATRIX_H
#define LDA_MATRIX_H


class Matrix {
protected:
    int nrow_;
    int ncol_;

    Matrix(int nrow, int ncol) :
            nrow_(nrow),
            ncol_(ncol) {}
};

template<class T>
class DenseMatrix : public Matrix {
public:
    DenseMatrix(int nrow, int ncol)
            : Matrix(nrow, ncol),
              arr_(nrow * ncol * sizeof(T)) {}

private:
    ArrayHost<T> arr_;
};

#endif //LDA_MATRIX_H
