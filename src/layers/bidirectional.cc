/*
 * bidirectional.cc
 * Copyright (C) 2019 xiaominfc(武汉鸣鸾信息科技有限公司) <xiaominfc@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "bidirectional.h"
#include "../model.h"


namespace keras2cpp{
    namespace layers{
        Bidirectional::Bidirectional(Stream& file) {
            forward_layer = Model::make_layer(file);
            backward_layer = Model::make_layer(file);
        }

        Tensor Bidirectional::operator()(const Tensor& in) const noexcept {
            auto fout = (*forward_layer)(in);
            auto bout = (*backward_layer)(in);
            size_t row_len = fout.dims_[0];
            size_t col_len =  fout.dims_[1] + bout.dims_[1];
            size_t half_col_len = fout.dims_[1];
            Tensor out{row_len,col_len};

            // fout.dims_[1] == bout.dims_[1]
            for(int i =0;i < row_len; i++) {
                for(int j=0; j < col_len / 2;j++){
                    out.data_[col_len*i + j] = fout(i,j);
                    out.data_[col_len*i + j + half_col_len] = bout(row_len -i - 1,j);
                }
            }
            return out;
        }
    }
}
