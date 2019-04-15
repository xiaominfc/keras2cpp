/*
 * bidirectional.h
 * Copyright (C) 2019 xiaominfc(武汉鸣鸾信息科技有限公司)
 * Email: xiaominfc@gmail.com
 * Distributed under terms of the MIT license.
 */

#ifndef BIDIRECTIONAL_H
#define BIDIRECTIONAL_H

#include "../baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class Bidirectional final : public Layer<Bidirectional> {
            //Tensor weights_;
            std::unique_ptr<BaseLayer> forward_layer;
            std::unique_ptr<BaseLayer> backward_layer;

        public:
            Bidirectional(Stream& file);
            Tensor operator()(const Tensor& in) const noexcept override;
        };
    }
}

#endif /* !BIDIRECTIONAL_H */
