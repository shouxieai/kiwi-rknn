#ifndef KIWI_INFER_RKNN_HPP
#define KIWI_INFER_RKNN_HPP

#include "kiwi-infer.hpp"

namespace rknn{

    std::shared_ptr<kiwi::Infer> load_infer_from_memory(const void* pdata, size_t size);
    std::shared_ptr<kiwi::Infer> load_infer(const std::string& file);

}; // namespace kiwi

#endif // KIWI_INFER_RKNN_HPP