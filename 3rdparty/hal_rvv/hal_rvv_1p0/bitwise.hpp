// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_BITWISE_HPP_INCLUDED
#define OPENCV_HAL_RVV_BITWISE_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_and8u
#define cv_hal_and8u cv::cv_hal_rvv::and8u
#undef cv_hal_or8u
#define cv_hal_or8u cv::cv_hal_rvv::or8u
#undef cv_hal_xor8u
#define cv_hal_xor8u cv::cv_hal_rvv::xor8u
#undef cv_hal_not8u
#define cv_hal_not8u cv::cv_hal_rvv::not8u

// Bitwise Binary Operations
struct AndOp {
    static vuint8m8_t apply(vuint8m8_t a, vuint8m8_t b, size_t vl) {
        return __riscv_vand_vv_u8m8(a, b, vl);
    }
};

struct OrOp {
    static vuint8m8_t apply(vuint8m8_t a, vuint8m8_t b, size_t vl) {
        return __riscv_vor_vv_u8m8(a, b, vl);
    }
};

struct XorOp {
    static vuint8m8_t apply(vuint8m8_t a, vuint8m8_t b, size_t vl) {
        return __riscv_vxor_vv_u8m8(a, b, vl);
    }
};

template<typename Op>
inline int bitwise_binary_op_impl(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                             uchar* dst, size_t step, int width, int height) {
    for (size_t h = 0; h < static_cast<size_t>(height); ++h) {
        const uchar* src1Row = src1 + h * step1;
        const uchar* src2Row = src2 + h * step2;
        uchar* dstRow = dst + h * step;

        size_t vl;
        for (size_t x = 0; x < static_cast<size_t>(width); x += vl) {
            vl = __riscv_vsetvl_e8m8(width - x);
            vuint8m8_t src1_vec = __riscv_vle8_v_u8m8(src1Row + x, vl);
            vuint8m8_t src2_vec = __riscv_vle8_v_u8m8(src2Row + x, vl);
            vuint8m8_t v_res = Op::apply(src1_vec, src2_vec, vl);
            __riscv_vse8_v_u8m8(dstRow + x, v_res, vl);
        }
    }
    return CV_HAL_ERROR_OK;
}

inline int and8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                 uchar* dst, size_t step, int width, int height) {
    return bitwise_binary_op_impl<AndOp>(src1, step1, src2, step2, dst, step, width, height);
}

inline int or8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                uchar* dst, size_t step, int width, int height) {
    return bitwise_binary_op_impl<OrOp>(src1, step1, src2, step2, dst, step, width, height);
}

inline int xor8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                 uchar* dst, size_t step, int width, int height) {
    return bitwise_binary_op_impl<XorOp>(src1, step1, src2, step2, dst, step, width, height);
}

// Bitwise Unary Operation
inline int not8u(const uchar* src1, size_t step1, uchar* dst, size_t step, int width, int height) {
    for (size_t h = 0; h < static_cast<size_t>(height); h++) {
        const uchar *src1Row = src1 + h * step1;
        uchar *dstRow = dst + h * step;

        size_t vl;
        for (size_t x = 0; x < static_cast<size_t>(width); x += vl)
        {
            vl = __riscv_vsetvl_e8m4(width - x);
            vuint8m4_t src1_vec = __riscv_vle8_v_u8m4(src1Row + x, vl);
            vuint8m4_t v_res = __riscv_vnot_v_u8m4(src1_vec, vl);
            __riscv_vse8_v_u8m4(dstRow + x, v_res, vl);
        }
    }
    return CV_HAL_ERROR_OK;
}

}}

#endif
