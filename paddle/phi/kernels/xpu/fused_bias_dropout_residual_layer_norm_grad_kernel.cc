// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
#include "paddle/phi/kernels/xpu/xpu_fused_common_function.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedBiasDropoutResidualLnGradKernel(
    const Context& dev_ctx,
    const DenseTensor& y_grad,
    const DenseTensor& x,
    const DenseTensor& residual,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& ln_scale,
    const paddle::optional<DenseTensor>& ln_bias,
    const DenseTensor& ln_mean,
    const DenseTensor& ln_variance,
    const DenseTensor& bias_dropout_residual_out,
    const DenseTensor& dropout_mask_out,
    const float dropout_rate,
    const bool is_test,
    const bool dropout_fix_seed,
    const int dropout_seed,
    const std::string& dropout_implementation,
    const float ln_epsilon,
    DenseTensor* x_grad,
    DenseTensor* residual_grad,
    DenseTensor* bias_grad,
    DenseTensor* ln_scale_grad,
    DenseTensor* ln_bias_grad) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;
  using U = float;
  using XPUTypeU = float;

  const XPUTypeT* d_y_data =
      reinterpret_cast<const XPUTypeT*>(y_grad.data<T>());
  const XPUTypeT* ln_scale_data =
      ln_scale.get_ptr() == nullptr
          ? nullptr
          : reinterpret_cast<const XPUTypeT*>(ln_scale->data<T>());
  const XPUTypeT* dropout_mask_out_data =
      reinterpret_cast<const XPUTypeT*>(dropout_mask_out.data<T>());
  const XPUTypeT* bias_dropout_residual_out_data =
      reinterpret_cast<const XPUTypeT*>(bias_dropout_residual_out.data<T>());
  const XPUTypeU* ln_mean_data =
      reinterpret_cast<const XPUTypeU*>(ln_mean.data<U>());
  const XPUTypeU* ln_var_data =
      reinterpret_cast<const XPUTypeU*>(ln_variance.data<U>());

  XPUTypeT* d_x_data = reinterpret_cast<XPUTypeT*>(
      dev_ctx.template Alloc<T>(x_grad, x_grad->numel() * sizeof(T)));
  XPUTypeT* d_residual_data =
      reinterpret_cast<XPUTypeT*>(dev_ctx.template Alloc<T>(
          residual_grad, residual_grad->numel() * sizeof(T)));

  XPUTypeT* d_bias_data =
      bias_grad == nullptr
          ? nullptr
          : reinterpret_cast<XPUTypeT*>(dev_ctx.template Alloc<T>(
                bias_grad, bias_grad->numel() * sizeof(T)));
  XPUTypeT* d_ln_scale_data =
      ln_scale_grad == nullptr
          ? nullptr
          : reinterpret_cast<XPUTypeT*>(dev_ctx.template Alloc<T>(
                ln_scale_grad, ln_scale_grad->numel() * sizeof(T)));
  XPUTypeT* d_ln_bias_data =
      ln_bias_grad == nullptr
          ? nullptr
          : reinterpret_cast<XPUTypeT*>(dev_ctx.template Alloc<T>(
                ln_bias_grad, ln_bias_grad->numel() * sizeof(T)));

  const auto input_x_dims = y_grad.dims();
  int bsz_seq = 1;
  for (int i = 0; i < input_x_dims.size() - 1; i++) {
    bsz_seq *= input_x_dims[i];
  }
  int dim_embed = input_x_dims[input_x_dims.size() - 1];

  xpu::DropoutAddLayernormParam xpu_param;
  xpu_param.is_test = is_test;
  xpu_param.is_upscale_in_train =
      (dropout_implementation == "upscale_in_train");
  xpu_param.dropout_prob = dropout_rate;
  xpu_param.seed_val =
      dropout_fix_seed ? dropout_seed : dev_ctx.GetGenerator()->Random64();
  xpu_param.is_layernorm = true;
  xpu_param.eps = ln_epsilon;
  xpu_param.m = bsz_seq;
  xpu_param.n = dim_embed;

  if (d_ln_bias_data != nullptr) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "FusedBiasDropoutResidualLnKernel not surpport bias"));
  }

  xpu::Context* xpu_ctx = dev_ctx.x_context();

  // d_residual_data is equal with d_bias_dropout_residual_out_data
  int r = xpu::dropout_add_layernorm_grad_v2(xpu_ctx,
                                             bias_dropout_residual_out_data,
                                             dropout_mask_out_data,
                                             d_y_data,
                                             d_x_data,
                                             d_residual_data,
                                             ln_scale_data,
                                             ln_mean_data,
                                             ln_var_data,
                                             d_ln_scale_data,
                                             d_ln_bias_data,
                                             xpu_param);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_add_layernorm_v2");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_dropout_residual_layer_norm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasDropoutResidualLnGradKernel,
                   float,
                   phi::dtype::float16) {}
