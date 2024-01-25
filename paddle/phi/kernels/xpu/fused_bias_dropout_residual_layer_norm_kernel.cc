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
void FusedBiasDropoutResidualLnKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& residual,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& ln_scale,
    const paddle::optional<DenseTensor>& ln_bias,
    const float dropout_rate,
    const bool is_test,
    const bool dropout_fix_seed,
    const int dropout_seed,
    const std::string& dropout_implementation,
    const float ln_epsilon,
    DenseTensor* y,
    DenseTensor* bias_dropout_residual_out,
    DenseTensor* dropout_mask_out,
    DenseTensor* ln_mean,
    DenseTensor* ln_variance) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;
  using U = float;
  using XPUTypeU = float;

  const XPUTypeT* x_data = reinterpret_cast<const XPUTypeT*>(x.data<T>());
  const XPUTypeT* bias_data =
      bias.get_ptr() == nullptr
          ? nullptr
          : reinterpret_cast<const XPUTypeT*>(bias->data<T>());
  const XPUTypeT* residual_data =
      reinterpret_cast<const XPUTypeT*>(residual.data<T>());
  const XPUTypeT* ln_scale_data =
      ln_scale.get_ptr() == nullptr
          ? nullptr
          : reinterpret_cast<const XPUTypeT*>(ln_scale->data<T>());
  const XPUTypeT* ln_bias_data =
      ln_bias.get_ptr() == nullptr
          ? nullptr
          : reinterpret_cast<const XPUTypeT*>(ln_bias->data<T>());
  XPUTypeT* bias_dropout_residual_out_data =
      reinterpret_cast<XPUTypeT*>(dev_ctx.template Alloc<T>(
          bias_dropout_residual_out,
          bias_dropout_residual_out->numel() * sizeof(T)));
  XPUTypeU* ln_mean_data = reinterpret_cast<XPUTypeU*>(
      dev_ctx.template Alloc<U>(ln_mean, ln_mean->numel() * sizeof(U)));
  XPUTypeU* ln_var_data = reinterpret_cast<XPUTypeU*>(
      dev_ctx.template Alloc<U>(ln_variance, ln_variance->numel() * sizeof(U)));
  XPUTypeT* dropout_mask_out_data =
      dropout_mask_out == nullptr
          ? nullptr
          : reinterpret_cast<XPUTypeT*>(dev_ctx.template Alloc<T>(
                dropout_mask_out, dropout_mask_out->numel() * sizeof(T)));
  XPUTypeT* y_data = reinterpret_cast<XPUTypeT*>(
      dev_ctx.template Alloc<T>(y, y->numel() * sizeof(T)));

  const auto input_x_dims = x.dims();
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

  xpu::Context* xpu_ctx = dev_ctx.x_context();
  if (bias_data == nullptr) {
    int r = xpu::dropout_add_layernorm_v2(xpu_ctx,
                                          x_data,
                                          residual_data,
                                          ln_scale_data,
                                          ln_bias_data,
                                          bias_dropout_residual_out_data,
                                          dropout_mask_out_data,
                                          y_data,
                                          ln_mean_data,
                                          ln_var_data,
                                          xpu_param);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_add_layernorm_v2");
  } else {
    int r = xpu::dropout_add_layernorm(xpu_ctx,
                                       x_data,
                                       residual_data,
                                       ln_scale_data,
                                       ln_bias_data,
                                       bias_dropout_residual_out_data,
                                       dropout_mask_out_data,
                                       y_data,
                                       ln_mean_data,
                                       ln_var_data,
                                       xpu_param,
                                       bias_data);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_add_layernorm");
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_dropout_residual_layer_norm,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasDropoutResidualLnKernel,
                   float,
                   phi::dtype::float16) {}
