# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, List

import torch

import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

# from model import Transformer  # pyre-ignore[21]
from torch.utils._pytree import tree_flatten, tree_unflatten

from .utils import TORCH_VERSION_AFTER_2_3
from typing import Any, Dict, Tuple, Optional
from .unified import Quantizer
from functools import reduce
from math import gcd

aten = torch.ops.aten

## eval.py ##

try:
    import lm_eval  # pyre-ignore[21]  # noqa: F401

    lm_eval_available = True
except:
    lm_eval_available = False

if lm_eval_available:
    try:  # lm_eval version 0.4
        from lm_eval.evaluator import evaluate  # pyre-ignore[21]
        from lm_eval.models.huggingface import HFLM as eval_wrapper  # pyre-ignore[21]
        from lm_eval.tasks import get_task_dict  # pyre-ignore[21]
    except:  # lm_eval version 0.3
        from lm_eval import base, evaluator, tasks

        eval_wrapper = base.BaseLM
        get_task_dict = tasks.get_task_dict
        evaluate = evaluator.evaluate
else:
    logging.info("lm_eval is not installed, GPTQ may not be usable")

if lm_eval_available:
    class InputRecorder(eval_wrapper):
        """
        This is a fake evaluation wrapper from the lm_eval library that just records the inputs
        so that they can be used in calibration.

        If pad_calibration_inputs is enabled, the input recorder will take
        each input and pad/truncate it down to the calibration_seq_length.
        (if using padding you should set the embeddings for the pad_token to 0
        in the model)

        Note: after padding/truncation, input_prep_function is called to bring
        it to the proper form to be inserted into a given model.

        If not, it will only truncate inputs to the desired length.
        """

        def __init__(
            self,
            tokenizer,
            calibration_seq_length,
            input_prep_func=None,
            pad_calibration_inputs=False,
            vocab_size=32000,
            pad_token=0,
            device="cpu",
        ):
            super().__init__()
            self._tokenizer = tokenizer
            self._device = torch.device(device)
            self.vocab_size = vocab_size
            self._max_seq_length = calibration_seq_length
            self.calibration_seq_length = calibration_seq_length

            # need to take inps and convert to corrent input
            # for model
            self.input_prep_func = (
                input_prep_func if input_prep_func is not None
                else lambda x: (x,)
            )

            self.pad_calibration_inputs = pad_calibration_inputs
            self.pad_token = pad_token

            self.inputs = None

        @property
        def eot_token_id(self):
            return self._tokenizer.eos_id()

        @property
        def max_length(self):
            return self._max_seq_length

        @property
        def max_gen_toks(self):
            return 50

        @property
        def batch_size(self):
            return 1

        @property
        def device(self):
            return self._device

        def tok_encode(self, string: str, **kwargs):
            # TODO: verify this for multi-batch as well
            tokens = self._tokenizer.encode(string)
            if hasattr(self._tokenizer, "bos_id"):
                tokens = [self._tokenizer.bos_id()] + tokens
            return tokens

        def tok_decode(self, tokens):
            decoded = self._tokenizer.decode(tokens)
            return decoded

        def add_input(self, args):
            if self.inputs is None:
                self.inputs = [MultiInput([arg]) for arg in args]
            else:
                self.inputs = [
                    multi.add_input(arg) for (multi, arg) in zip(self.inputs, args)
                ]

        def record_inputs(
            self,
            calibration_tasks,
            calibration_limit,
        ):
            try:
                lm_eval.tasks.initialize_tasks()
            except:
                pass

            task_dict = get_task_dict(calibration_tasks)
            print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

            evaluate(
                self,
                task_dict,
                limit=calibration_limit,
            )
            return self

        def get_inputs(self):
            return self.inputs

        def _model_call(self, inps):
            inps = inps.squeeze(0)
            T = len(inps)
            if (
                # can't use inputs that are too short when padding disabled
                (T < self.calibration_seq_length and not self.pad_calibration_inputs)
                or
                # can't use inputs that actually use token we use for padding
                (self.pad_calibration_inputs and self.pad_token in inps)
            ):
                # give random output
                return torch.randn(
                    (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
                )

            # pad or truncate to the right size
            if T >= self.calibration_seq_length:
                inps = inps[: self.calibration_seq_length]
            else:
                inps = F.pad(inps, (self.pad_token, self.calibration_seq_length - T))

            inps = inps.unsqueeze(0)
            model_in = self.input_prep_func(inps)

            self.add_input(model_in)

            # output `something` with correct shape to keep eval going
            return torch.randn(
                (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
            )

        def _model_generate(self, context, max_length, eos_token_id):
            raise Exception("unimplemented")


class MultiInput:

    def __init__(self, inputs):

        self.values = list(inputs)

    def add_input(self, input):
        self.values.append(input)
        return self

    def __getitem__(self, slice):
        return MultiInput(self.values[slice])

    def cuda(self):
        self.values = [
            val.cuda() if isinstance(val, torch.Tensor) else val for val in self.values
        ]


class GenericGPTQRunner(fx.Interpreter):
    """
    This is a generic GPTQ runner that takes an existing model and applies GPTQ.
    It uses torch._dynamo.export to obtain a graph of the model and then hooks
    into function calls and when it detects a linear, it applies GPTQ to the weight
    given the calibration of inputs passed in at initialization. It puts the results
    into the state_dict so that the quantized model weights/qparams can be loaded
    directly into the model.

    intended to be used in concert with a GPTQQuantizer class to define the quantization mode.
    """

    def __init__(
        self,
        model,
        inputs: MultiInput,
        blocksize=128,
        percdamp=0.01,
        groupsize=128,
    ):

        self.id_to_name = {
            id(value): name for name, value in dict(model.named_parameters()).items()
        }

        # trace model for one input
        one_input = [multi.values[0].cpu() for multi in inputs]  # pyre-ignore[16]
        exported_model = torch._dynamo.export(
            model.cpu(), aten_graph=True, pre_dispatch=True, tracing_mode="fake"
        )(*one_input)
        super().__init__(exported_model.graph_module)

        self.new_state_dict = model.state_dict()

        self.blocksize = blocksize

        self.percdamp = percdamp

        self.groupsize = groupsize
        self.inputs = inputs
        self.gptq_done = False
        self.debug = True

    def configure_quantization_mode(
        self,
        get_qparams_func,
        quantize_func,
        dequantize_func,
        combine_qparams_list_func,
        make_names_and_values_dict_func,
        skip_layer_func,
        act_fake_quant_func = None,
    ):
        # these functions need to already be curried with all inputs other than weight, qparams

        self.get_qparams_func = (
            get_qparams_func  # accepts [2d weight tensor], outputs qparams.
        )

        self.quantize_func = quantize_func  # accepts [2d weight tensor], [qparams], outputs a 2d quantized tensor of desired dtype

        self.dequantize_func = dequantize_func
        # accepts [quantized] tensor and [qparams], outputs a 2d dequantized tensor of type float,
        # assumes this output .to(w_orig_dtype) is ~eventual desired dequant behavior

        #  `combine_qparams_list_func`.
        self.combine_qparams_list_func = combine_qparams_list_func
        # accepts [`list` of qparams] from quantizing one group at a time,
        # outputs a qparams object that could be passed into quant/dequantize_func

        self.skip_layer_func = skip_layer_func  # accepts [weight tensor], outputs a bool on whether or not to apply gptq to this layer

        #  `make_names_and_values_dict_func`.
        self.make_names_and_values_dict_func = make_names_and_values_dict_func  # accepts [2d quantized tensor], [qparams], returns a dict of names, values to put in state_dict
        # note any final packing for storage should happen here

        # `act_fake_quant_func`
        self.act_fake_quant_func = act_fake_quant_func # accepts [activation tensor], returns a fake-quantized activation tensor
        return self

    def run(self):
        assert (
            self.get_qparams_func is not None
        ), "need to configure quantization mode before running"
        self.gptq_done = True
        super().run(*self.inputs)

    def get_quantized_state_dict(self):
        assert (
            self.gptq_done
        ), "need to run GPTQRunner before you can get_quantized_state_dict"
        quantized_state_dict = self.new_state_dict
        # Don't want to store/load the kv_cache so remove it from the state_dict
        del_list = []
        for param_fqn in quantized_state_dict:
            if "kv_cache" in param_fqn:
                del_list.append(param_fqn)
        for param_fqn in del_list:
            quantized_state_dict.pop(param_fqn)
        return quantized_state_dict

    def call_function(self, target, args, kwargs, skip_quant=False):  # noqa: C901

        def tensors_to_cuda(args):
            new_args = []
            for x in args:
                new_args.append(x.cuda() if isinstance(x, torch.Tensor) else x)
            return new_args

        # flatten args and kwargs together
        flat_args, spec = tree_flatten((args, kwargs))
        # move all single tensors to cuda, will move MultiInputs to cuda one at a time
        flat_args = tensors_to_cuda(flat_args)

        has_multi_input = MultiInput in [type(x) for x in flat_args]
        if has_multi_input:
            # Just some trickery to convert
            # [MultiInput[a, a, a], MultiInput(b, b, b)] => [a, b], [a, b], [a, b]
            multi_input_count = max(
                [len(x.values) if isinstance(x, MultiInput) else 1 for x in flat_args]
            )
            transposed_args = list(
                zip(
                    *[
                        (
                            x.values
                            if isinstance(x, MultiInput)
                            else [x] * multi_input_count
                        )
                        for x in flat_args
                    ]
                )
            )
        else:
            transposed_args = [flat_args]
        outputs = []

        # check whether we apply GPTQ to this module
        quantize_linear = (
            (target == aten.linear.default)  # if its a linear
            and id(args[1]) in self.id_to_name  # and if we know the layer name
            and not skip_quant  # and if we weren't told to skip quantization
            # and if the skip_layer_func doesn't say we should skip
            and not (self.skip_layer_func is not None and self.skip_layer_func(args[1]))
        )  # then we will quantize this linear layer/weight

        if quantize_linear:  # instantiate variables for GPTQ
            H = 0
            total_batches = 0

        for inp in transposed_args:
            inp = tensors_to_cuda(inp)
            cur_args, cur_kwargs = tree_unflatten(inp, spec)

            if (
                quantize_linear
            ):  # calculate H instead of output (will run the linear eventually with updated weight)
                x = cur_args[0].float()
                if self.act_fake_quant_func is not None:
                    x = self.act_fake_quant_func(x)
                shape = x.shape
                n = 1 if len(shape) == 2 else shape[0]
                H *= total_batches / (total_batches + n)
                total_batches += n
                x = ((2 / total_batches) ** (1 / 2)) * x.reshape(
                    -1, shape[-1]
                ).t().float()
                H += x.matmul(x.t())
            else:
                # get output if its not a linear
                out = super().call_function(target, cur_args, cur_kwargs)
                # if isinstance(out, torch.Tensor) and (out.isnan().max() or out.sum()==0 or out.isinf().max()):
                #     breakpoint()
                if isinstance(out, torch.Tensor):
                    outputs.append(out.cpu())
                else:
                    outputs.append(out)

        if quantize_linear:
            mod_fqn = ".".join(self.id_to_name[id(args[1])].split(".")[:-1])

            W = args[1].to(H.device)

            Q, DQ, qparams = self.faster_quant(H, W.detach())
            print(mod_fqn)

            #  `make_names_and_values_dict_func`.
            names_and_values_dict = self.make_names_and_values_dict_func(Q, qparams)

            # delete old weight
            if mod_fqn + ".weight" in self.new_state_dict:
                self.new_state_dict.pop(mod_fqn + ".weight")
            if len(args) > 2:
                self.new_state_dict[mod_fqn + ".bias"] = args[2]
            for name, value in names_and_values_dict.items():
                self.new_state_dict[mod_fqn + "." + name] = value

            # run linear with new weight to get corrected output
            new_out = self.call_function(
                target, (args[0], DQ, *args[2:]), kwargs, skip_quant=True
            )

            if self.debug:
                old_out = self.call_function(
                    target, (args[0][:2], args[1], *args[2:]), kwargs, skip_quant=True
                )

                def SQNR(x, y):
                    # TODO: Use of deprecated function torch.norm
                    return 20 * torch.log10(
                        torch.linalg.norm(x) / torch.linalg.norm(x - y)
                    )

                #  `dequantize_func`.
                DQ_after = self.dequantize_func(Q, qparams).to(W.dtype)
                print(
                    "SQNR for QDQ (this should be inf)", SQNR(DQ, DQ_after)
                )  # matches
                print(
                    "SQNR for weight (can be low)", SQNR(W, DQ.cuda())
                )  # fine to not match
                print(
                    "SQNR for output with GPTQ (hopefully 35+)",
                    torch.cat(
                        [
                            SQNR(old.cpu(), new.cpu()).unsqueeze(0)
                            for (old, new) in zip(old_out.values, new_out.values[:2])
                        ]
                    ).mean(),
                )

                #  `get_qparams_func`.
                qparams2 = self.get_qparams_func(W)

                Q2 = self.quantize_func(W, qparams2)
                DQ2 = self.dequantize_func(Q2, qparams2).to(W.dtype)
                old_q_out = self.call_function(
                    target, (args[0][:2], DQ2, *args[2:]), kwargs, skip_quant=True
                )

                print(
                    "SQNR for output without GPTQ (should be less than above)",
                    torch.cat(
                        [
                            SQNR(old.cpu(), old_q.cpu()).unsqueeze(0)
                            for (old, old_q) in zip(old_out.values, old_q_out.values)
                        ]
                    ).mean(),
                )
            return new_out

        return MultiInput(outputs) if has_multi_input else outputs[0]

    def faster_quant(self, H, W):
        percdamp = self.percdamp
        blocksize = self.blocksize
        groupsize = self.groupsize
        orig_dtype = W.dtype
        W = W.detach().float()
        _, columns = W.shape[0], W.shape[1]
        device = W.device

        if groupsize == -1:

            cur_qparams = self.get_qparams_func(W)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        DQ = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        all_qparams = []
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            DQ1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:  # start of new group
                    cur_qparams = self.get_qparams_func(
                        W[:, (i1 + i) : (i1 + i + groupsize)]
                    )
                    all_qparams.append(cur_qparams)

                q = self.quantize_func(w.unsqueeze(1), cur_qparams).flatten()

                #  `dequantize_func`.

                dq = self.dequantize_func(q.unsqueeze(1), cur_qparams).flatten()

                DQ1[:, i] = dq
                Losses1[:, i] = (w - dq) ** 2 / d**2

                err1 = (w - dq) / d
                W1[:, i:] -= (
                    err1.to(Hinv1.dtype).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                )
                Err1[:, i] = err1

            DQ[:, i1:i2] = DQ1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.to(Hinv.dtype).matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if all_qparams == []:

            all_qparams.append(cur_qparams)

        # convert a list of qparams objects into a single one. enerally by
        # concatenating a bunch of n,1 scale/zeros tensors into a n,num_groups tensor

        #  `combine_qparams_list_func`.
        all_qparams = self.combine_qparams_list_func(all_qparams)
        Q = self.quantize_func(DQ, all_qparams)
        return Q, DQ.to(orig_dtype), all_qparams


if TORCH_VERSION_AFTER_2_3:
    from .quant_primitives import (
        get_group_qparams_symmetric,
        group_quantize_tensor_symmetric,
        per_token_dynamic_quant,
    )

    class GPTQQuantizer(Quantizer):
        """
        This class implements a GPTQ Quantizer that can be used to apply GPTQ to a model in concert with the GenericGPTQRunner class.
        Unlike the base Quantizer class, the user does not need to implement the create_quantized_state_dict, instead they have to reimplement
        __init__ such that it defines the functions for the quantization mode. User is expected to reimplement convert_for_runtime.

        The following functions (which must be defined in __init__) are used to define the quantization mode for both GPTQ and
        create_quantized_state_dict. Here is a description of each function.

        get_qparams_func:
            A function that calculates the quantization qparams for an input tensor.
            Args:
                weight: A 2d weight tensor with non-integer dtype.
            Returns:
                qparams: it can have any format but will need to be handled by the other defined functions below.

        quantize_func:
            A function that applies quantization to an input tensor. It should be noted
            that this function needs to be able to handle quantizing the entire weight tensor, a single group,
            or a single column.
            Args:
                weight: A 2d weight tensor with non-integer dtype.
                qparams: the output from get_qparams_func
            Returns:
                quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)


        dequantize_func:
            A function that dequantizes an input quantized weight tensor. It should be noted
            that this function needs to be able to handle dequantizing the entire weight tensor, a single group,
            or a single column.
            Args:
                quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
                qparams: the output from get_qparams_func
            Returns:
                weight: A 2d weight tensor with non-integer dtype.

        act_fake_quant_func (optional):
             A function that (dynamically) quantizes activation to input
             Args:
                 input: input Tensor in f32/bf16/f16
             Returns:
                 output: dynamically quantized and dequantized Tensor (with the same dtype as input)

        combine_qparams_list_func:
            A function that combines several qparams into one qparam.
            Args:
                qparams_list: a list of qparams objects, each obtained by calling get_qparams_func
                on a single group from a weight tensor
            Returns:
                qparams: an object of the same format as the qparams above.

        skip_layer_func:
            A function that determines which linear layers should be skipped during GPTQ
            Args:
                weight: A 2d weight tensor with non-integer dtype.
            Returns:
                skip: boolean indicating whether layer should be skipped

        make_names_and_values_dict_func:
            A function that prepares the qparams and quantized_weight and creates a dictionary indicating how they
            should be inserted into the state_dict. Generally any packing of the weight and qparams should be done here.
            Args:
                quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
                qparams: the output from get_qparams_func
            Returns:
                names_and_values_dict: a dictionary mapping the name of the parameters of the quantized module to the
                corresponding quantized weights and qparams.
        """

        def __init__(self):

            assert self.get_qparams_func is not None

            assert self.quantize_func is not None

            assert self.dequantize_func is not None

            assert self.combine_qparams_list_func is not None

            #  `make_names_and_values_dict_func`.
            assert self.make_names_and_values_dict_func is not None

        @torch.no_grad()
        def _create_quantized_state_dict(
            self,
            model,
            inputs,
            blocksize,
            percdamp,
            groupsize,
            #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
        ) -> Dict:
            print("Tracing model for GPTQ")
            GPTQ_runner = GenericGPTQRunner(
                model,
                inputs,
                blocksize,
                percdamp,
                groupsize,
            ).configure_quantization_mode(
                self.get_qparams_func,  # pyre-ignore[16]
                self.quantize_func,  # pyre-ignore[16]
                self.dequantize_func,  # pyre-ignore[16]
                self.combine_qparams_list_func,  # pyre-ignore[16]
                self.make_names_and_values_dict_func,  # pyre-ignore[16]
                self.skip_layer_func,  # pyre-ignore[16]
                self.act_fake_quant_func if hasattr(self, "act_fake_quant_func") else None,  # pyre-ignore[16]
            )
            print("Applying GPTQ to weights")
            GPTQ_runner.run()
            return GPTQ_runner.get_quantized_state_dict()

        def _convert_for_runtime(self, model: torch.nn.Module) -> "nn.Module":
            raise NotImplementedError("_convert_for_runtime not implemented")

        @torch.no_grad()
        def quantize(self, model: torch.nn.Module, inputs: List[MultiInput], **kwargs: Any) -> torch.nn.Module:
            pass


    def linear_forward_8da4w(
        x,
        weight_int8,
        scales,
        zeros,
        out_features,
        groupsize,
        precision,
    ):
        x = per_token_dynamic_quant(x)
        # TODO: verify and remove following reshape code
        # origin_x_size = x.size()
        # x = x.reshape(-1, origin_x_size[-1])

        # TODO: better API
        # weight_int8 = torch.ops.quantized_decomposed.unpack_int4_to_int8(weight_int4packed)
        n_bit = 4
        quant_min = -(2 ** (n_bit - 1))
        quant_max = 2 ** (n_bit - 1) - 1
        w_dq = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            weight_int8,
            scales,
            zeros,
            quant_min,
            quant_max,
            torch.int8,
            groupsize,
            precision,
        )

        # x = x.to(torch.float16)
        # w_dq = w_dq.to(torch.float16)
        c = torch.nn.functional.linear(x, w_dq)

        # new_shape = origin_x_size[:-1] + (out_features,)
        # c = c.reshape(new_shape)

        return c


    class WeightOnlyInt4Linear(torch.nn.Module):
        __constants__ = ['in_features', 'out_features']
        in_features: int
        out_features: int
        weight: torch.Tensor

        def __init__(
                self, in_features: int, out_features: int,
                bias=True, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8, use_cuda=True,
        ) -> None:
            super().__init__()
            self.padding = _check_linear_int4_k(in_features, groupsize, inner_k_tiles)
            if self.padding:
                from model import find_multiple
                self.origin_in_features = in_features
                in_features = find_multiple(in_features, 1024)

            self.in_features = in_features
            self.out_features = out_features
            assert not bias, "require bias=False"
            self.groupsize = groupsize
            self.inner_k_tiles = inner_k_tiles

            assert out_features % 8 == 0, "require out_features % 8 == 0"
            assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
            if use_cuda:
                self.register_buffer(
                    "weight",
                    torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
                )
            else:
                self.register_buffer(
                    "weight",
                    torch.empty((out_features, in_features // 2), dtype=torch.uint8)
                )
            self.register_buffer(
                "scales_and_zeros",
                torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            input = input.to(torch.bfloat16)
            if self.padding:
                import torch.nn.functional as F
                input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
            return linear_forward_int4(
                input,
                self.weight, self.scales_and_zeros, self.out_features, self.groupsize
            )


    class Int8DynActInt4WeightLinear(torch.nn.Module):
        __constants__ = ["in_features", "out_features"]

        in_features: int
        out_features: int
        weight: torch.Tensor

        """
        This module implements a dynamic quantized linear layer with int4 weight.
        Weights are per channel groupwise quantized. Parameters of importance
        groupsize: the number of elements in each quantized group
        precision: precision of input and output. e.g. torch.float32 means input
        activation is float32 and output is float32.
        scales_precision: precision of per group scale.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias=True,
            device=None,
            dtype=None,
            groupsize: int = 256,
            precision: torch.dtype = torch.float32,
            scales_precision: torch.dtype = torch.float32,
        ) -> None:
            super().__init__()
            # always pad if needed since it becomes a noop at runtime if not needed
            # self.origin_in_features = in_features
            assert (
                in_features % groupsize == 0
            ), f"require in_features:{in_features} % groupsize:{groupsize} == 0"
            # in_features = _calc_padded_size_linear_int4(
            #    in_features, groupsize
            # )
            self.in_features = in_features
            self.out_features = out_features
            assert not bias, "require bias=False"
            # TODO: align groupsize naming
            self.groupsize = groupsize
            # Precision of the activation which also indicates
            # output precision of the dynamically quantized linear layer
            # that his module represents.
            self.precision = precision

            # currently storing unpacked int8 weights
            self.register_buffer(
                "weight",
                torch.empty((out_features, in_features), dtype=torch.int8),
            )
            self.register_buffer(
                "scales",
                torch.empty(
                    (out_features, in_features // groupsize),
                    dtype=scales_precision,
                ),
            )
            self.register_buffer(
                "zeros",
                torch.empty(
                    (out_features, in_features // groupsize),
                    dtype=scales_precision,
                ),
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            input = input.to(self.precision)
            # padding is removed for perf
            # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
            return linear_forward_8da4w(
                input,
                self.weight,
                self.scales,
                self.zeros,
                self.out_features,
                self.groupsize,
                self.precision,
            )


    def find_multiple(n: int, *args: Tuple[int]) -> int:
        k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
        if n % k == 0:
            return n
        return n + k - (n % k)

    def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = None):
        k_divisible_by_groupsize = k % groupsize == 0
        if inner_k_tiles is not None:
            k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
            return k_divisible_by_groupsize and k_divisible_by_16_times_inner_k_tiles
        return k_divisible_by_groupsize

    def _calc_padded_size_linear_int4(k, groupsize=1):
        return find_multiple(k, groupsize)

    def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
        origin_x_size = x.size()
        x = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
        new_shape = origin_x_size[:-1] + (out_features,)
        c = c.reshape(new_shape)
        return c

    def pack_scales_and_zeros(scales, zeros, precision=torch.float32):
        assert scales.shape == zeros.shape
        assert scales.dtype == precision
        assert zeros.dtype == precision
        return (
            torch.cat(
                [
                    scales.reshape(scales.size(0), scales.size(1), 1),
                    zeros.reshape(zeros.size(0), zeros.size(1), 1),
                ],
                2,
            )
            .transpose(0, 1)
            .contiguous()
        )

    def unpack_scales_and_zeros(scales_and_zeros):
        assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
        assert scales_and_zeros.dtype == torch.float
        return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)

    def replace_linear_int4(module, groupsize, inner_k_tiles, padding_allowed, use_cuda):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles) or padding_allowed:
                    setattr(module, name, WeightOnlyInt4Linear(
                        child.in_features, child.out_features, bias=False,
                        groupsize=groupsize, inner_k_tiles=inner_k_tiles, use_cuda=use_cuda
                    ))
            else:
                replace_linear_int4(child, groupsize, inner_k_tiles, padding_allowed, use_cuda)

    def replace_linear_8da4w(
        module,
        groupsize,
        padding_allowed,
        precision,
        scales_precision,
    ):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if _check_linear_int4_k(child.in_features, groupsize) or padding_allowed:
                    setattr(
                        module,
                        name,
                        Int8DynActInt4WeightLinear(
                            child.in_features,
                            child.out_features,
                            bias=False,
                            groupsize=groupsize,
                            precision=precision,
                            scales_precision=scales_precision,
                        ),
                    )
            else:
                replace_linear_8da4w(
                    child,
                    groupsize,
                    padding_allowed,
                    precision,
                    scales_precision,
                )

    def pack_scales_and_zeros(scales, zeros):
        assert scales.shape == zeros.shape
        assert scales.dtype == torch.bfloat16
        assert zeros.dtype == torch.bfloat16
        return (
            torch.cat(
                [
                    scales.reshape(scales.size(0), scales.size(1), 1),
                    zeros.reshape(zeros.size(0), zeros.size(1), 1),
                ],
                2,
            )
            .transpose(0, 1)
            .contiguous()
        )


    class Int8DynActInt4WeightQuantizer(Quantizer):
        def __init__(
            self,
            groupsize: int = 256,
            padding_allowed: bool = False,
            precision: torch.dtype = torch.float32,
            scales_precision: torch.dtype = torch.float32,
            inner_k_tiles: Optional[int] = None,
            _is_gpt_fast: bool = False,
            _use_cuda: bool = True,
        ) -> None:
            super().__init__()
            if _is_gpt_fast:
                assert inner_k_tiles in [2, 4, 8]
                assert groupsize in [32, 64, 128, 256]
            else:
                assert inner_k_tiles is None
            self._is_gpt_fast = _is_gpt_fast
            self._use_cuda = _use_cuda
            self.inner_k_tiles = inner_k_tiles
            self.groupsize: int = groupsize
            self.padding_allowed: bool = padding_allowed
            self.precision: torch.dtype = precision
            self.scales_precision: torch.dtype = scales_precision

        @torch.no_grad()
        def _create_quantized_state_dict(
            self, model: torch.nn.Module
        ) -> Dict[str, torch.Tensor]:
            cur_state_dict = model.state_dict()
            for fqn, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    assert not mod.bias
                    out_features = mod.out_features
                    in_features = mod.in_features
                    # assert out_features % 8 == 0, "require out_features % 8 == 0"
                    print(f"linear: {fqn}, in={in_features}, out={out_features}")

                    assert (
                        in_features % self.groupsize == 0
                    ), f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"

                    weight = mod.weight.data
                    if not _check_linear_int4_k(
                        in_features, self.groupsize, self.inner_k_tiles
                    ):
                        if self.padding_allowed:
                            from model import find_multiple
                            import torch.nn.functional as F
                            print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                            padded_in_features = find_multiple(in_features, 1024)
                            weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                        else:
                            print(f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
                                  "and that groupsize and inner_k_tiles*16 evenly divide into it")
                            continue
                    (
                        weight_int8,
                        scales,
                        zeros,
                    ) = group_quantize_tensor_symmetric(
                        weight.to(self.precision),
                        4,  # n_bit
                        self.groupsize,
                        self.scales_precision,
                    )
                    if self._is_gpt_fast:
                        weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int8.to(torch.int32), self.inner_k_tiles)
                        scales_and_zeros = pack_scales_and_zeros(scales, zeros)
                        cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                        cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")
                    else:
                        cur_state_dict[f"{fqn}.weight"] = weight_int8.to("cpu")
                        cur_state_dict[f"{fqn}.scales"] = scales.to("cpu")
                        cur_state_dict[f"{fqn}.zeros"] = zeros.to("cpu")
                    # TODO: support bias?

            return cur_state_dict

        def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
            if self._is_gpt_fast:
                # TODO: temporary path for gpt-fast, will remove later
                replace_linear_int4(
                    model,
                    self.groupsize,
                    self.inner_k_tiles,
                    self.padding_allowed,
                    self._use_cuda,
                )
            else:
                replace_linear_8da4w(
                    model,
                    self.groupsize,
                    self.padding_allowed,
                    self.precision,
                    self.precision,
                )
            return model

        def quantize(
            self, model: torch.nn.Module, *args: Any, **kwargs: Any
        ) -> torch.nn.Module:
            state_dict = self._create_quantized_state_dict(model)
            model = self._convert_for_runtime(model)
            # TODO: make it strict
            model.load_state_dict(state_dict, strict=False)
            return model


    # TODO: consolidate with other quantizers
    class Int4WeightQuantizer(Quantizer):
        def __init__(
            self,
            groupsize: int = 256,
            padding_allowed: bool = False,
            precision: torch.dtype = torch.float32,
            inner_k_tiles: Optional[int] = None,
            _use_cuda: bool = True,
        ) -> None:
            super().__init__(
                groupsize,
                padding_allowed,
                precision,
                torch.float32,  # scales_precision
                inner_k_tiles,
                True,  # _is_gpt_fast
                _use_cuda,
            )


    class Int8DynActInt4WeightGPTQQuantizer(GPTQQuantizer):
        def __init__(
            self,
            blocksize,
            percdamp,
            groupsize,
            inner_k_tiles=8,
            padding_allowed=True,
            precision=torch.float32,
            _is_gpt_fast=False,
            _use_cuda=True,
        ):
            self._is_gpt_fast = _is_gpt_fast
            self._use_cuda = _use_cuda
            self.blocksize = blocksize
            self.percdamp = percdamp
            self.groupsize = groupsize
            self.inner_k_tiles = inner_k_tiles
            self.padding_allowed = padding_allowed
            self.precision = precision

            self.act_fake_quant_func = per_token_dynamic_quant
            n_bit = 4
            self.get_qparams_func = lambda w: get_group_qparams_symmetric(
                w, n_bit, groupsize, self.precision
            )
            quant_min = -(2 ** (n_bit - 1))
            quant_max = 2 ** (n_bit - 1) - 1

            self.quantize_func = lambda w, qparams: torch.ops.quantized_decomposed.quantize_per_channel_group(
                w, qparams[0], qparams[1], quant_min, quant_max, torch.int8, groupsize
            )

            self.dequantize_func = lambda q, qparams: torch.ops.quantized_decomposed.dequantize_per_channel_group(
                q,
                qparams[0],
                qparams[1],
                quant_min,
                quant_max,
                torch.int8,
                groupsize,
                self.precision,
            )

            self.combine_qparams_list_func = lambda qparams_list: [
                torch.cat(x, dim=1) for x in zip(*qparams_list)
            ]
            # skip unless padding_allowed=True or its correctly sized

            self.skip_layer_func = lambda linear_weight: not (
                _check_linear_int4_k(linear_weight.shape[-1], groupsize) or padding_allowed
            )

            # we need to do the padding here, both for q and the qparams if necessary

            # TODO: this is the gpt-fast version, merge with the main version later
            def make_names_and_values_dict_func_gpt_fast(q, qparams):
                k = q.shape[1]
                new_k = find_multiple(k, 1024)
                # how much we need to pad the weight
                delta_k = new_k - q.shape[1]
                q = q.to(torch.int32)
                final_q = torch.ops.aten._convert_weight_to_int4pack(F.pad(q, pad=(0, delta_k)), inner_k_tiles)
                scales = qparams[0].to(torch.bfloat16)
                zeros = qparams[1].to(torch.bfloat16)
                scales_and_zeros = pack_scales_and_zeros(scales, zeros)
                # how many new groups we need for padded weight
                delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
                final_s_and_z = F.pad(scales_and_zeros, pad=(0,0,0,0,0, delta_groups), value=1)
                return {"weight": final_q, "scales_and_zeros": final_s_and_z}

            def make_names_and_values_dict_func(q, qparams):
                k = q.shape[1]
                new_k = _calc_padded_size_linear_int4(k, groupsize)
                # how much we need to pad the weight
                delta_k = new_k - q.shape[1]
                final_q = F.pad(q, pad=(0, delta_k))
                scales = qparams[0].to(self.precision)
                zeros = qparams[1].to(self.precision)
                return {"weight": final_q, "scales": scales, "zeros": zeros}

            self.make_names_and_values_dict_func = make_names_and_values_dict_func_gpt_fast if self._is_gpt_fast else make_names_and_values_dict_func
            super().__init__()

        def _convert_for_runtime(self, model):
            if self._is_gpt_fast:
                # TODO: temporary path for gpt-fast, will remove later
                replace_linear_int4(
                    model,
                    self.groupsize,
                    self.inner_k_tiles,
                    self.padding_allowed,
                    self._use_cuda,
                )
            else:
                replace_linear_8da4w(
                    model,
                    self.groupsize,
                    self.padding_allowed,
                    self.precision,
                    self.precision,
                )
            return model

        def quantize(self, model: torch.nn.Module, inputs: List[MultiInput], **kwargs: Any) -> torch.nn.Module:
            state_dict = self._create_quantized_state_dict(
                model,
                inputs,
                self.blocksize,
                self.percdamp,
                self.groupsize,
            )
            model = self._convert_for_runtime(model)
            model.load_state_dict(state_dict, strict=False)
            return model


    # TODO: consolidate with other quantizers
    class Int4WeightGPTQQuantizer(Int8DynActInt4WeightGPTQQuantizer):

        def __init__(
            self,
            tokenizer,
            blocksize,
            percdamp,
            groupsize,
            calibration_tasks,
            calibration_limit,
            calibration_seq_length,
            pad_calibration_inputs,
            inner_k_tiles=8,
            padding_allowed=True,
            precision=torch.float32,
            _use_cuda=True,
        ):
            super().__init__(
                tokenizer,
                blocksize,
                percdamp,
                groupsize,
                calibration_tasks,
                calibration_limit,
                calibration_seq_length,
                pad_calibration_inputs,
                inner_k_tiles=8,
                padding_allowed=True,
                precision=torch.float32,
                _is_gpt_fast=True,
                _use_cuda=_use_cuda,
            )