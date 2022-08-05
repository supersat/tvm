import tvm
import numpy as np

import pytest
from tvm import tir
from tvm.script import tir as T
from tvm.tir import IndexMap
from .infrastructure import allocate_hexagon_array


@tvm.testing.fixture
def numpy_data_range(dtype):
    if dtype == "int8":
        return (-128, 127)
    elif dtype == "uint8":
        return (0, 255)
    else:
        raise "Unsupported data type"


@tvm.testing.fixture
def tvm_target(target_str):
    if target_str == "hexagon":
        target_hexagon = tvm.target.hexagon("v68")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
    else:
        target = tvm.target.Target(target_str)
    return target


def module_loader(mod, hexagon_session=None):
    if mod.type_key == "hexagon":
        assert hexagon_session != None
        mod = hexagon_session.load_module(mod)
    return mod


def single_dma_primfunc(size, dtype, x_scope, y_scope, tensorize_dma):

    if tensorize_dma:

        @T.prim_func
        def dma_copy_single(x: T.handle, y: T.handle) -> None:
            X = T.match_buffer(x, size, dtype=dtype, scope=x_scope)
            Y = T.match_buffer(y, size, dtype=dtype, scope=y_scope)
            T.evaluate(
                T.tvm_call_packed(
                    "device_api.hexagon.mem_copy_DLTensor",
                    T.tvm_stack_make_array(
                        X.data,
                        T.tvm_stack_make_shape(size, dtype="handle"),
                        0,
                        1,
                        X.dtype,
                        0,
                        dtype="handle",
                    ),
                    T.tvm_stack_make_array(
                        Y.data,
                        T.tvm_stack_make_shape(size, dtype="handle"),
                        0,
                        1,
                        Y.dtype,
                        0,
                        dtype="handle",
                    ),
                    T.cast(size, dtype="int"),
                    dtype="int32",
                )
            )

        return dma_copy_single
    else:

        @T.prim_func
        def dma_copy_single(x: T.handle, y: T.handle) -> None:
            X = T.match_buffer(x, size, dtype=dtype, scope=x_scope)
            Y = T.match_buffer(y, size, dtype=dtype, scope=y_scope)

            for i in T.serial(size):
                with T.block("dma_copy"):
                    vi = T.axis.spatial(size, i)
                    Y[vi] = X[vi]

        return dma_copy_single


class TestSingleDMA:
    target_str = tvm.testing.parameter("hexagon")

    dtype = tvm.testing.parameter("int8")
    size = tvm.testing.parameter(
        # 128,
        # 256,
        # 512,
        # 1024,
        # 2048,
        # 4096,
        # 8192,
        # 16384,
        # 32768,
        # 65536,
        # 131072,
        # 262144,
        524288,
        # 1048576, # TODO(csullivan): Investigate bug with DMA copy for this size.
    )
    x_scope = tvm.testing.parameter("global")
    y_scope = tvm.testing.parameter("global.vtcm")
    tensorize_dma = tvm.testing.parameter(False, True)

    @tvm.testing.fixture
    def dma_ndarrays(self, size, dtype, hexagon_session, numpy_data_range, x_scope, y_scope):
        x = np.random.randint(
            low=numpy_data_range[0], high=numpy_data_range[1], size=size, dtype=dtype
        )
        y = np.random.randint(
            low=numpy_data_range[0], high=numpy_data_range[1], size=size, dtype=dtype
        )

        numpy_tensors = zip([x, y], [x_scope, y_scope])
        arrays = [
            allocate_hexagon_array(hexagon_session.device, data=tensor, mem_scope=scope)
            for tensor, scope in numpy_tensors
        ]
        return arrays

    @tvm.testing.requires_hexagon
    def test_dma(
        self,
        hexagon_session,
        dtype,
        size,
        x_scope,
        y_scope,
        tvm_target,
        dma_ndarrays,
        tensorize_dma,
    ):
        sch = tir.Schedule(single_dma_primfunc(size, dtype, x_scope, y_scope, tensorize_dma))
        if tensorize_dma == False:
            dma_block = sch.get_block("dma_copy")
            (i,) = sch.get_loops(dma_block)
            o, _, io, ii = sch.split(i, factors=[4, None, 2, 128])
            sch.unroll(io)
            sch.parallel(o)
            sch.vectorize(ii)
        print("Vector latency: ", sch.mod["main"].script())
        mod = tvm.build(sch.mod["main"], target=tvm_target)
        mod = module_loader(mod, hexagon_session)

        x, y = dma_ndarrays
        timer = mod.time_evaluator("__tvm_main__", hexagon_session.device, number=100, repeat=2)
        timing_result = timer(x, y)
        print("DMA latency: ", timing_result)
        tvm.testing.assert_allclose(x.numpy(), y.numpy(), atol=1e-4, rtol=1e-4)
