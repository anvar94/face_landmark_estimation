# Importing necessary libraries
import argparse
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import numpy
import pycuda.autoinit
import torchvision.transforms as transforms
import torch
# from functions import *  # Appears to be importing other helper functions, but it's commented out
from PIL import Image

# Setting the TensorRT logger to INFO level to get detailed logs
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Initialize TensorRT plugins
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def get_engine(engine_file_path):
    """
    Load the TensorRT engine from a serialized engine file.
    """
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """
    Allocate memory for network inputs/outputs including both host and device memory.
    Returns allocated memory.
    """

    # Internal class to hold memory pointers
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            """
            host_mem: CPU memory
            device_mem: GPU memory
            """
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    inputs = []   # List to store input memory
    outputs = []  # List to store output memory
    bindings = [] # List of memory bindings
    stream = cuda.Stream() # CUDA stream for async operations

    # For every binding (input/output), allocate memory
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate pinned CPU memory and GPU memory
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        # Store to appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def Do_Inference(context, bindings, inputs, outputs, stream):
    """
    Run inference using given context and memory pointers.
    Returns network outputs.
    """

    # Copy input data to GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Execute the inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy output data back to CPU
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Wait for CUDA stream to finish
    stream.synchronize()

    return [out.host for out in outputs]
