# Import necessary TensorRT and PyCUDA modules
import tensorrt as trt
import pycuda.autoinit

# Create a default logger with a warning level
logger = trt.Logger(trt.Logger.WARNING)

# Define a custom logger class
class MyLogger(trt.ILogger):
    def __init__(self):
        # Initialize the base ILogger class
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        # Print the log message (you can also redirect it to a file or any other output if desired)
        print("msg", msg)
        pass # Custom logging implementation can be placed here

# Use the custom logger
logger = MyLogger()

# Create a TensorRT builder and a network using the logger
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Create an ONNX parser to convert the ONNX model to a TensorRT network using the logger
parser = trt.OnnxParser(network, logger)

# Parse the ONNX model from a file
success = parser.parse_from_file('resnet18.onnx')

# If there are errors during parsing, print them
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

# Check if parsing was successful; if not, handle the error
if not success:
    print("here")
    pass # Error handling code can be placed here

# Create a builder configuration
config = builder.create_builder_config()

# Set a memory pool limit for the workspace
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

# Build the serialized TensorRT network using the builder and the configuration
serialized_engine = builder.build_serialized_network(network, config)

# Write the serialized TensorRT engine to a file for future deployment
with open('resnet18.engine', 'wb') as f:
    f.write(serialized_engine)
