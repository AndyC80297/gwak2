# import logging
# import time


import numpy as np

# import numpy as np

# from tritonclient import grpc as triton

# from hermes.aeriel.serve import serve





# import torch
# from torch.utils.data import IterableDataset

# class MyIterableDataset(IterableDataset):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data

#     def __iter__(self):
#         # Return an iterator over the data
#         for item in self.data:
#             yield item

# # Example usage
# data = [1, 2, 3, 4, 5]  # Your data source torch.randn((3,200,2))
# dataset = MyIterableDataset(data)




import psutil
import socket

from hermes.aeriel.serve import serve
from hermes.aeriel.client import InferenceClient
# from tritonclient import grpc as triton

model_repo_dir = "/home/hongyin.chen/Xperimental/GWAK/export_repos"
# image = "/cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:23.01"
image = "hermes/tritonserver:23.01"

def get_ip_address() -> str:
    """
    Get the local nodes cluster-internal IP address
    """
    for _, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if (
                addr.family == socket.AF_INET
                and not addr.address.startswith("127.")
            ):
                
                # print(addr.address)
                return addr.address


x = np.random.normal(0, 1, (10, 200, 2))
x = x.astype("float32")



serve_context = serve(
    model_repo_dir, 
    image, 
    log_file="/home/hongyin.chen/Xperimental/GWAK/export_log/log.log", 
    wait=False
)


ip = get_ip_address()

address=f"{ip}:8001"
# model_name = "gwak-background"
model_name = "gwak-sine_gaussian_hf"
print(f"{address = }")

import time
with serve_context:

    time.sleep(5)
    client = InferenceClient(address, model_name)

    print("Before Sending data to infering model!")
    breakpoint()
    for i in range(30):
        x = np.random.normal(0, 1, (10, 200, 2))
        x = x.astype("float32")
        client.infer(x)
    result = client.get()
    breakpoint()
    print(result)

# raise ValueError("No valid IP address found")



# from hermes.aeriel.client import InferenceClient





# with serve_context:
    
#     print("The server will be live!")
#     client = triton.InferenceServerClient(address)
#     # breakpoint()
#     # assert client.is_server_live()

#     try:
#         client.is_server_live()
#     except triton.InferenceServerException:
#         print("All done!")

#     print("After assert")
#     breakpoint()
#     client.infer("gwak-background", 30)

# exiting the context will spin down the server









# repo = "/home/hongyin.chen/Xperimental/GWAK/export_repos/gwak-background"


# # breakpoint()

# client = InferenceClient(address, model_name, model_version=1)
#     "http://localhost:1001", 
#     repo
# )

# def infer(
#     client: InferenceClient, sequence: Sequence, postprocessor: Postprocessor
# ):
    
#     client.infer()
    
#     result = client.get()
    
#     background, foreground = result
    
    
# """
# Help on function infer in module hermes.aeriel.client.client:

# infer(
#     self, 
#     x: Union[numpy.ndarray, Dict[str, numpy.ndarray]], 
#     request_id: Optional[int] = None, 
#     sequence_id: Optional[int] = None, 
#     sequence_start: bool = False, 
#     sequence_end: bool = False
# ) -> None

#     Make an asynchronous inference request to the service

#     Use the indicated input or inputs to make an inference
#     request to the model sitting on the inference service.
#     If this model requires multiple inputs or states, `x`
#     should be a dictionary mapping from the name of each
#     input or state to the corresponding value. Otherwise,
#     `x` may be a single numpy array representing the input
#     data for this inference request.

#     Responses from the inference service will be handled
#     in an asynchronous callback thread, with the parsed and
#     postprocessed values placed in this object's `callback_q`.
#     As a simple way to retrieve the values from that queue,
#     consider using the `InferenceClient.get` method.

#     Responses follow the same structure as the inputs: if
#     there are multiple outputs, the response will be a
#     dictionary mapping from output names to values. If there
#     is only a single output, it will be returned as a NumPy
#     array.

#     Args:
#         x:
#             The inputs to the model sitting on the server.
#             If the model has multiple inputs (stateful or
#             stateless), this should be a `dict` mapping from
#             input names to corresopnding NumPy arrays. If
#             the model has only a single input, this may be
#             a single NumPy array containing that input's value.
#         request_id:
#             An identifier to associate with this inference request.
#             Will be passed along to the `callback` specified
#             at initialization, or if this is `None` it will
#             be placed in the `callback_q` alongside the response
#             values.
#         sequence_id:
#             An identifier to associate this request with a particular
#             state on the inference server. Required if the
#             indicated model has any stateful inputs, otherwise
#             won't do anything.
#         sequence_start:
#             Indicates whether this request is the first in a
#             new sequence. Won't do anything if the
#             model has no stateful inputs.
#         sequence_end:
#             Indicates whether this request is the final one
#             in a sequence. Won't do anything if the model has
#             no stateful inputs.
# """


# x = [
#     [3,3,3], 
#     [2,2,2],
#     [1,1,1]
# ]



# print(x.shape)





