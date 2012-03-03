# This is an example of pyOpenCL which will sort <arrsize> numbers using ONLY OpenCL
# this does not do any merge functionality in the python code itself.
# 
# Author Joshua Backfield <suroot.chmod777@gmail.com>
import pyopencl as cl
import numpy
import numpy.linalg as la
import random
from array import array
from datetime import datetime

# we do not support greater than 9k array size, not quite sure what the issue is here?
arrsize = 9000
# create the array of numbers
input = array('i')
input.fromlist([random.randint(0, arrsize) for i in range(arrsize)])
# Get the length
len = array('i', [arrsize])

print input

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
# Create the buffers so that we can send them to the OpenCL program
input_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=input)
len_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=len)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, input.buffer_info()[1] * input.itemsize)

# OpenCL program itself
prg = cl.Program(ctx, """

  __kernel int merge(__global const int *arr1, int arr1size, __global const int *arr2, int arr2size,
  __global int *out) {
    int x = 0;
    int y = 0;
    int n = 0;

    while(x < arr1size || y < arr2size) {
      if(x < arr1size && y < arr2size) {
        if(arr1[x] < arr2[y]) {
          out[n] = arr1[x];
          n++,x++;
        } else {
          out[n] = arr2[y];
          n++,y++;
        }
      } else if(x < arr1size) {
        out[n] = arr1[x];
        n++,x++;
      } else if(y < arr2size) {
        out[n] = arr2[y];
        n++,y++;
      }
    }
    return (x + y) - (arr1size + arr2size); 
  }

  __kernel int cpy(__global int *out, __global const int *in, int size) {
    int x = 0;
    for(x = 0; x < size; x++) {
      out[x] = in[x];
    }
    return 1;
  }

  __kernel void mergesort(__global int *a,
  __global int *b, __global int *len)
  {
    int size = 1;
    int indx = get_global_id(0) * 2;
    int i = 0;

    while(size <= *len) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if((indx + size) < *len) {
        i = (indx + size) + (((indx + (size * 2)) > *len) ? *len - (indx + size) : size);
        merge(&a[indx], size, &a[indx + size], ((indx + (size * 2)) > *len) ? *len - (indx + size) : size, &b[indx]);
      } else {
        return;
      }
      size *= 2;
      cpy(&a[indx], &b[indx], size);
      if(indx%(2*size) != 0) {
        return;
      }
    }
    *len = i;
  }
  """).build()

# Execute the mergesort function (we start with len/2 threads)
start = datetime.now()
prg.mergesort(queue, (input.buffer_info()[1]/2,), None, input_buf, dest_buf, len_buf)
end = datetime.now()

print "Took: ", end - start

# Create the arrays that will contain the output
input_mult = array('i')
input_mult.fromlist([0 for i in range(arrsize)])

# Create an array that will contain len_buf
len_out = array('i')
len_out.fromlist([0])

# Copy the data from the OpenCL context to our local buffer
cl.enqueue_copy(queue, input_mult, dest_buf)
cl.enqueue_copy(queue, len_out, len_buf)

# Print everything out
print input_mult
print len_out
