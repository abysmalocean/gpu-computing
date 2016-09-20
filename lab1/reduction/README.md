Data Parallel Reduction

1)  Extract the lab template into your home directory.

2)  Edit the source files vector_reduction.cu and vector_reduction_kernel.cu
    to complete the functionality of the parallel addition reduction on the
    device.  The size of the array is guaranteed to be equal to 512 elements
    for this assignment.

3)  There are two modes of operation for the application.

    No arguments:  The application will create a randomly initialized array to
    process.  After the device kernel is invoked, it will compute
    the correct solution value using the CPU, and compare that solution with
    the device-computed solution.  If it matches (within a certain tolerance),
    if will print out "Test PASSED" to the screen before exiting.

    One argument:  The application will initialize the input array with
    the values found in the file provided as an argument.

    In either case, the program will print out the final result of the CPU and
    GPU computations, and whether or not the comparison passed.

4)  Answer the following questions:

    1.  How many times does your thread block synchronize to reduce the array
        of 512 elements to a single value?

    2. Over the life of the program, how many warps will be adversely affected
       by SIMD divergence?  Remember that SIMD divergence happens when threads
       within a warp take different code paths through the program.

