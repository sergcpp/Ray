Small pathtracing lib created for learning purposes. Includes CPU and GPU (OpenCL) backends.
~~CPU backends dragging behind and do not include any shading.~~ Its fine now, but should be optimized.


- Full application : https://bitbucket.org/Apfel1994/raydemo
- Video : https://www.youtube.com/watch?v=MHk9jXcdrZs

![Screenshot](img1.jpg)|![Screenshot](img2.jpg)|![Screenshot](img3.jpg)
:-------------------------:|:-------------------------:|:-------------------------:

- Uses plucker test for intersection with precomputed data per triangle as described in 'Ray-Triangle Intersection Algorithm for Modern CPU Architectures' paper.
- Uses SAH-based BVH with stackless traversal as described in 'Efficient Stack-less BVH Traversal for Ray Tracing' paper.
- Uses ray differentials for choosing mip level and filter kernel as described in 'Tracing Ray Differentials' paper.
- Textures are packed in 2d texture array atlas for easier passing to OpenCL kernel.
- Halton sequence is used for sampling.
