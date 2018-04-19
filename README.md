April 19, 2018

TensorFlow.js was recently released. It is well engineered, provides an
autograd-style interface to backprop, and has committed to supporting Node.
This satisfies our requirements. It is counterproductive to pursue a parallel
effort. Thus we are abandoning our backprop implementation, TF C binding, and
the TF/DL bridge, which made up the foundation of the Propel library. We intend
to rebase our work on top of TFJS.

Our high-level goal continues to be a productive workflow for scientific
computing in JavaScript. Building on top of TFJS allows us to focus on
higher-level functionality.

We have no release at this time.
