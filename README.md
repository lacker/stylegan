Notes just on this fork, go see Nvidia's repository if you want to read the original readme.

I'm using:

* Ubuntu 18 LTS
* Python 3.6.10, via conda
* Tensorflow 1.15
* cuda 10.0
* cudnn 7.6.5

Since the dependencies are between python libraries and c libraries,
dependency tools generally don't manage them correctly, and you
have to worry about minor version compatibility yourself. Prepare for
a higher annoyance level than usual. You have to join the Nvidia
developer program, for example.
