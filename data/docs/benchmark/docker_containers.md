# Docker Containers

Docker packages an application together with its runtime, libraries and configuration into an image. A container is a running instance of that image, isolated from other processes but sharing the host kernel, which makes it far lighter than a virtual machine.

Because the image is identical everywhere, the same container behaves the same on a laptop, in CI and in production. A Dockerfile describes how the image is built layer by layer; ordering layers so that dependencies are installed before the application code is copied makes rebuilds faster through layer caching.
