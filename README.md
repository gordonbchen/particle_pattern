# particle_pattern
Creating patterns with particle simulations.

Simulates 2D elastic collisions between equal-mass and equal-sized particles. Particle positions can be initialized by drawing, from the edges of a given image (convolutional edge detector), or randomly. Simulation can be run in interactive game mode, where the user can reverse time and pause, or as an animation creator. 

Inspired by [MarbleScience Message Defying Entropy](https://marblescience.com/message-defying-entropy).

## Basic Usage
* Requires `python3` and `poetry`
* Install requirements and activate shell with `poetry`
* Run `python3 particle_pattern/run_sim.py`. This will start up a `pygame` window allowing you to place particles by drawing. Hit ENTER when done. An animation of seemingly randomly placed particles colliding and making your drawing will be saved to `outputs`.
* For other options and more advanced usage, run `python3 particle_pattern/run_sim.py --help` and read the code.

## Sources
* [MarbleScience Instagram Reel](https://www.instagram.com/marblescience/reel/DDfB63stC4x/)
* [MarbleScience Message Defying Entropy](https://marblescience.com/message-defying-entropy)
* [If Feynman Were Teaching Today… A Simplified Python Simulation of Diffusion: The Python Coding Stack](https://www.thepythoncodingstack.com/p/python-diffusion-simulation-demo-turtle)
* [2D Elastic Collisions: Physics LibreTexts](https://phys.libretexts.org/Bookshelves/Classical_Mechanics/Classical_Mechanics_(Dourmashkin)/15%3A_Collision_Theory/15.06%3A_Two_Dimensional_Elastic_Collisions)
