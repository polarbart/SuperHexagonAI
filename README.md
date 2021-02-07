# Super Hexagon AI

 - Based on reinforcement learning (Distributional Q-Learning [1] with Noisy Nets [2])
 - Beats all six levels easily end to end (i. e. from pixel input to action output)
   - A level counts as beaten if one survives at least 60 seconds
 - Fast C++ implementation that hooks into the game and hands the frames over to the python process

## Example
<a href="https://youtu.be/gjqSZ_4mQjg" target="_blank"> <img src="images/thumbnail+controls.png"> </a>

## Results

 - The graphs below show the performance of the AI
 - The AI was trained for roughly 1.5 days resulting in roughly 9 days of in-game time
 - The AI quickly learns to complete all levels, after a few hours of in-game time
   - A level counts as completed if the player survives for 60+ seconds
 
<div align="center">
<img src="images/smoothed_time_alive_2.png" width="100%">
<br>
<img src="images/highscores_2.png" width="100%">
</div>

### Glitches
The AI discovered that it can glitch through a wall in some occurrences. 

The left glitch was also discovered by <a href="https://youtu.be/CI04x9au_Es" target="_blank">others</a>.
The right glitch may only be possible when rotation is disabled.

<div align="center">
<img src="images/a_compressed.gif" width="47%">
&nbsp;
<img src="images/b_compressed2.gif" width="47%">
</div>

## Implementational Details

### Training
 - The reinforcement learning agent receives a reward of -1 if it dies otherwise a reward of 0
 - The network has two convolutional streams
   - For the first stream the frame is cropped to a square and resized to 60 by 60 pixel
   - Since the small triangle which is controlled by the player is barely visible for the first stream,
     a zoomed-in input is given to the second stream
   - For the fully connected part of the network the feature maps of both streams are flattened and concatenated
   - See `utils.Network` for more implementational details
 - Additionally, a threshold function is applied to the frame
     such that the walls and the player belong to the foreground and everything else belongs to the background
   - See `superhexagon.SuperHexagonInterface._preprocess_frame` for more implementational details
 - The used hyperparameters can be found at the bottom of `trainer.py` below `if __name__ == '__main__':`
 
 
### Rainbow
 - All six Rainbow [3] extensions have been evaluated
 - Double Q-Learning [4] and Dueling Networks [5] did not improve the performance
 - n-step significantly decreased the performance
 - Prioritized experience replay [6] at first performs better, 
   however, after roughly 300,000 training steps the agent trained without prioritized experience replay performs better
 - The distributional approach [1] significantly increases the performance of the agent
   - Distributional RL with quantile regression [7] gives similar results
 - Noisy networks [2] facilitate the exploration process
   - However, the noise is turned off after 500,000 training iterations

### PyRLHook
In order to efficiently train the agent, a C++ library was written. This library serves two functions:
Firstly, it efficiently retrieves the frames and sends them to the python process. 
Secondly, it intercepts the system calls used to get the system time such that the game can be run at a desired speed.

To do so, the library injects a DLL into the game's process.
This DLL hooks into the OpenGL function `wglSwapBuffers` as well as the system calls `timeGetTime`, `GetTickCount`, `GetTickCount64`, and `RtlQueryPerformanceCounter`.

The function `wglSwapBuffers` is called every time the game finishes rendering a frame in order to swap the back and front buffer. 
The `wglSwapBuffers` hook first locks the games execution. 
If one wants to advance the game for one step and retrieve the next frame `GameInterface.step` can be called from python.
The this releases the lock until `wglSwapBuffers` is called again. 
Then the back buffer is copied into a shared memory space in order to be returned by `GameInterface.step`.

The time perceived by the game can be adjusted with the methods `GameInterface.set_speed(double factor)` and `GameInterface.run_afap(double fps)`. 
`set_speed` adjusts the perceived time by the given factor. I. e. if the factor is `0.5` the game runs at half the speed. 
`run_afap` makes the game think it runs with the specified FPS i. e. the current time is incremented by `1/fps` every time `GameInterface.step` is called.

For more implementational details see `RLHookLib/PyRLHook/GameInterface.cpp` (especially the constructor and the method `step`) 
as well as `RLHookLib/RLHookDLL/GameInterfaceDll.cpp` (especially the methods `onAttach`, `initFromRenderThread`, and `hWglSwapBuffers`).

In theory, this library should also work for other games written with OpenGL.

## Usage

The following python libraries are required:

```
numpy
pytorch
opencv-python
```

Note that both the python process as well as the game process need to be run with admin privileges.
In order to always run the game with admin privileges right click on the Super Hexagon executable `superhexagon.exe`, 
select `Properties` and within the `Compatability` tab check `Run this program as an administrator`.

The game should be in windowed mode and VSync should be disabled.

Additionally, you need to compile the C++ library. 
Make sure you have a Visual Studio compiler (MSVC C++ x64/x86 build tools) installed, 
including CMake (C++ CMake tools for Windows) and a Windows 10 SDK. 
Also make sure that CMake can be called from the commandline.

Clone the repository.
```
git clone https://github.com/polarbart/SuperHexagonAI.git
cd SuperHexagonAI
```

Compile the DLL as well as a helper executable.
```
cd RLHookLib
python compile_additional_binaries.py
```

Then you can install the library globally using pip.

```
pip install .
```

### Evaluation
In order to run the AI, first download the pretrained network [super_hexagon_net](https://github.com/polarbart/SuperHexagonAI/releases/tag/v1.0)
and place it in the main folder (i. e. the folder where `eval.py` is located).

Then, start the game in windowed mode and execute the `eval.py` script, both with admin privileges.

The level being played as well as other parameters can be adjusted within the script.

### Training
In order to train your own AI run `trainer.py` with admin privileges. 

Make sure that the game is in windowed mode and VSync is disabled.

Note that the AI is trained on all six levels simultaneously and that you do not need to start the game manually, 
since the script starts the game automatically. 
Please adjust the path to the Super Hexagon executable at the bottom of the trainer script if necessary 
and make sure that the game is always run with admin privileges as described above.

Since sometimes the game gets stuck within one level, the script will sometimes automatically restart the game. 
Therefore, you may want to disable the message box asking for admin privileges.


## Other Peoples Approaches
This person (<a href="http://cs231n.stanford.edu/reports/2016/pdfs/115_Report.pdf" target="_blank">pdf</a>) first takes a screenshot of the game.
Then they used a CNN in order to detect the walls and the player. 
This information is then used by a hand-crafted decision maker in order to select an action.

This person (<a href="https://github.com/adrianchifor/super-hexagon-ai" target="_blank">github</a>) reads the game's state from the game's memory. 
Then they write the correct player position into the game's memory. 

This person (<a href="https://crackedopenmind.com/portfolio/super-hexagon-bot/" target="_blank">crackedopenmind.com</a>) 
also retrieves the frames by hooking `SwapBuffers`. 
Then they analyze the frames with OpenCV and use a hand-crafted decision maker in order to select an action.

Let me know if you find any other approaches, so i can add them here :)

# References
[1] Bellemare, Marc G., Will Dabney, and Rémi Munos. "A distributional perspective on reinforcement learning." arXiv preprint arXiv:1707.06887 (2017).

[2] Fortunato, Meire, et al. "Noisy networks for exploration." arXiv preprint arXiv:1706.10295 (2017).

[3] Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

[4] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.

[5] Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.

[6] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

[7] Dabney, Will, et al. "Distributional reinforcement learning with quantile regression." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.