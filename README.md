You can find the code for the project I developed on real-time drone tracking using a camera in the attached files. To briefly explain the files:

In the Training.py file, you can train a model using YOLOv11 with your own images. I collected the dataset I used for training from Kaggle. You can train the model not only on drones but also on various objects and make small adjustments to the code to develop your own system.

With the code in Real_Time_Drone_Tracking.py, you can perform real-time object tracking. If your system is compatible with CUDA, it will run with CUDA; otherwise, it will default to CPU. As long as the required libraries are installed, the code should run without any issues. Additionally, you can observe performance variations by applying different real-time image filters.

The Video_Test.py file allows you to test the model on a pre-recorded video instead of real-time tracking.

The project is still under development, and I will continue to make updates.
