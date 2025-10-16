FROM ros:jazzy-ros-base
ENV PYTHONUNBUFFERED 1

# --- System-Pakete ---
RUN apt update && apt install -y \
    python3-pip python3-venv python3-opencv python3-numpy python3-yaml \
    ros-jazzy-cv-bridge ros-jazzy-image-transport ros-jazzy-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# X-Forwarding
# RUN apt install -y libgl1-mesa-glx libx11-6

# --- Python Abhängigkeiten + venv ---
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- Projektcode ---
WORKDIR /app

# --- ROS2 Umgebung ---
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# ENTRYPOINT ["bash", "-c", "source /opt/ros/jazzy/setup.bash && python3 live-demo/demo.py"]
CMD [ "tail", "-f", "/dev/null"]