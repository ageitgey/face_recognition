@echo off
echo Make sure you've installed the Python 3.6 version of Anaconda: https://www.anaconda.com/download/
conda install -c conda-forge dlib numpy Pillow
conda install scipy Click
pip install -r requirements_windows.txt
pip install -U --no-deps .