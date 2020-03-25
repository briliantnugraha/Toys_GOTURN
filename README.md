# Toys_GOTURN
 This contains implementation of customized goturn with (20fps or ~50ms speed on gtx1080).

## The purpose of it is to detect:
- Shifting of the previous rectangle
- Redraw the current location of the shifting rectangle

## The assumption of it is that:
- We have the current center rectangle location (mask_inp) and previous rectangle location (mask_gt)

## How To Use it:
- Open and inspect main_train.py
- If you want to train, then uncomment the adjust_mask.train
- If you want to detect, then just run the code with python3 main_train.py

## Dependencies
- [x] Pytorch 1.3 ++
- [x] time
- [x] Matplotlib
- [x] Numpy
- [x] PIL
- [x] os
- [x] CV2

## Contact ME
For further info, please use issues or contact me directly through issues, thank you.

## Task List
- [x] Build Customized GoTurn with Resnext101
- [x] Build random point location of rectangle as dataset
- [x] Validate training and testing
- [x] Push on github
