# Toys_GOTURN
 This contains implementation of customized goturn with (20fps or ~50ms speed on gtx1080).

The purpose of it is to detect:
- Shifting of the previous rectangle
- Redraw the current location of the shifting rectangle

The assumption of it is that:
- We have the current center rectangle location (mask_inp) and previous rectangle location (mask_gt)