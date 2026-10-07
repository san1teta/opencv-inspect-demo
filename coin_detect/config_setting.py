gaussian_kernel = 5
canny_low = 50
canny_high = 150
min_area = 100

ppm = 10.0  #待测定
calib_image_size = (0, 0)  # 标定所用图像的(宽,高)；(0,0)=未标定。测量图必须与此一致，否则ppm不适用

min_circularity = 0.9  #1是正圆，过/不过
grade_light = 0.3  #刀口1，0 < 超出量 ≤ 0.3 → 轻度
grade_mid = 0.6  #刀口2，0.3 < 超出量 ≤ 0.6 → 中度；else重度

coin_specs = {
    '1元':{'diameter': 25.0, 'tol': 1.25},  #允差计算25*5%
    '5角':{'diameter': 20.5, 'tol': 1.03},  #20.5*5%
    '1角':{'diameter': 19.0, 'tol': 0.95},  #19,0*5%
}

