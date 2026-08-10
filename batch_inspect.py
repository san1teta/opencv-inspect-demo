"""
batch_inspect.py - 批量图片检测演示程序
功能:批量读取图片 → 灰度/滤波 → Canny边缘检测 → 轮廓查找 → 特征计算与标注
流程:读图 → 预处理 → 边缘检测 → 轮廓筛选 → 绘制外接矩形/最小外接圆 → 保存结果
适用:OpenCV 基础视觉流程整合演示
"""
#导入必要的库
import cv2
import os
import matplotlib.pyplot as plt
import glob

#定位寻址
img_folder = os.path.join("D:\\", "vscode_workplace", "pre_pics")
result_folder = os.path.join("D:\\", "vscode_workplace", "pre_pics", "result")
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
print(f"共找到{len(img_paths)}张图片,开始处理")
for img_path in img_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue

#预处理四步
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)

#轮廓过滤
    min_area = 100
    script = rgb.copy()
    contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

#绘制外接矩形和最小外接圆
        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt, True)
        
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(script, (x,y), (x+w, y+h), (0,255,0), 2)

        (cx,cy),radius = cv2.minEnclosingCircle(cnt)
        center = (int(cx), int(cy))
        radius = int(radius)
        cv2.circle(script, center, radius, (255,0,0), 2)

#添加文本
        text = f"area:{area:2.2f};perimeter:{perimeter:2.2f}"
        cv2.putText(script, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

#展示对比结果
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(canny, cmap='gray')
    axes[0, 1].set_title('Canny Edge Detection')
    axes[1, 0].imshow(script)
    axes[1, 0].set_title('Contours and Shapes')

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()

#保存图片
    compare_path = os.path.join(result_folder, f"对比_{os.path.basename(img_path)}")
    plt.savefig(compare_path)
    plt.close()
    print(f"处理完成: {compare_path}")
print("处理完成,所有图片已保存到结果文件夹中")
