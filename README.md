# PhotoTool_v2.0
活动会议照片快速分类筛选工具
# 📸(会议摄影智能初筛工具)

专为会议、活动摄影师打造的本地 AI 选片助手。利用深度学习和计算机视觉技术，直接读取相机 RAW 文件，自动识别人物闭眼、曝光异常和对焦模糊的废片，极大提升后期挑片效率。

## ✨ 核心特性

* **📂 全面支持 RAW 格式**：基于 `rawpy`，直接支持解析 `.cr2`, `.cr3` (Canon), `.arw` (Sony), `.nef` (Nikon), `.rw2`, `.raf`, `.dng` 等主流 RAW 格式及标准图片。
* **👤 强悍的人脸检测**：集成 `YOLOv8-Face`，在复杂的会议合影、暗光环境下依然能保持极高的检出率。
* **👁️ 精准的闭眼识别**：基于 `MediaPipe Face Landmarker` 提取面部特征点，通过计算眼睛纵横比（EAR，Eye Aspect Ratio）精准判断是否处于闭眼/眨眼状态。
* **📊 综合图像质量评估**：
    * **模糊检测**：使用拉普拉斯算子方差 (Laplacian Variance) 评估画面锐度。
    * **曝光检测**：分析灰度直方图，计算过曝/死黑溢出比例，识别曝光失误。
* **🤖 智能分类系统**：工具会综合“可用性得分(usable score)”，自动将照片输出至四个目录：`reject` (废片/闭眼), `review` (待复查), `keep` (可用), `best` (优选)。
* **🖼️ 可视化标注诊断**：支持导出带有 Bounding Box、EAR 数值、以及判断理由的诊断预览图，方便排查算法判定原因。

## 🛠️ 安装指南

1. **克隆仓库**
   ```bash
   git clone [https://github.com/yourusername/smart-photo-culler.git](https://github.com/yourusername/smart-photo-culler.git)
   cd smart-photo-culler
