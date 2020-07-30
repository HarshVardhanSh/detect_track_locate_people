# Person detection,tracking and locating on the map of floor using YOLOv3 and Deep Sort

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you haven't set it up already)
```bash
# Ubuntu 18.04
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
### Downloading official pretrained weights
For Linux: Let's download official yolov3 weights pretrained on COCO dataset. 

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
```
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and yolov3-tiny [here](https://pjreddie.com/media/files/yolov3-tiny.weights) then save them to the weights folder.
  
### Saving yolov3 weights as a TensorFlow model.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .tf model files!

```
# yolov3
python load_weights.py

# yolov3-tiny
python load_weights.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf --tiny
```

## Running the Object Tracker
Now you can run the object tracker for whichever model you have created, pretrained, tiny, or custom.
```
# yolov3 on video
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi

#yolov3 on webcam 
python object_tracker.py --video 0 --output ./data/video/results.avi

#yolov3-tiny 
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi --weights ./weights/yolov3-tiny.tf --tiny
```

## Command Line Args Reference
```
load_weights.py:
  --output: path to output
    (default: './weights/yolov3.tf')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './weights/yolov3.weights')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)
    
object_tracker.py:
  --room_size: size of the square room (an integer)
  --classes: path to classes file
    (default: './data/labels/coco.names')
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './weights/yolov3.tf')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)
  --yolo_max_boxes: maximum number of detections at one time
    (default: '100')
    (an integer)
  --yolo_iou_threshold: iou threshold for how close two boxes can be before they are detected as one box
    (default: 0.5)
    (a float)
  --yolo_score_threshold: score threshold for confidence level in detection for detection to count
    (default: 0.5)
    (a float)
```

## Acknowledgments
* thAIGuy's https://github.com/theAIGuysCode/yolov3_deepsort.git
* [Yolov3 TensorFlow Amazing Implementation](https://github.com/zzh8829/yolov3-tf2)
* [Deep SORT Repository](https://github.com/nwojke/deep_sort)
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
