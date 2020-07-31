import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('room_size', 800, 'size of the square room in feet')
flags.DEFINE_string('output_track', None, 'path to the output track video')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    #location_transforamtion quad_coordinates
    quad_coords = {
    "target": np.array([
        [FLAGS.room_size-10, 0], #top right
        [0, 0], # top left
        [0, FLAGS.room_size-10], # bottom left
        [FLAGS.room_size-10, FLAGS.room_size-10] #  bottom right
    ]),
    "source": np.array([
        [1900, 10], # top right
        [800, 50], #  top left
        [0, 480], #  bottom left
        [1900, 1100] # bottom right
    ])}

    #perspective transform 
    M = cv2.getPerspectiveTransform(np.float32(quad_coords["source"]),np.float32(quad_coords["target"]))
    
    #initialize deep sortß
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    gender_list = ['Male', 'Female']
    logging.info('gender list loaded')

    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    logging.info("gender model loaded")
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        out_track_pad = cv2.VideoWriter(FLAGS.output_track, codec, fps, (FLAGS.room_size, FLAGS.room_size))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    
    fps = 0.0
    count = 0 
    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        # cv2.circle(img, (quad_coords["source"][0][0], quad_coords['source'][0][1]), radius=3, color=(0, 0, 0), thickness=5)
        # cv2.circle(img, (quad_coords["source"][1][0], quad_coords['source'][1][1]), radius=3, color=(0, 0, 0), thickness=5)
        # cv2.circle(img, (quad_coords["source"][2][0], quad_coords['source'][2][1]), radius=3, color=(0, 0, 0), thickness=5)
        # cv2.circle(img, (quad_coords["source"][3][0], quad_coords['source'][3][1]), radius=3, color=(0, 0, 0), thickness=5)

        pts = quad_coords['source'].reshape(-1, 1, 2)
        cv2.polylines(img,[pts],True,(0,255,255))

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    
        for (x, y, w, h )in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            # Get Face 
            face_img = img[y:y+h, h:h+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
    
            overlay_text = "%s" % (gender)
            cv2.putText(img, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        track_pad = 255*np.ones((FLAGS.room_size, FLAGS.room_size, 3), np.uint8)

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)
        #track_pad = transform_images(track_pad, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        #classes = [classes[index_of_person],]
        #print(classes)
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        #names = [names[names.index("person")],]
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features) if "person" in class_name]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            source = np.array([int(bbox[0] + (bbox[2]-bbox[0])//2), int(bbox[1] + (bbox[3]-bbox[1])//2)]).reshape(1,2)
            #source = np.array([int(bbox[2]), int(bbox[3])]).reshape(1,2)
            source = np.concatenate([source, np.ones((source.shape[0],1))], axis=1)
            target = np.dot(M,source.T)
            target = (target[:2,:]/target[2,:]).T
            centroid_x = int(target[0][0])
            centroid_y = int(target[0][1])
            cv2.circle(track_pad, (centroid_x, centroid_y), radius=3, color=color, thickness=5)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cv2.putText(track_pad, str(track.track_id),(centroid_x, centroid_y-20),0, 0.75, (0, 0, 0), 2)
            #cv2.putText(track_pad, "("+str(centroid_x)+","+str(centroid_y)+")", (centroid_x+10, centroid_y), 0, 0.5, color, 2)
            #img, ((int((bbox[2] -bbox[0])/2), (int((bbox[3] - bbox[1])/2))))
            
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        # for det in detections:
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
        #                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        #img = cv2.resize(img, (1000, 1000))
        cv2.imshow('output', img)
        cv2.imshow('track pad', track_pad)
        if FLAGS.output:
            out.write(img)
            out_track_pad.write(track_pad)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
        out_track_pad.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
