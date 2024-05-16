import cv2
import depthai as dai
import time
import numpy as np


num_classes = 20
source = 'rgb'
nn_path = 'blob/model.blob'
shape = 256

def decode_model(output_tensor):
    output = output_tensor.reshape(shape,shape)

    output = np.array(output) * (255/num_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    output_colors[output == 0] = [0,0,0]

    return output_colors

def show_model(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.4,0)




pipe = dai.Pipeline()

pipe.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2022_1)


detect_nn = pipe.create(dai.node.NeuralNetwork)
detect_nn.setBlobPath(nn_path)

detect_nn.setNumPoolFrames(4)
detect_nn.input.setBlocking(False)
detect_nn.setNumInferenceThreads(2)

cam=None
if source == 'rgb':
    cam = pipe.create(dai.node.ColorCamera)
    cam.setPreviewSize(shape,shape)
    cam.setInterleaved(False)
    cam.preview.link(detect_nn.input)
elif source == 'left':
    cam = pipe.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif source == 'right':
    cam = pipe.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if source != 'rgb':
    manip = pipe.create(dai.node.ImageManip)
    manip.setResize(shape,shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detect_nn.input)

cam.setFps(40)

xout_rgb = pipe.create(dai.node.XLinkOut)
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detect_nn.passthrough.link(xout_rgb.input)

xout_nn = pipe.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detect_nn.out.link(xout_nn.input)

with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if source != "rgb" and not depth_enabled:
        raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(source, cams))
    device.startPipeline(pipe)

    
    q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        
        in_nn_input = q_nn_input.get()
        in_nn = q_nn.get()

        frame = in_nn_input.getCvFrame()

        layers = in_nn.getAllLayers()

        
        lay1 = np.array(in_nn.getFirstLayerInt32()).reshape(shape,shape)

        found_classes = np.unique(lay1)
        output_colors = decode_model(lay1)

        frame = show_model(output_colors, frame)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        
        cv2.imshow("Depth Segmentation", frame)

        counter+=1
        if (time.time() - start_time) > 1 :
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            break