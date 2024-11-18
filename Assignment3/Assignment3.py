#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net=jetson.inference.detectNet("ssd-mobilenet-v2",threshold=0.5) #load model
img = jetson.utils.loadImage("24c9c612c593f371.jpg")
#camera=jetson.utils.videoSource("/dev/video0") #opening the camera stream

#display=jetson.utils.videoOutput("display://0") #display loop
#while display.IsStreaming(): #main llop will go here
#  img = camera.Capture()
#  if img is None:
#    continue
#  display.Render(img)
#  display.SetStatus("Object Detection | Network {:.0f}FPS".format(net.GetNetworkFPS()))


from PIL import Image, ImageDraw, ImageFont

detections = net.Detect(img)

cuda_img = jetson.utils.cudaToNumpy(img)
pillow_img = Image.fromarray(cuda_img.astype('uint8'))

draw = ImageDraw.Draw(pillow_img)
font = ImageFont.load_default()  

for detection in detections:
    left, top, right, bottom = detection.Left, detection.Top, detection.Right, detection.Bottom
    class_id = detection.ClassID
    label = net.GetClassDesc(class_id)
    print(detection)

    draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)

    draw.text((left, top - 20), label, fill="green", font=font)


output_path = 'output.jpg' 
pillow_img.save(output_path)

