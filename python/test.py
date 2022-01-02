import video_functions as vid

vidName = '../data/videoCaptureTest1.avi'
channel = 0
width = 848
height = 480

inArgs = (vidName, channel, width,  height)
frameTensor = vid.genVideoTensor(inArgs)

idx1 = 100
idx2 = 160
delta = frameTensor[:,:, idx1] - frameTensor[:,:, idx2]

print(delta.max())